#!/usr/bin/env python3
"""
Circuit analysis using EAP-IG (Edge Attribution Patching with Integrated Gradients).

Matches the approach in https://github.com/hannamw/EAP-IG (ioi.ipynb):
- transformer_lens HookedTransformer
- eap.graph.Graph, eap.attribute.attribute, eap.evaluate
- Gradient-based edge attribution (EAP or EAP-IG-inputs)
- Graph-based circuit extraction via apply_topn()
- For "correct vs incorrect" generalization: labels as [correct_idx, incorrect_idx] and
  logit_diff metric (correct - incorrect), as in the notebook.

Requires:
  pip install transformer_lens
  pip install git+https://github.com/hannamw/EAP-IG   # eap package (circuit analysis)
  # Note: PyPI "eap" is a different package. Use the EAP-IG repo.
Optional: pip install pygraphviz  (for circuit graph visualization)

Supported models: gpt2-small, gpt2-medium, meta-llama/Llama-3.1-8B, etc. (transformer_lens compatible).
Default matches run_all_circuits.py: meta-llama/Llama-3.1-8B.

Output: C_arith.json, eap_graph.json, analysis_output/ compatible with rest of pipeline.
"""
# Force CPU before torch imports when --device cpu (must run before importing torch)
import sys
import os
if "--device" in sys.argv:
    try:
        i = sys.argv.index("--device")
        if i + 1 < len(sys.argv) and sys.argv[i + 1].lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except (ValueError, IndexError):
        pass

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Check dependencies
try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

try:
    from eap.graph import Graph
    from eap.evaluate import evaluate_graph, evaluate_baseline
    from eap.attribute import attribute
except ImportError:
    Graph = attribute = evaluate_graph = evaluate_baseline = None

from eval_counterfactual import (
    load_counterfactual_dataset,
    COUNTERFACTUAL_DATASETS,
)


def collate_eap(xs):
    """
    Collate for EAP: (clean, corrupted, labels).
    Labels: list of [correct_idx, incorrect_idx] per sample (for logit_diff)
    or list of single token ids (for log_prob metric). Collate stacks to tensor
    of shape (batch, 2) or (batch,) so the metric can use torch.gather(..., labels).
    """
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels, dtype=torch.long)  # (batch,) or (batch, 2)
    return clean, corrupted, labels


class ArithmeticEAPDataset(Dataset):
    """
    Adapts counterfactual arithmetic data to EAP format (aligned with ioi.ipynb).
    clean = x, corrupted = x_prime.
    If use_logit_diff: label = [correct_idx, incorrect_idx] (token ids for y and y_prime).
    Else: label = token id of correct answer (y) only.
    """

    def __init__(self, items, tokenizer, use_logit_diff: bool = True, filter_length_mismatch: bool = True):
        self.tokenizer = tokenizer
        self.use_logit_diff = use_logit_diff
        self._label_cache = {}
        # EAP-IG requires clean/corrupted to have same token count; filter mismatched pairs
        if filter_length_mismatch:
            filtered = []
            for item in items:
                clean_ids = tokenizer.encode(str(item["x"]).strip(), add_special_tokens=False)
                corrupt_ids = tokenizer.encode(str(item["x_prime"]).strip(), add_special_tokens=False)
                if len(clean_ids) == len(corrupt_ids):
                    filtered.append(item)
                # else: skip (e.g. Italian "meno" vs "più" tokenize to different lengths)
            self.items = filtered
            if filtered and len(filtered) < len(items):
                print(f"Filtered {len(items) - len(filtered)} samples with clean/corrupted length mismatch")
        else:
            self.items = items

    def __len__(self):
        return len(self.items)

    def _token_to_id(self, text: str):
        if not (text or str(text).strip()):
            return 0
        ids = self.tokenizer.encode(str(text).strip(), add_special_tokens=False)
        return ids[0] if ids else 0

    def _get_label(self, idx):
        item = self.items[idx]
        y = item.get('y', item.get('y_expected', '')).strip()
        y_prime = item.get('y_prime', item.get('y_prime_expected', '')).strip()
        correct_id = self._token_to_id(y)
        incorrect_id = self._token_to_id(y_prime)
        if self.use_logit_diff:
            return [correct_id, incorrect_id]
        return correct_id

    def __getitem__(self, idx):
        item = self.items[idx]
        clean = item['x']
        corrupted = item['x_prime']
        label = self._get_label(idx)
        return clean, corrupted, label

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_eap, shuffle=False)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    """Extract logits at last token position (where we predict the answer)."""
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    return logits[idx, input_length - 1]


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    loss=False,
):
    """
    Metric: (logit correct - logit incorrect) at last position.
    As in ioi.ipynb: good_bad = gather(logits, labels); results = good_bad[:, 0] - good_bad[:, 1].
    Higher = better (model prefers correct over incorrect). For loss (minimize), use loss=True.
    labels: (batch, 2) with [correct_idx, incorrect_idx] per row.
    """
    logits = get_logit_positions(logits, input_length)
    labels = labels.long().to(logits.device)
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1).expand(-1, 2)
    good_bad = torch.gather(logits, -1, labels)
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def get_arithmetic_metric(tokenizer, use_logit_diff: bool = True):
    """
    Return metric for EAP. If use_logit_diff: logit_diff (correct - incorrect), as in ioi.ipynb.
    Else: log prob of correct answer token at last position.
    Higher = better. For loss (minimize), use loss=True.
    """

    if use_logit_diff:
        return logit_diff

    def arithmetic_metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,
        mean=True,
        loss=False,
    ):
        logits = get_logit_positions(logits, input_length)
        log_probs = torch.log_softmax(logits, dim=-1)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        labels = labels.long().to(logits.device)
        if labels.dim() == 2:
            labels = labels[:, 0]
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        correct_log_probs = log_probs[batch_idx, labels]
        if loss:
            correct_log_probs = -correct_log_probs
        if mean:
            return correct_log_probs.mean()
        return correct_log_probs

    return arithmetic_metric


def run_eap_ig_circuit_analysis(
    model_name: str,
    dataset_type: str,
    split: str = 'test',
    max_samples: int = 100,
    batch_size: int = 10,
    method: str = 'EAP-IG-inputs',
    ig_steps: int = 5,
    top_n: int = 20000,
    output_dir: str = 'analysis_output',
    device: str = None,
    use_logit_diff: bool = True,
):
    """
    Run EAP-IG circuit analysis and save results.

    Returns:
        dict with circuit, graph info, baseline/circuit performance
    """
    if HookedTransformer is None:
        raise ImportError("Install transformer_lens: pip install transformer_lens")
    if Graph is None or attribute is None:
        raise ImportError("Install eap: pip install eap")

    output_path = Path(output_dir) / dataset_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load dataset
    dataset_path = COUNTERFACTUAL_DATASETS.get(dataset_type)
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    items = load_counterfactual_dataset(dataset_path, split, max_samples)
    if not items:
        raise ValueError("No samples loaded")

    # Load model (Llama 3 8B: use notebook settings from ioi.ipynb)
    print(f"Loading {model_name} as HookedTransformer...")
    load_kw = dict(device=device)
    if 'llama' in model_name.lower() or 'Llama' in model_name or 'meta-llama' in model_name:
        load_kw.update(
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            dtype=torch.float16 if device != 'cpu' else torch.float32,
        )
    model = HookedTransformer.from_pretrained(model_name, **load_kw)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, 'ungroup_grouped_query_attention'):
        model.cfg.ungroup_grouped_query_attention = True

    # Build dataloader and metric (logit_diff for correct vs incorrect generalization)
    ds = ArithmeticEAPDataset(items, model.tokenizer, use_logit_diff=use_logit_diff)
    if len(ds) == 0:
        raise ValueError(
            f"No samples with matching clean/corrupted token lengths for {dataset_type}. "
            "EAP-IG requires aligned sequences (e.g. Italian 'meno' vs 'più' can tokenize differently)."
        )
    dataloader = ds.to_dataloader(batch_size)
    metric = get_arithmetic_metric(model.tokenizer, use_logit_diff=use_logit_diff)

    # Build graph and run attribution
    print(f"Building graph and running {method}...")
    g = Graph.from_model(model)
    attr_kw = {"method": method}
    if "IG" in method or "ig" in method.lower():
        attr_kw["ig_steps"] = ig_steps
    attribute(model, g, dataloader, partial(metric, loss=True, mean=True), **attr_kw)

    # Extract circuit
    g.apply_topn(top_n, True)
    n_nodes, n_edges = g.count_included_nodes(), g.count_included_edges()
    print(f"Circuit: {n_nodes} nodes, {n_edges} edges")

    # Evaluate
    baseline = evaluate_baseline(
        model, dataloader,
        partial(metric, loss=False, mean=False),
    ).mean().item()
    circuit_perf = evaluate_graph(
        model, g, dataloader,
        partial(metric, loss=False, mean=False),
    ).mean().item()
    print(f"Baseline (clean) metric: {baseline:.4f}")
    print(f"Circuit metric: {circuit_perf:.4f}")

    # Export graph (JSON for portability; .pt if available, as in ioi.ipynb)
    graph_json_path = output_path / 'eap_graph.json'
    g.to_json(str(graph_json_path))
    print(f"Saved graph to {graph_json_path}")
    if hasattr(g, 'to_pt'):
        pt_path = output_path / 'eap_graph.pt'
        g.to_pt(str(pt_path))
        print(f"Saved graph to {pt_path}")

    # Convert to C_arith format (layer-level from graph nodes)
    circuit_nodes = _extract_circuit_layers(g, graph_json_path)
    c_arith = {
        "method": method,
        "dataset": dataset_type,
        "use_logit_diff": use_logit_diff,
        "per_format": {dataset_type: circuit_nodes},
        "combined": circuit_nodes,
        "top_n_nodes": top_n,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "baseline_metric": baseline,
        "circuit_metric": circuit_perf,
    }
    c_arith_path = output_path / 'C_arith.json'
    with open(c_arith_path, 'w') as f:
        json.dump(c_arith, f, indent=2)
    print(f"Saved C_arith to {c_arith_path}")

    # Optional: graph image
    try:
        import pygraphviz
        img_path = output_path / 'eap_circuit_graph.png'
        g.to_image(str(img_path))
        print(f"Saved circuit visualization to {img_path}")
    except ImportError:
        print("Install pygraphviz for circuit visualization: pip install pygraphviz")
    except ValueError as e:
        # Log diagnostics when viz fails (e.g. "Number of positions must match")
        import traceback
        print(f"Graph image failed ({dataset_type}): {e}")
        print(f"  Circuit stats: {n_nodes} nodes, {n_edges} edges")
        nodes = getattr(g, "nodes", None)
        if isinstance(nodes, dict):
            names = list(nodes.keys())[:20]
            print(f"  Sample node names: {names}")
        traceback.print_exc()

    return c_arith


def _node_name_to_layer(name: str) -> str:
    """
    Map EAP/TransformerLens node names to model.layers.N for C_arith.
    EAP uses e.g. blocks.0.attn, blocks.0.mlp, blocks.0.hook_resid_pre -> model.layers.0.
    """
    if not name:
        return ""
    name = str(name).strip()
    if name.startswith("model.layers."):
        return name
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"model.layers.{parts[1]}"
    if "." in name:
        first = name.split(".")[0]
        second = name.split(".")[1] if len(name.split(".")) > 1 else ""
        if first == "blocks" and second.isdigit():
            return f"model.layers.{second}"
    return name


def _extract_circuit_layers(g, graph_json_path: Path = None) -> list:
    """
    Extract layer/component names from EAP graph for C_arith compatibility.
    EAP uses names like blocks.N.attn - we collect unique layer names model.layers.N
    (and raw node names) so downstream (Grad-CAM, viz) can identify relevant layers.
    """
    raw_names = []
    # Prefer graph internals: g.nodes is a Dict[str, Node] with .in_graph flag
    nodes = getattr(g, "nodes", None)
    if isinstance(nodes, dict):
        for name, node in nodes.items():
            try:
                in_graph = getattr(node, "in_graph", True)
            except Exception:
                in_graph = True
            if in_graph and name and name not in raw_names:
                raw_names.append(name)
    # Fallback: load from saved JSON
    if not raw_names and graph_json_path and graph_json_path.exists():
        with open(graph_json_path) as f:
            d = json.load(f)
        if isinstance(d, dict):
            for n in d.get('nodes', []) or []:
                name = n.get('name', n) if isinstance(n, dict) else str(n)
                if name and (not isinstance(n, dict) or n.get('included', True)):
                    raw_names.append(name)
    # Normalize to model.layers.N and dedupe, keep order
    layer_set = set()
    result = []
    for name in raw_names:
        layer = _node_name_to_layer(name)
        if layer and layer not in layer_set:
            layer_set.add(layer)
            result.append(layer)
    return result if result else raw_names


def main():
    parser = argparse.ArgumentParser(
        description='Circuit analysis using EAP-IG (same method as hannamw/EAP-IG)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B',
                        help='Model (e.g. meta-llama/Llama-3.1-8B, gpt2-small)')
    parser.add_argument('--dataset', type=str, default='numeric',
                        choices=list(COUNTERFACTUAL_DATASETS.keys()),
                        help='Dataset to analyze')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size (notebook uses 10)')
    parser.add_argument('--method', type=str, default='EAP-IG-inputs',
                        choices=['EAP', 'EAP-IG-inputs', 'clean-corrupted'],
                        help='Attribution method')
    parser.add_argument('--ig-steps', type=int, default=5,
                        help='IG steps for EAP-IG-inputs')
    parser.add_argument('--top-n', type=int, default=20000,
                        help='Top N edges for circuit (notebook uses 20000)')
    parser.add_argument('--no-logit-diff', action='store_true',
                        help='Use log-prob of correct token instead of logit_diff (correct - incorrect)')
    parser.add_argument('--output-dir', type=str, default='analysis_output')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, mps, cpu. Use cpu if you get "Torch not compiled with CUDA"')

    args = parser.parse_args()

    print("=" * 60)
    print("EAP-IG Circuit Analysis")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print()

    try:
        run_eap_ig_circuit_analysis(
            model_name=args.model,
            dataset_type=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            method=args.method,
            ig_steps=args.ig_steps,
            top_n=args.top_n,
            output_dir=args.output_dir,
            device=args.device,
            use_logit_diff=not args.no_logit_diff,
        )
        print("\nDone.")
    except ImportError as e:
        print("Missing dependency:", e)
        print("Install with: pip install transformer_lens eap")
        raise SystemExit(1)
    except Exception as e:
        print("Error:", e)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
