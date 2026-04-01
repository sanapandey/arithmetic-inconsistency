#!/usr/bin/env python3
"""
EAP-IG circuit analysis with IOI-style *frozen* label indices (ioi_llama.csv pattern).

Same pipeline as circuit_analysis_eap_ig.py (HookedTransformer, Graph, attribute, logit_diff,
evaluate_baseline / evaluate_graph, C_arith export). Difference:

- After the same filtering as the main script (length-aligned, optional model-correct), we
  write a CSV: clean, corrupted, correct_idx, incorrect_idx (+ audit columns). Rows are fixed
  integers for that tokenizer, matching Hanna W's EAP-IG IOI demo.
- The EAP Dataset reads only that CSV — labels are not recomputed in __getitem__.

Rebuild the CSV when you change model, tokenizer, filters, or counterfactual data.
Default cache path: data/eap_frozen/<stem>.csv (see default_frozen_stem).

Requires: transformer_lens, eap (EAP-IG repo), same as circuit_analysis_eap_ig.py.
"""
import argparse
import csv
import json
import sys
import os
if "--device" in sys.argv:
    try:
        i = sys.argv.index("--device")
        if i + 1 < len(sys.argv) and sys.argv[i + 1].lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except (ValueError, IndexError):
        pass

from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

from eval_counterfactual import (
    load_counterfactual_dataset,
    COUNTERFACTUAL_DATASETS,
)

from circuit_analysis_eap_ig import (
    collate_eap,
    continuation_first_token_id,
    filter_items_model_first_token_correct,
    filter_length_aligned_items,
    get_arithmetic_metric,
    print_label_token_debug,
    _extract_circuit_layers,
)

FROZEN_CSV_ROOT = Path(__file__).resolve().parent / "data" / "eap_frozen"

CSV_FIELDS = [
    "problem_id",
    "clean",
    "corrupted",
    "correct_idx",
    "incorrect_idx",
    "y",
    "y_prime",
]


def default_frozen_stem(
    model_name: str,
    dataset_type: str,
    split: str,
    max_samples: int,
    filter_model_correct: bool,
) -> str:
    safe_model = model_name.replace("/", "__").replace(" ", "_")
    filt = "model_correct" if filter_model_correct else "all_aligned"
    return f"{safe_model}__{dataset_type}__{split}__n{max_samples}__{filt}"


def _write_frozen_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_frozen_arithmetic_csv(
    items: list,
    tokenizer,
    csv_path: Path,
    *,
    model_name: str,
    dataset_type: str,
    split: str,
    max_samples_requested: int,
    filter_model_correct: bool,
    n_loaded: int,
) -> int:
    """
    Serialize filtered items to CSV. Each correct_idx/incorrect_idx is
    continuation_first_token_id for y / y_prime (same as circuit_analysis_eap_ig.py).
    Skips degenerate or invalid id rows.
    """
    rows_out = []
    skipped = {"bad_or_degenerate": 0}
    for item in items:
        y = item.get("y", item.get("y_expected", ""))
        yp = item.get("y_prime", item.get("y_prime_expected", ""))
        cid = continuation_first_token_id(tokenizer, y)
        iid = continuation_first_token_id(tokenizer, yp)
        if cid == 0 or iid == 0 or cid == iid:
            skipped["bad_or_degenerate"] += 1
            continue
        rows_out.append(
            {
                "problem_id": item.get("id", ""),
                "clean": str(item["x"]),
                "corrupted": str(item["x_prime"]),
                "correct_idx": cid,
                "incorrect_idx": iid,
                "y": str(y) if y is not None else "",
                "y_prime": str(yp) if yp is not None else "",
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows_out:
            w.writerow(
                {
                    "problem_id": row["problem_id"],
                    "clean": row["clean"],
                    "corrupted": row["corrupted"],
                    "correct_idx": row["correct_idx"],
                    "incorrect_idx": row["incorrect_idx"],
                    "y": row["y"],
                    "y_prime": row["y_prime"],
                }
            )

    meta_path = csv_path.with_suffix(".meta.json")
    _write_frozen_meta(
        meta_path,
        {
            "model_name": model_name,
            "dataset": dataset_type,
            "split": split,
            "max_samples_requested": max_samples_requested,
            "filter_model_correct": filter_model_correct,
            "n_loaded_from_json": n_loaded,
            "n_rows_written": len(rows_out),
            "n_skipped_bad_or_degenerate": skipped["bad_or_degenerate"],
            "label_rule": "continuation_first_token_id",
            "csv_path": str(csv_path.resolve()),
        },
    )
    print(
        f"Wrote frozen CSV: {csv_path} ({len(rows_out)} rows; "
        f"skipped bad/degenerate: {skipped['bad_or_degenerate']})"
    )
    return len(rows_out)


class FrozenArithmeticEAPDataset(Dataset):
    """IOI-style: rows from CSV with precomputed correct_idx / incorrect_idx."""

    def __init__(self, csv_path: Path, use_logit_diff: bool = True):
        self.use_logit_diff = use_logit_diff
        self.rows: List[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows in frozen CSV: {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        clean = row["clean"]
        corrupted = row["corrupted"]
        if self.use_logit_diff:
            label = [int(row["correct_idx"]), int(row["incorrect_idx"])]
        else:
            label = int(row["correct_idx"])
        return clean, corrupted, label

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_eap, shuffle=False)


def _warn_meta_mismatch(meta_path: Path, model_name: str) -> None:
    if not meta_path.is_file():
        return
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model_name") != model_name:
            print(
                f"WARNING: frozen CSV was built for {meta.get('model_name')!r} "
                f"but you are loading {model_name!r}. Indices may be wrong for this vocab."
            )
    except (json.JSONDecodeError, OSError):
        pass


def run_eap_ig_circuit_analysis_frozen(
    model_name: str,
    dataset_type: str,
    split: str = "test",
    max_samples: int = 100,
    batch_size: int = 10,
    method: str = "EAP-IG-inputs",
    ig_steps: int = 5,
    top_n: int = 20000,
    output_dir: str = "analysis_output",
    device: str = None,
    use_logit_diff: bool = True,
    filter_model_correct: bool = True,
    debug_label_tokenization: int = 0,
    frozen_csv_path: Optional[str] = None,
    rebuild_frozen_csv: bool = False,
    csv_only: bool = False,
):
    if HookedTransformer is None:
        raise ImportError("Install transformer_lens: pip install transformer_lens")
    if not csv_only:
        try:
            from eap.graph import Graph
            from eap.evaluate import evaluate_graph, evaluate_baseline
            from eap.attribute import attribute
        except ImportError:
            raise ImportError("Install eap: pip install git+https://github.com/hannamw/EAP-IG")

    output_path = Path(output_dir) / dataset_type
    output_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    dataset_path = COUNTERFACTUAL_DATASETS.get(dataset_type)
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    items = load_counterfactual_dataset(dataset_path, split, max_samples)
    if not items:
        raise ValueError("No samples loaded")
    n_loaded = len(items)

    print(f"Loading {model_name} as HookedTransformer...")
    load_kw = dict(device=device)
    if "llama" in model_name.lower() or "Llama" in model_name or "meta-llama" in model_name:
        load_kw.update(
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    model = HookedTransformer.from_pretrained(model_name, **load_kw)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = True

    tokenizer = model.tokenizer
    items = filter_length_aligned_items(items, tokenizer)
    if not items:
        raise ValueError(
            f"No samples with matching clean/corrupted token lengths for {dataset_type}. "
            "EAP-IG requires aligned sequences."
        )
    if debug_label_tokenization > 0:
        print_label_token_debug(model, tokenizer, items, debug_label_tokenization)
    if filter_model_correct:
        items = filter_items_model_first_token_correct(model, items, tokenizer)
    if not items:
        raise ValueError(
            f"No samples left after filtering for {dataset_type}. "
            "Try --no-filter-model-correct, or increase max_samples."
        )

    if frozen_csv_path:
        csv_path = Path(frozen_csv_path).expanduser()
    else:
        stem = default_frozen_stem(model_name, dataset_type, split, max_samples, filter_model_correct)
        FROZEN_CSV_ROOT.mkdir(parents=True, exist_ok=True)
        csv_path = FROZEN_CSV_ROOT / f"{stem}.csv"

    meta_path = csv_path.with_suffix(".meta.json")

    if rebuild_frozen_csv or not csv_path.is_file():
        build_frozen_arithmetic_csv(
            items,
            tokenizer,
            csv_path,
            model_name=model_name,
            dataset_type=dataset_type,
            split=split,
            max_samples_requested=max_samples,
            filter_model_correct=filter_model_correct,
            n_loaded=n_loaded,
        )
    else:
        print(f"Using existing frozen CSV: {csv_path}")
        _warn_meta_mismatch(meta_path, model_name)

    ds = FrozenArithmeticEAPDataset(csv_path, use_logit_diff=use_logit_diff)
    n_csv = len(ds)
    if n_csv == 0:
        raise ValueError("Frozen dataset is empty.")
    dataloader = ds.to_dataloader(batch_size)
    metric = get_arithmetic_metric(model.tokenizer, use_logit_diff=use_logit_diff)

    if csv_only:
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for i in range(len(ds)):
                clean_s, _, lab = ds[i]
                logits = model(clean_s)
                last = logits[0, -1]
                if use_logit_diff:
                    cid, iid = int(lab[0]), int(lab[1])
                    total += (last[cid] - last[iid]).item()
                else:
                    cid = int(lab)
                    total += torch.log_softmax(last, dim=-1)[cid].item()
                count += 1
        mean_m = total / count
        label_s = "logit_diff (correct − incorrect)" if use_logit_diff else "log prob correct token"
        print(f"Smoke (csv-only, frozen indices, clean forward): mean {label_s} = {mean_m:.4f} ({count} rows)")
        return {
            "frozen_csv": str(csv_path.resolve()),
            "n_rows": n_csv,
            "smoke_mean_metric": mean_m,
        }

    print(f"Building graph and running {method}...")
    g = Graph.from_model(model)
    attr_kw = {"method": method}
    if "IG" in method or "ig" in method.lower():
        attr_kw["ig_steps"] = ig_steps
    attribute(model, g, dataloader, partial(metric, loss=True, mean=True), **attr_kw)

    g.apply_topn(top_n, True)
    n_nodes, n_edges = g.count_included_nodes(), g.count_included_edges()
    print(f"Circuit: {n_nodes} nodes, {n_edges} edges")

    baseline = evaluate_baseline(
        model, dataloader, partial(metric, loss=False, mean=False)
    ).mean().item()
    circuit_perf = evaluate_graph(
        model, g, dataloader, partial(metric, loss=False, mean=False)
    ).mean().item()
    print(f"Baseline (clean) metric: {baseline:.4f}")
    print(f"Circuit metric: {circuit_perf:.4f}")

    graph_json_path = output_path / "eap_graph_frozen.json"
    g.to_json(str(graph_json_path))
    print(f"Saved graph to {graph_json_path}")
    if hasattr(g, "to_pt"):
        pt_path = output_path / "eap_graph_frozen.pt"
        g.to_pt(str(pt_path))
        print(f"Saved graph to {pt_path}")

    circuit_nodes = _extract_circuit_layers(g, graph_json_path)
    c_arith = {
        "method": method,
        "dataset": dataset_type,
        "use_logit_diff": use_logit_diff,
        "label_source": "frozen_csv_ioi_style",
        "frozen_csv": str(csv_path.resolve()),
        "label_tokenization": "continuation_first_token_id_at_csv_build",
        "filter_model_correct": filter_model_correct,
        "n_samples_loaded": n_loaded,
        "n_samples_after_filters_before_csv": len(items),
        "n_rows_in_csv": n_csv,
        "per_format": {dataset_type: circuit_nodes},
        "combined": circuit_nodes,
        "top_n_nodes": top_n,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "baseline_metric": baseline,
        "circuit_metric": circuit_perf,
    }
    c_arith_path = output_path / "C_arith_frozen.json"
    with open(c_arith_path, "w", encoding="utf-8") as f:
        json.dump(c_arith, f, indent=2)
    print(f"Saved C_arith to {c_arith_path}")

    try:
        import pygraphviz
        img_path = output_path / "eap_circuit_graph_frozen.png"
        g.to_image(str(img_path))
        print(f"Saved circuit visualization to {img_path}")
    except ImportError:
        print("Install pygraphviz for circuit visualization: pip install pygraphviz")
    except ValueError as e:
        import traceback
        print(f"Graph image failed ({dataset_type}): {e}")
        traceback.print_exc()

    return c_arith


def main():
    parser = argparse.ArgumentParser(
        description="EAP-IG with IOI-style frozen CSV (correct_idx / incorrect_idx on disk).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="numeric",
        choices=list(COUNTERFACTUAL_DATASETS.keys()),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--method",
        type=str,
        default="EAP-IG-inputs",
        choices=["EAP", "EAP-IG-inputs", "clean-corrupted"],
    )
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20000)
    parser.add_argument("--no-logit-diff", action="store_true")
    parser.add_argument("--output-dir", type=str, default="analysis_output_frozen")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-filter-model-correct",
        action="store_true",
        help="Keep all length-aligned rows when building the frozen CSV.",
    )
    parser.add_argument("--debug-label-tokenization", type=int, default=0, metavar="N")
    parser.add_argument(
        "--frozen-csv",
        type=str,
        default=None,
        help="Explicit path to CSV. If omitted, uses data/eap_frozen/<auto stem>.csv",
    )
    parser.add_argument(
        "--rebuild-frozen-csv",
        action="store_true",
        help="Regenerate CSV from JSON even if a file already exists.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only build/reuse frozen CSV and print mean metric on frozen indices (no Graph / eap).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EAP-IG (frozen labels / IOI-style CSV)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print()

    try:
        run_eap_ig_circuit_analysis_frozen(
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
            filter_model_correct=not args.no_filter_model_correct,
            debug_label_tokenization=args.debug_label_tokenization,
            frozen_csv_path=args.frozen_csv,
            rebuild_frozen_csv=args.rebuild_frozen_csv,
            csv_only=args.csv_only,
        )
        print("\nDone.")
    except ImportError as e:
        print("Missing dependency:", e)
        raise SystemExit(1)
    except Exception as e:
        print("Error:", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
