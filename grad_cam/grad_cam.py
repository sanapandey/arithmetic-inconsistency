#!/usr/bin/env python3
"""
Phase II: Grad-CAM for Diagnosing Arithmetic Failures

Layer-wise Transformer Grad-CAM:
- F(x) = logit(y_correct) - logit(y_counterfactual)
- α_l = (1/Td) * sum(dF/dA_l)
- GradCAM_l = ReLU(α_l * A_l)

Computes normative saliency μ_l from correct runs, then D_sal for incorrect runs.
Uses C_arith (established circuit from activation patching) for comparison.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_circuit(circuit_path: str) -> Dict:
    """Load C_arith from analysis output."""
    with open(circuit_path, "r") as f:
        return json.load(f)


def load_counterfactual_results(results_path: str) -> Dict:
    """Load counterfactual evaluation results."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_answer_token_id(tokenizer, answer: str) -> int:
    """Get first token ID for an answer (handles leading space for numbers)."""
    s = str(answer).strip()
    if not s:
        return tokenizer.eos_token_id
    # Try with and without leading space (common for numeric answers)
    for candidate in [f" {s}", s, f" {s}"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return ids[0]
    return tokenizer.eos_token_id


def extract_answer(text: str, dataset_type: str) -> str:
    """Extract answer from model output."""
    text = text.strip()
    if dataset_type == "numeric":
        m = re.search(r"-?\d+", text)
        return m.group() if m else text
    text = text.split("\n")[0]
    text = re.split(r"[.!?]", text)[0]
    return text.strip()


def get_layer_names(model) -> List[str]:
    """Get transformer layer names for the model (Llama, GPT-2, etc.)."""
    seen = set()
    names = []
    for name, _ in model.named_modules():
        # Llama: model.model.layers.0, model.model.layers.1, ...
        m = re.search(r"(model\.model\.layers\.\d+|transformer\.h\.\d+|model\.layers\.\d+)", name)
        if m:
            prefix = m.group(1)
            if prefix not in seen:
                seen.add(prefix)
                names.append(prefix)
    return sorted(names, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0)


def compute_layer_wise_gradcam(
    model: nn.Module,
    tokenizer,
    prompt: str,
    correct_token_id: int,
    counterfactual_token_id: int,
    layer_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute layer-wise Grad-CAM scores.
    
    F(x) = logit(y_correct) - logit(y_counterfactual)
    α_l = (1/Td) * sum(dF/dA_l)
    GradCAM_l = ReLU(α_l * A_l) -> scalar = mean(ReLU(α_l * A_l))
    
    Returns:
        Dict mapping layer_name -> GradCAM scalar score
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Hook to store layer outputs (need grad for backprop)
    stored = {}
    
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                act = out[0]
            else:
                act = out
            act.retain_grad()
            stored[name] = act
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append((name, h))
    
    try:
        # Forward
        outputs = model(**inputs, output_attentions=False)
        logits = outputs.logits  # (batch, seq, vocab)
        
        # F = logit(y_correct) - logit(y_counterfactual) at last position
        last_pos = logits.shape[1] - 1
        logit_correct = logits[0, last_pos, correct_token_id]
        logit_counterfactual = logits[0, last_pos, counterfactual_token_id]
        F = logit_correct - logit_counterfactual
        
        # Backward
        model.zero_grad()
        F.backward()
        
        # Compute α_l and GradCAM_l per layer
        gradcam_scores = {}
        for name in layer_names:
            if name not in stored:
                continue
            A = stored[name]
            if A.grad is None:
                gradcam_scores[name] = 0.0
                continue
            
            grad = A.grad  # dF/dA_l
            T, d = A.shape[1], A.shape[2]
            alpha_l = grad.sum() / (T * d)
            # GradCAM_l = ReLU(alpha_l * A_l), scalar = mean
            cam = torch.relu(alpha_l * A)
            gradcam_scores[name] = cam.mean().item()
        
        return gradcam_scores
    
    finally:
        for _, h in hooks:
            h.remove()


def run_gradcam_analysis(
    model,
    tokenizer,
    results: List[Dict],
    circuit: Dict,
    dataset_type: str,
    layer_names: List[str],
    max_correct: Optional[int] = None,
    max_incorrect: Optional[int] = None,
    device: torch.device = None,
) -> Dict:
    """
    Run Grad-CAM on correct (for μ) and incorrect (for D_sal) samples.
    
    Returns:
        Dict with normative_mean, incorrect_results, D_sal_per_sample, etc.
    """
    if device is None:
        device = next(model.parameters()).device
    
    correct_samples = [r for r in results if r.get("correct_x", False)]
    incorrect_samples = [r for r in results if not r.get("correct_x", True)]
    
    if max_correct:
        correct_samples = correct_samples[:max_correct]
    if max_incorrect:
        incorrect_samples = incorrect_samples[:max_incorrect]
    
    c_arith_layers = set(circuit.get("combined", []))
    if not c_arith_layers:
        c_arith_layers = set(circuit.get("per_format", {}).get(dataset_type, []))
    if not c_arith_layers:
        # Fallback: use top 10 layers when C_arith not available
        c_arith_layers = set(layer_names[: min(10, len(layer_names))])
    
    # Restrict to layers we can compute
    available_layers = [l for l in layer_names if l in c_arith_layers]
    if not available_layers:
        available_layers = layer_names[: min(10, len(layer_names))]
    
    # Compute μ_l from correct runs
    normative_scores = {l: [] for l in available_layers}
    
    for item in tqdm(correct_samples, desc="Grad-CAM on correct (μ)"):
        x = item["x"]
        y_expected = item["y_expected"].strip()
        y_prime_expected = item["y_prime_expected"].strip()
        
        correct_id = get_answer_token_id(tokenizer, y_expected)
        counterfactual_id = get_answer_token_id(tokenizer, y_prime_expected)
        
        try:
            scores = compute_layer_wise_gradcam(
                model, tokenizer,
                x, correct_id, counterfactual_id,
                available_layers, device,
            )
            for l, s in scores.items():
                normative_scores[l].append(s)
        except Exception as e:
            continue
    
    mu_l = {l: float(np.mean(v)) if v else 0.0 for l, v in normative_scores.items()}
    
    # Compute Grad-CAM and D_sal for incorrect runs
    incorrect_results = []
    
    for item in tqdm(incorrect_samples, desc="Grad-CAM on incorrect (D_sal)"):
        x = item["x"]
        y_expected = item["y_expected"].strip()
        y_prime_expected = item["y_prime_expected"].strip()
        
        correct_id = get_answer_token_id(tokenizer, y_expected)
        counterfactual_id = get_answer_token_id(tokenizer, y_prime_expected)
        
        try:
            scores = compute_layer_wise_gradcam(
                model, tokenizer,
                x, correct_id, counterfactual_id,
                available_layers, device,
            )
        except Exception as e:
            incorrect_results.append({
                "id": item["id"],
                "x": x,
                "y_expected": y_expected,
                "predicted_x": item.get("predicted_x", ""),
                "gradcam_l": {},
                "D_sal": float("nan"),
                "error": str(e),
            })
            continue
        
        # D_sal = sum over l in C_arith of |GradCAM_{l} - μ_l|
        d_sal = 0.0
        for l in available_layers:
            if l in mu_l:
                d_sal += abs(scores.get(l, 0.0) - mu_l[l])
        
        incorrect_results.append({
            "id": item["id"],
            "x": x,
            "y_expected": y_expected,
            "predicted_x": item.get("predicted_x", ""),
            "gradcam_l": {k: float(v) for k, v in scores.items()},
            "D_sal": float(d_sal),
        })
    
    return {
        "dataset_type": dataset_type,
        "normative_mean_mu_l": mu_l,
        "C_arith_layers": list(available_layers),
        "n_correct_used": len(correct_samples),
        "n_incorrect_used": len(incorrect_results),
        "incorrect_results": incorrect_results,
        "D_sal_mean": float(np.nanmean([r["D_sal"] for r in incorrect_results if "error" not in r])),
        "D_sal_std": float(np.nanstd([r["D_sal"] for r in incorrect_results if "error" not in r])),
    }


def create_visualizations(analysis: Dict, output_dir: Path):
    """Create Grad-CAM heatmaps and D_sal plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not found, skipping visualizations")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. Layer-wise normative μ_l vs incorrect Grad-CAM
    mu = analysis.get("normative_mean_mu_l", {})
    if mu:
        layers = sorted(mu.keys(), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0)
        values = [mu[l] for l in layers]
        labels = [l.split(".")[-1] if "." in l else l for l in layers]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(layers)), values, color="steelblue", alpha=0.7)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Normative μ_l (Grad-CAM)")
        ax.set_title("Established Arithmetic Circuit: Mean Layer Saliency (correct runs)")
        plt.tight_layout()
        plt.savefig(output_dir / "gradcam_normative_mu.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'gradcam_normative_mu.png'}")
    
    # 2. D_sal distribution for incorrect predictions
    incorrect = analysis.get("incorrect_results", [])
    d_sals = [r["D_sal"] for r in incorrect if "error" not in r and not np.isnan(r["D_sal"])]
    if d_sals:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d_sals, bins=min(20, len(d_sals)), color="coral", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(d_sals), color="red", linestyle="--", label=f"Mean: {np.mean(d_sals):.3f}")
        ax.set_xlabel("D_sal (Saliency Divergence)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of D_sal on Incorrect Predictions")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "gradcam_D_sal_distribution.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'gradcam_D_sal_distribution.png'}")
    
    # 3. Top incorrect samples by D_sal (bar chart of Grad-CAM vs μ)
    top_incorrect = sorted(
        [r for r in incorrect if "error" not in r],
        key=lambda x: x["D_sal"],
        reverse=True,
    )[:5]
    
    if top_incorrect and mu:
        fig, axes = plt.subplots(len(top_incorrect), 1, figsize=(10, 3 * len(top_incorrect)))
        if len(top_incorrect) == 1:
            axes = [axes]
        layers = list(mu.keys())
        labels = [l.split(".")[-1] if "." in l else l for l in layers]
        
        for idx, item in enumerate(top_incorrect):
            ax = axes[idx]
            g_l = item.get("gradcam_l", {})
            vals = [g_l.get(l, 0) - mu.get(l, 0) for l in layers]
            colors = ["coral" if v > 0 else "steelblue" for v in vals]
            ax.bar(range(len(layers)), vals, color=colors, alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Grad-CAM - μ")
            ax.set_title(f"#{item['id']} D_sal={item['D_sal']:.3f} | x={item['x'][:40]}...")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "gradcam_top_deviations.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'gradcam_top_deviations.png'}")


def generate_report(analysis: Dict, output_dir: Path):
    """Generate Grad-CAM analysis report."""
    report_path = output_dir / "GRADCAM_REPORT.md"
    
    with open(report_path, "w") as f:
        f.write("# Phase II: Grad-CAM Analysis Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Dataset**: {analysis.get('dataset_type', 'N/A')}\n")
        f.write(f"- **Correct samples used for μ**: {analysis.get('n_correct_used', 0)}\n")
        f.write(f"- **Incorrect samples analyzed**: {analysis.get('n_incorrect_used', 0)}\n")
        f.write(f"- **D_sal mean (incorrect)**: {analysis.get('D_sal_mean', 0):.4f}\n")
        f.write(f"- **D_sal std (incorrect)**: {analysis.get('D_sal_std', 0):.4f}\n\n")
        
        f.write("## C_arith Layers (Established Circuit)\n\n")
        for l in analysis.get("C_arith_layers", [])[:15]:
            mu = analysis.get("normative_mean_mu_l", {}).get(l, 0)
            f.write(f"- {l}: μ_l = {mu:.4f}\n")
        f.write("\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **D_sal** measures how much incorrect predictions deviate from the normative circuit.\n")
        f.write("- High D_sal: failure likely due to under-activation, misallocation, or token misalignment.\n")
        f.write("- See `gradcam_D_sal_distribution.png` and `gradcam_top_deviations.png` for details.\n\n")
        
        f.write("## Top 10 Incorrect Samples by D_sal\n\n")
        incorrect = analysis.get("incorrect_results", [])
        top = sorted(
            [r for r in incorrect if "error" not in r],
            key=lambda x: x["D_sal"],
            reverse=True,
        )[:10]
        for r in top:
            f.write(f"- **{r['id']}** D_sal={r['D_sal']:.4f} | x={r['x']} | expected={r['y_expected']} | predicted={r.get('predicted_x','')[:30]}\n")
    
    print(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase II: Grad-CAM for diagnosing arithmetic failures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--results-file", type=str, required=True,
                        help="Path to counterfactual_results.json")
    parser.add_argument("--circuit-file", type=str, required=True,
                        help="Path to C_arith.json from analysis")
    parser.add_argument("--output-dir", type=str, default="gradcam_output")
    parser.add_argument("--dataset-type", type=str, default="numeric",
                        choices=["numeric", "english", "spanish", "italian"])
    parser.add_argument("--max-correct", type=int, default=50,
                        help="Max correct samples for μ")
    parser.add_argument("--max-incorrect", type=int, default=100,
                        help="Max incorrect samples for D_sal")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and tokenizer...")
    try:
        import accelerate
        device_map = "auto"
    except ImportError:
        device_map = None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    
    device = next(model.parameters()).device
    layer_names = get_layer_names(model)
    
    print("Loading circuit and results...")
    circuit = load_circuit(args.circuit_file)
    data = load_counterfactual_results(args.results_file)
    results = data.get("results", data) if isinstance(data, dict) else data
    
    print(f"Running Grad-CAM analysis ({args.dataset_type})...")
    analysis = run_gradcam_analysis(
        model, tokenizer, results, circuit, args.dataset_type,
        layer_names,
        max_correct=args.max_correct,
        max_incorrect=args.max_incorrect,
        device=device,
    )
    
    # Save JSON
    out_json = {k: v for k, v in analysis.items() if k != "incorrect_results"}
    out_json["incorrect_results_count"] = len(analysis.get("incorrect_results", []))
    with open(output_dir / "gradcam_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved: {output_dir / 'gradcam_analysis.json'}")
    
    # Visualizations and report
    create_visualizations(analysis, output_dir)
    generate_report(analysis, output_dir)
    
    print("\nGrad-CAM analysis complete.")


if __name__ == "__main__":
    main()
