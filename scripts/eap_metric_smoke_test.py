#!/usr/bin/env python3
"""
Quick check: mean logit_diff on clean prompts using continuation vs stripped label token ids.
No EAP graph — fast once the model is loaded.

Example:
  python scripts/eap_metric_smoke_test.py --model gpt2-small --dataset numeric --max-samples 200
  python scripts/eap_metric_smoke_test.py --no-filter-model-correct  # include rows where argmax ≠ y’s first token
"""

import argparse
import sys
from pathlib import Path

import torch

# Repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circuit_analysis_eap_ig import (
    continuation_first_token_id,
    filter_items_model_first_token_correct,
    filter_length_aligned_items,
    stripped_first_token_id,
)
from eval_counterfactual import COUNTERFACTUAL_DATASETS, load_counterfactual_dataset


def main():
    p = argparse.ArgumentParser(
        description="Smoke test: mean logit_diff with new vs old label ids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="gpt2-small")
    p.add_argument("--dataset", default="numeric", choices=list(COUNTERFACTUAL_DATASETS.keys()))
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-filter-model-correct",
        action="store_true",
        help="Do not restrict to rows where argmax at last position equals the first token of y "
        "(default: filter is ON, matching circuit_analysis_eap_ig.py).",
    )
    args = p.parse_args()
    filter_model_correct = not args.no_filter_model_correct

    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        print("pip install transformer_lens")
        sys.exit(1)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    load_kw = dict(device=device)
    if "llama" in args.model.lower() or "meta-llama" in args.model.lower():
        load_kw.update(
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )

    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(args.model, **load_kw)
    model.eval()
    tok = model.tokenizer

    path = COUNTERFACTUAL_DATASETS[args.dataset]
    items = load_counterfactual_dataset(path, args.split, args.max_samples)
    items = filter_length_aligned_items(items, tok)
    if not items:
        print("No length-aligned samples.")
        sys.exit(1)

    n_after_len = len(items)
    if filter_model_correct:
        items = filter_items_model_first_token_correct(model, items, tok)
    if not items:
        print("No samples left after filters. Increase --max-samples or use --no-filter-model-correct.")
        sys.exit(1)

    diffs_new = []
    diffs_old = []
    n_skip_new = 0
    n_skip_old = 0
    vocab = model.cfg.d_vocab

    with torch.inference_mode():
        for item in items:
            y = item.get("y", item.get("y_expected", ""))
            yp = item.get("y_prime", item.get("y_prime_expected", ""))
            cn = continuation_first_token_id(tok, y)
            inn = continuation_first_token_id(tok, yp)
            co = stripped_first_token_id(tok, y)
            io = stripped_first_token_id(tok, yp)

            text = str(item["x"]).strip()
            logits = model(text)[0, -1]

            if 0 <= cn < vocab and 0 <= inn < vocab and cn != inn:
                diffs_new.append((logits[cn] - logits[inn]).item())
            else:
                n_skip_new += 1

            if 0 <= co < vocab and 0 <= io < vocab and co != io:
                diffs_old.append((logits[co] - logits[io]).item())
            else:
                n_skip_old += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print()
    print(f"Loaded up to {args.max_samples} rows; length-aligned: {n_after_len}; "
          f"after model-correct filter: {len(items)} (filter={'on' if filter_model_correct else 'off'})")
    print()
    print(f"Mean logit_diff — continuation labels (current):     {mean(diffs_new):+.4f}  (n={len(diffs_new)}, skipped={n_skip_new})")
    print(f"Mean logit_diff — stripped-y labels (old bug):       {mean(diffs_old):+.4f}  (n={len(diffs_old)}, skipped={n_skip_old})")
    print()
    if mean(diffs_new) > mean(diffs_old):
        print("Continuation labels give a higher mean than stripped labels (usually expected after the fix).")
    elif filter_model_correct and mean(diffs_new) > 0:
        print("Mean is positive with continuation labels: correct id is favored over y' on average.")
    print()


if __name__ == "__main__":
    main()
