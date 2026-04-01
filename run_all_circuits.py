#!/usr/bin/env python3
"""
Run EAP-IG circuit analysis on all arithmetic datasets and
sanity-check that the logit-diff labels are non-degenerate
(i.e. first tokens for y and y' are not identical).
"""

from functools import partial
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from circuit_analysis_eap_ig import (
    continuation_first_token_id,
    run_eap_ig_circuit_analysis,
)
from eval_counterfactual import (
    COUNTERFACTUAL_DATASETS,
    load_counterfactual_dataset,
)


DATASET_TYPES = ["numeric", "english", "spanish", "italian"]


def check_label_token_diff_for_dataset(
    model_name: str,
    dataset_type: str,
    split: str = "test",
    max_samples: int = 100,
    device: str | None = None,
) -> None:
    """
    For a given dataset, report how often the first token IDs for y and y'
    (or y_expected / y_prime_expected) are identical. In those cases,
    logit_diff(correct - incorrect) is structurally zero.
    """

    dataset_path = COUNTERFACTUAL_DATASETS.get(dataset_type)
    if not dataset_path or not Path(dataset_path).exists():
        print(f"[{dataset_type}] Dataset not found at {dataset_path}")
        return

    # Device
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # Load model *only* to get the tokenizer
    print(f"[{dataset_type}] Loading {model_name} to check labels...")
    load_kw = dict(device=device)
    if "llama" in model_name.lower() or "Llama" in model_name or "meta-llama" in model_name:
        load_kw.update(
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    model = HookedTransformer.from_pretrained(model_name, **load_kw)
    tokenizer = model.tokenizer

    items = load_counterfactual_dataset(dataset_path, split, max_samples)
    if not items:
        print(f"[{dataset_type}] No samples loaded for split='{split}'")
        return

    n_total = len(items)
    n_same_first_token = 0
    examples_with_same: list[tuple[str, str, str]] = []

    for it in items:
        y = it.get("y", it.get("y_expected", ""))
        y_prime = it.get("y_prime", it.get("y_prime_expected", ""))
        id_y = continuation_first_token_id(tokenizer, y)
        id_y_prime = continuation_first_token_id(tokenizer, y_prime)
        if id_y == id_y_prime:
            n_same_first_token += 1
            if len(examples_with_same) < 5:
                examples_with_same.append((it.get("id", "NA"), y, y_prime))

    pct_same = n_same_first_token / n_total
    if n_same_first_token == 0:
        print(f"[{dataset_type}] All {n_total} examples have distinct first label tokens (y vs y') — logit_diff is well-defined.")
    else:
        print(
            f"[{dataset_type}] WARNING: {n_same_first_token}/{n_total} examples ({pct_same:.1%}) "
            f"have identical first label token IDs (y vs y'); logit_diff will be zero for those."
        )
    if examples_with_same:
        print(f"[{dataset_type}] Example cases where first tokens match:")
        for ex_id, y, y_prime in examples_with_same:
            print(f"  id={ex_id!r}  y={y!r}  y_prime={y_prime!r}")
    print()


def run_all_circuit_analyses(
    model_name: str = "meta-llama/Llama-3.1-8B",
    split: str = "test",
    max_samples: int = 100,
    batch_size: int = 10,
    method: str = "EAP-IG-inputs",
    ig_steps: int = 5,
    top_n: int = 20000,
    output_dir: str = "analysis_output",
    use_logit_diff: bool = True,
    device: str | None = None,
    filter_model_correct: bool = True,
    debug_label_tokenization: int = 0,
):
    """
    Run EAP-IG circuit analysis on all configured arithmetic datasets,
    and for each dataset also report how often y and y' share the same
    first token ID (which would force logit_diff to zero).
    """

    for dataset_type in DATASET_TYPES:
        print("=" * 80)
        print(f"Dataset: {dataset_type}")
        print("=" * 80)

        # 1) Sanity-check label tokens for potential zero logit-diff issues
        check_label_token_diff_for_dataset(
            model_name=model_name,
            dataset_type=dataset_type,
            split=split,
            max_samples=max_samples,
            device=device,
        )

        # 2) Run the full circuit analysis using your existing function
        print(f"[{dataset_type}] Running circuit analysis...")
        run_eap_ig_circuit_analysis(
            model_name=model_name,
            dataset_type=dataset_type,
            split=split,
            max_samples=max_samples,
            batch_size=batch_size,
            method=method,
            ig_steps=ig_steps,
            top_n=top_n,
            output_dir=output_dir,
            device=device,
            use_logit_diff=use_logit_diff,
            filter_model_correct=filter_model_correct,
            debug_label_tokenization=debug_label_tokenization,
        )
        print(f"[{dataset_type}] Circuit analysis complete.\n")


if __name__ == "__main__":
    run_all_circuit_analyses()