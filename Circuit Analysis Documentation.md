# EAP-IG Full Pipeline Documentation

Explains the core EAP-IG circuit stack in this repo: how it works, how to run it, and how it aligns with Hanna M. Wang’s Edge Attribution Patching with Integrated Gradients (**EAP-IG**) implementation and methodology.

---

## 1. Basic intuition

The pipeline discovers a sparse subgraph (“circuit”) of a causal LM’s internal computation graph that approximately preserves a chosen task metric on pairs of inputs:

- **Clean prompt** `x` — the model should favor the **correct** next-token target.
- **Corrupted prompt** `x'` — a minimal edit that should favor a **different** (incorrect) target.

Attribution scores edges in a TransformerLens-style graph; the subgraph is formed by keeping the highest-magnitude edges (top- N). Performance is compared before (full model) and after (circuit only) using the same metric.

Implementation entry point: `circuit_analysis_eap_ig.py`.

---

## 2. Files in scope


| File                                                                                             | Role                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `circuit_analysis_eap_ig.py`                                       | Full pipeline: load data → filter → build dataloader → graph → attribute → top- N → evaluate → export.                                                                                                                                          |
| `eval_counterfactual.py`                                               | `COUNTERFACTUAL_DATASETS` paths and `load_counterfactual_dataset` (split + optional `max_samples` truncation).                                                                                                                                  |
| `run_eap_ig_circuit.sh`                                                 | Shell wrapper: default Llama 3.1 8B, `EAP-IG-inputs`, `ig_steps=5`, `top_n=20000`, `batch_size=10`.                                                                                                                                             |
| `run_all_circuits.py`                                                     | Runs **numeric, english, spanish, italian**; loads the model once, then prints degenerate-label stats (same first token for `y` and `y'`) and runs the full pipeline per dataset.                                                               |
| `scripts/eap_metric_smoke_test.py`                           | Fast check: mean logit diff on filtered data without building a full graph.                                                                                                                                                                     |
| `smoke_tests/circuit_analysis_eap_ig_frozen.py` | Optional: same pipeline but with **frozen** `(correct_idx, incorrect_idx)` rows in CSV under `data/eap_frozen/`, matching an IOI-style “frozen indices” workflow. Run from repo root: `python smoke_tests/circuit_analysis_eap_ig_frozen.py …`. |


**Data:**

- `data/json/arith_dataset_numeric_counterfactual.json`
- `data/json/arith_dataset_english_counterfactual.json`
- `data/json/arith_dataset_spanish_counterfactual.json`
- `data/json/arith_dataset_italian_counterfactual.json`

Paths are defined in `eval_counterfactual.COUNTERFACTUAL_DATASETS`.

---

## 3. End-to-end execution flow (code order and design)

1. Load counterfactual JSON for `--dataset` and `--split`, cap with `--max-samples`.
2. Load `HookedTransformer` for `--model` (CLI `--device`: `cuda` / `mps` / `cpu`; the bundled **EAP-IG** attribution path still expects **CUDA** for internal tensors—see §4.1).
3. Filter length-matched `(x, x_prime)` pairs.
4. Filter to model-first-token-correct items.
5. Build `ArithmeticEAPDataset` and `DataLoader` with `batch_size` (default = 10).
6. `Graph.from_model(model)` — full computational graph for attribution.
7. `attribute(..., method=..., ig_steps=...)` — populate edge scores (`EAP`, `EAP-IG-inputs`, or `clean-corrupted`).
8. `apply_topn(top_n, True)` — keep the top `top_n` edges (default = 20000).
9. `evaluate_baseline` vs `evaluate_graph` on the same metric (mean logit diff or log-prob).
10. Save `eap_graph.json`, optional `eap_graph.pt`, `C_arith.json`, optional PNG via PyGraphviz.

---

## 4. How to run

### 4.1 Dependencies

Install TransformerLens and the EAP-IG package from GitHub (the PyPI package named `eap` is **not** the right library):

```bash
pip install transformer_lens
pip install git+https://github.com/hannamw/EAP-IG
```

Optional (circuit diagram image):

```bash
pip install pygraphviz
```

The published **EAP-IG** library allocates internal tensors on **CUDA**; run the full graph pipeline (`attribute` → `evaluate_graph`) on a **GPU node with CUDA PyTorch**. Metric-only checks (e.g. `scripts/eap_metric_smoke_test.py`) can use CPU.

### 4.2 One command (shell)

```bash
./run_eap_ig_circuit.sh [MODEL] [DATASET] [MAX_SAMPLES]
# defaults: meta-llama/Llama-3.1-8B  numeric  100
```

### 4.3 CLI (Python)

```bash
python3 circuit_analysis_eap_ig.py \
  --model meta-llama/Llama-3.1-8B \
  --dataset numeric \
  --split test \
  --max-samples 100 \
  --method EAP-IG-inputs \
  --ig-steps 5 \
  --top-n 20000 \
  --batch-size 10 \
  --output-dir analysis_output
```

Useful flags:

- `--device cpu` — forces CPU (also hides CUDA devices when passed as in the script).
- `--no-logit-diff` — single-token log-prob metric.
- `--no-filter-model-correct` — keep all length-aligned pairs even when the model’s first-token prediction on `x` does not match the correct label token.
- `--debug-label-tokenization N` — print label vs argmax diagnostics.

### 4.4 All four formats

```bash
python3 run_all_circuits.py
```

Edit the script arguments if you need a different model, `max_samples`, or `output_dir`.

---

## 5. Outputs

Default layout when `--output-dir analysis_output`:

```text
analysis_output/
└── <dataset>/           # numeric | english | spanish | italian
    ├── eap_graph.json   # Full graph + inclusion flags (portable)
    ├── eap_graph.pt     # If the installed `eap` exposes `to_pt`
    ├── C_arith.json     # Metadata + list of included layer/node names
    └── eap_circuit_graph.png   # If PyGraphviz succeeds
```

`C_arith.json` records `method`, `dataset`, `use_logit_diff`, `label_tokenization`, filter flags, sample counts, `top_n_nodes`, node/edge counts, and baseline vs circuit metric values.

Git LFS: Large `analysis_output` JSON/PNG assets may be tracked via `.gitattributes`; use Git LFS when cloning or committing those files.

---

## 6. Upstream implementation: Hanna’s EAP-IG repo

### 6.1 Reference implementation

The code is explicitly aligned with the public library and demos in:

- **Repository:** [github.com/hannamw/EAP-IG](https://github.com/hannamw/EAP-IG)
- **Primary demo:** the IOI notebook pattern (`ioi.ipynb` in that repo)

This arithmetic repo calls the same `eap` package APIs: `Graph.from_model`, `attribute`, `evaluate_baseline`, `evaluate_graph`, and graph export (`to_json`, optional `to_pt`, optional `to_image`).

### 6.2 Method: EAP vs EAP-IG-inputs

Edge Attribution Patching (EAP) attributes edges using gradients of the task metric with respect to edge interventions along clean vs corrupted runs.

EAP-IG-inputs (this repo’s default: `--method EAP-IG-inputs`) uses integrated gradients along a path between clean and corrupted inputs, with multiple steps (`--ig-steps`, default = 5). That can yield a more stable signal than a single-point gradient when the landscape is flat or saturated.

### 6.3 Task metric: `logit_diff` (IOI-style)

The IOI demo uses a two-token supervision signal: maximize the logit gap between a correct and incorrect token at the prediction position. In `circuit_analysis_eap_ig.py` this is implemented as `logit_diff` at the last prompt position:

\text{metric} = \text{logit}[\text{correctidx}] - \text{logit}[\text{incorrectidx}]

using per-batch `torch.gather` (same structure as the notebook: `good_bad[:, 0] - good_bad[:, 1]`). For attribution, the code minimizes negative mean logit diff (`loss=True`), so edges that increase the gap receive higher attribution under gradient-based methods.

Optional mode: `--no-logit-diff` switches to log-probability of the correct token only (single-label metric).

---

## 7. One-to-one mapping from Hanna’s repo to this repo


| IOI / `hannamw/EAP-IG` concept                  | This repository                                                                                                                                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clean vs corrupted text pairs                   | `x` vs `x_prime` from counterfactual JSON                                                                                                                                                                           |
| Correct vs incorrect label token ids            | First token ids of `y` vs `y_prime` (or `y_expected` / `y_prime_expected`) via `**continuation_first_token_id`** — **no** `.strip()` before encoding, so leading spaces in JSON match real continuations            |
| `Dataset` yielding `(clean, corrupted, labels)` | `ArithmeticEAPDataset` + `collate_eap` → labels shape `(batch, 2)` for logit diff                                                                                                                                   |
| Same token length for clean/corrupted           | Required by EAP-IG; pairs with mismatched token counts are filtered                                                                                                                                         |
| `HookedTransformer` + graph hooks               | `HookedTransformer.from_pretrained` + `Graph.from_model(model)`                                                                                                                                                     |
| Llama-friendly TL config                        | `center_writing_weights=False`, `center_unembed=False`, `fold_ln=False`, half precision on GPU; `use_split_qkv_input`, `use_attn_result`, `use_hook_mlp_in`, and `ungroup_grouped_query_attention` where applicable |
| Attribution                                     | `attribute(model, g, dataloader, partial(metric, loss=True, mean=True), method=..., ig_steps=...)`                                                                                                                  |
| Subgraph selection                              | `g.apply_topn(top_n, True)` — default `top_n=20000` (same order of magnitude as the IOI notebook)                                                                                                                   |
| Evaluation                                      | `evaluate_baseline` vs `evaluate_graph` with `loss=False`                                                                                                                                                           |
| Artifacts                                       | `eap_graph.json`, optional `eap_graph.pt`, `C_arith.json`, optional `eap_circuit_graph.png`                                                                                                                         |


---

## 8. Data format and filtering

This section matches the behaviors exercised in testing (length alignment, labels, optional filters).

### 8.1 Counterfactual items

Each example should provide at least:

- `x` — clean prompt string  
- `x_prime` — corrupted prompt (same “slot” structure as `x` after the counterfactual edit)  
- `y` / `y_prime` (or `y_expected` / `y_prime_expected`) — answer continuations as the model should see them after the prompt

The `split` field selects train / val / test. 

### 8.2 Length alignment

EAP-IG in this setup assumes clean and corrupted prompts tokenize to the same length. If not, the pair is dropped. The script reports how many were removed.

### 8.3 Label tokenization: `continuation_first_token_id`

Answers in JSON often start with a leading space (e.g. `" 22"`). The first token id of the string as stored is the correct supervision target for the first generated token after the prompt. Stripping before `encode` can swap in the wrong token id and break `logit_diff`.

### 8.4 Optional filter: `filter_model_correct` (default: on)

The pipeline can restrict to items where the model’s argmax at the last prompt position equals the correct label’s first token id. That targets the subset where the model is “on task” on the clean prompt. If too few items survive, use `--no-filter-model-correct` or increase `--max-samples`. Use `--debug-label-tokenization N` for diagnostics on the first `N` length-aligned examples.

### 8.5 Degenerate labels

If `y` and `y_prime` share the same first token id, `**logit_diff` is identically zero for that row. `run_all_circuits.py` warns when this happens frequently.

---


Optional **frozen-index** variant: `[smoke_tests/circuit_analysis_eap_ig_frozen.py](smoke_tests/circuit_analysis_eap_ig_frozen.py)` and `data/eap_frozen/`.
