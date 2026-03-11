# Phase II: Grad-CAM Guide

## Overview

Grad-CAM diagnoses arithmetic failures by comparing activation saliency on incorrect runs to the normative circuit (μ) established from correct runs. It uses C_arith from the activation patching analysis.

## Prerequisites

1. **Completed Phase I analysis**: Run `./run_full_analysis.sh` first.
2. **C_arith.json**: Saved to `analysis_output/C_arith.json` by comprehensive analysis.
3. **Counterfactual results**: `results/<model>/counterfactual/<dataset>_counterfactual_results.json`.

## Quick Start

```bash
# After run_full_analysis.sh has completed:
./run_gradcam.sh meta-llama/Llama-2-7b-hf numeric
```

## Manual Run

```bash
python3 grad_cam.py \
    --model meta-llama/Llama-2-7b-hf \
    --results-file results/meta-llama__Llama-2-7b-hf/counterfactual/numeric_counterfactual_results.json \
    --circuit-file analysis_output/C_arith.json \
    --output-dir gradcam_output \
    --dataset-type numeric \
    --max-correct 50 \
    --max-incorrect 100
```

## Arguments

- `--model`: HuggingFace model name
- `--results-file`: Path to counterfactual_results.json
- `--circuit-file`: Path to C_arith.json (from comprehensive analysis)
- `--output-dir`: Output directory (default: gradcam_output)
- `--dataset-type`: numeric, english, spanish, or italian
- `--max-correct`: Max correct samples for normative μ (default: 50)
- `--max-incorrect`: Max incorrect samples for D_sal (default: 100)

## Output

- **gradcam_analysis.json**: Full results (μ_l, D_sal per sample, etc.)
- **gradcam_normative_mu.png**: Layer-wise normative saliency (C_arith)
- **gradcam_D_sal_distribution.png**: Distribution of D_sal on incorrect predictions
- **gradcam_top_deviations.png**: Top 5 incorrect samples by D_sal (Grad-CAM vs μ)
- **GRADCAM_REPORT.md**: Written report

## C_arith (Configurable)

C_arith is loaded from `--circuit-file`. To use a custom circuit:

1. Create a JSON file with:
   ```json
   {
     "combined": ["model.model.layers.5", "model.model.layers.10", ...],
     "per_format": {
       "numeric": [...],
       "english": [...],
       ...
     },
     "layer_importance_per_format": {...}
   }
   ```
2. Pass with `--circuit-file path/to/custom_circuit.json`

## Interpretation

- **μ_l**: Mean layer saliency on correct runs (normative circuit engagement).
- **D_sal**: Sum over layers in C_arith of |Grad-CAM_l - μ_l|. High D_sal on incorrect runs indicates:
  - Under-activation: arithmetic heads under-engaged
  - Misallocation: saliency shifted to non-arithmetic heads
  - Token misalignment: operand/operator tokens weakly attended

## Relationship to EAP-IG

- **EAP-IG / Activation Patching**: Identifies structural causal edges (which layers matter).
- **Grad-CAM**: Measures runtime engagement of those layers on each inference.

Together they support circuit-level failure analysis rather than only output accuracy.
