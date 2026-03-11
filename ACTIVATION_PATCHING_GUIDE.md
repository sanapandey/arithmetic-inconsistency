# Activation Patching Guide for Counterfactual Arithmetic Analysis

This guide walks you through evaluating an 8B model on counterfactual datasets and using activation patching with EAP-IG visualization to identify arithmetic reasoning circuits.

## Overview

**Goal**: Identify which layers/components in the model are causally responsible for arithmetic reasoning by:
1. Evaluating on counterfactual pairs (x, x')
2. Patching activations from corrupted (x') into original (x)
3. Measuring how patching affects predictions
4. Visualizing results with EAP-IG

## Step-by-Step Process

### Step 1: Evaluate Model on Counterfactual Datasets

First, evaluate your 8B model on the counterfactual datasets and optionally collect activations:

```bash
# Basic evaluation (no activation collection - faster)
python eval_counterfactual.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset numeric \
    --split test \
    --max-samples 200

# With activation collection (slower, but needed for detailed analysis)
python eval_counterfactual.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset numeric \
    --split test \
    --collect-activations \
    --max-samples 100  # Start small - activation collection is memory-intensive
```

**What this does:**
- Loads counterfactual dataset (x, x', y, y' pairs)
- Evaluates model on both x and x'
- Optionally collects activations at each layer
- Saves results to `results/<model>/counterfactual/`

**Output files:**
- `{dataset}_counterfactual_results.json`: Full results with predictions
- `{dataset}_counterfactual_summary.json`: Accuracy summary

### Step 2: Run Activation Patching Experiments

Once you have evaluation results, run activation patching to see which layers causally affect predictions:

```bash
python activation_patching.py \
    --model meta-llama/Llama-2-7b-hf \
    --results-file results/meta-llama__Llama-2-7b-hf/counterfactual/numeric_counterfactual_results.json \
    --max-samples 200 \
    --format-eap-ig
```

**What this does:**
- Loads counterfactual evaluation results
- For each pair (x, x'):
  - Patches activations from x' into x at each layer
  - Measures how patching changes the prediction
- Formats results for EAP-IG visualization

**Output files:**
- `patching_results.json`: Detailed patching results
- `eap_ig_data.json`: Formatted data for EAP-IG visualization

### Step 3: Visualize with EAP-IG

Load the `eap_ig_data.json` file into your EAP-IG visualization tool:

```python
import json
from eap_ig import visualize  # Adjust import based on your EAP-IG installation

# Load data
with open('results/patching_results/eap_ig_data.json', 'r') as f:
    eap_data = json.load(f)

# Visualize (adjust based on EAP-IG API)
visualize(eap_data)
```

**What to look for:**
- **High-effect layers**: Layers where patching x' → x significantly changes predictions
- **Early vs late layers**: Where in the network arithmetic reasoning happens
- **Attention patterns**: Which positions matter most

## Understanding the Results

### Counterfactual Evaluation Results

Each result contains:
- `x`: Original problem (e.g., "38 + 27 =")
- `x_prime`: Corrupted problem (e.g., "38 - 27 =")
- `y_expected`: Correct answer to x (e.g., " 65")
- `y_prime_expected`: Correct answer to x' (e.g., " 11")
- `predicted_x`: Model's prediction for x
- `predicted_x_prime`: Model's prediction for x'
- `correct_x`: Whether prediction for x is correct
- `correct_x_prime`: Whether prediction for x' is correct

### Activation Patching Results

Each patching result shows:
- `layer_effects`: For each layer, how patching affects the output
  - `patched_output`: What the model predicts after patching
  - `prediction_changed`: Whether patching changed the prediction
  - `layer_index`: Which layer this is

**Interpretation:**
- If patching at layer L changes predictions → Layer L is causally important
- If patching at layer L doesn't change predictions → Layer L is not critical for this computation

## Example Workflow

### Quick Test (5-10 minutes)

```bash
# 1. Quick evaluation
python eval_counterfactual.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset numeric \
    --split test \
    --max-samples 50

# 2. Run patching on a few samples
python activation_patching.py \
    --model meta-llama/Llama-2-7b-hf \
    --results-file results/meta-llama__Llama-2-7b-hf/counterfactual/numeric_counterfactual_results.json \
    --max-samples 50 \
    --format-eap-ig
```

### Full Analysis (1-2 hours)

```bash
# 1. Full evaluation on test set
python eval_counterfactual.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset all \
    --split test \
    --max-samples 200

# 2. Run patching on all results
python activation_patching.py \
    --model meta-llama/Llama-2-7b-hf \
    --results-file results/meta-llama__Llama-2-7b-hf/counterfactual/numeric_counterfactual_results.json \
    --max-samples 200 \
    --format-eap-ig
```

## Troubleshooting

### Memory Issues

If you run out of memory when collecting activations:
- Reduce `--max-samples`
- Use smaller models for testing
- Collect activations in batches

### Layer Detection Issues

If activation patching can't find layers:
- Manually specify with `--layers model.layers.0 model.layers.1 ...`
- Check model architecture: `print(model)`

### EAP-IG Integration

If EAP-IG format doesn't match your installation:
- Check EAP-IG documentation for expected format
- Modify `format_for_eap_ig()` function in `activation_patching.py`
- Export activations directly and use EAP-IG's import functions

## Advanced: Custom Patching Strategies

You can modify `activation_patching.py` to:
- Patch specific positions (not just entire layers)
- Patch attention vs MLP components separately
- Patch in reverse direction (x → x')
- Measure effect size (not just binary change)

## Next Steps

1. **Identify key layers**: Which layers show strongest patching effects?
2. **Analyze attention**: Use attention visualization to see what the model focuses on
3. **Compare formats**: Does the model use different circuits for numeric vs verbal?
4. **Intervention experiments**: Can you fix errors by patching at specific layers?

## Files Created

- `eval_counterfactual.py`: Evaluation script for counterfactual datasets
- `activation_patching.py`: Activation patching experiments
- `ACTIVATION_PATCHING_GUIDE.md`: This guide

## Questions?

- Check the code comments in each script
- Review EAP-IG documentation for visualization options
- Start with small samples to understand the workflow

