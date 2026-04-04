#!/bin/bash
# Quick-start script for counterfactual analysis with activation patching

# Configuration
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
DATASET="${2:-numeric}"
SPLIT="${3:-test}"
MAX_SAMPLES="${4:-200}"

echo "=========================================="
echo "Counterfactual Analysis Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Max samples: $MAX_SAMPLES"
echo ""

# Step 1: Evaluate on counterfactual dataset
echo "Step 1: Evaluating on counterfactual dataset..."
python eval_counterfactual.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --max-samples "$MAX_SAMPLES" \
    --log-every 10

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

# Find results file
MODEL_SAFE=$(echo "$MODEL" | sed 's/\//__/g')
RESULTS_FILE="results/$MODEL_SAFE/counterfactual/${DATASET}_counterfactual_results.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file not found at $RESULTS_FILE"
    exit 1
fi

echo ""
echo "Step 2: Running activation patching experiments..."
python activation_patching.py \
    --model "$MODEL" \
    --results-file "$RESULTS_FILE" \
    --max-samples "$MAX_SAMPLES" \
    --format-eap-ig

if [ $? -ne 0 ]; then
    echo "Error: Activation patching failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: results/$MODEL_SAFE/counterfactual/"
echo "EAP-IG data: results/patching_results/eap_ig_data.json"
echo ""
echo "Next: Load eap_ig_data.json into EAP-IG visualization tool"

