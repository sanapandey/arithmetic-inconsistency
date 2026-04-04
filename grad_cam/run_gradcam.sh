#!/bin/bash
# Phase II: Grad-CAM analysis for diagnosing arithmetic failures
# Requires: completed analysis (C_arith.json), counterfactual results

# Configuration
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
DATASET_TYPE="${2:-numeric}"
RESULTS_DIR="${3:-results}"
ANALYSIS_DIR="${4:-analysis_output}"
OUTPUT_DIR="${5:-gradcam_output}"
MAX_CORRECT="${6:-50}"
MAX_INCORRECT="${7:-100}"

MODEL_SAFE=$(echo "$MODEL" | sed 's/\//__/g')
RESULTS_FILE="$RESULTS_DIR/$MODEL_SAFE/counterfactual/${DATASET_TYPE}_counterfactual_results.json"
CIRCUIT_FILE="$ANALYSIS_DIR/C_arith.json"

echo "=========================================="
echo "Phase II: Grad-CAM Analysis"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET_TYPE"
echo "Results: $RESULTS_FILE"
echo "Circuit: $CIRCUIT_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Counterfactual results not found at $RESULTS_FILE"
    echo "Run ./run_full_analysis.sh first."
    exit 1
fi

if [ ! -f "$CIRCUIT_FILE" ]; then
    echo "Error: C_arith.json not found at $CIRCUIT_FILE"
    echo "Run comprehensive analysis first (./run_full_analysis.sh) to generate C_arith."
    exit 1
fi

python3 grad_cam.py \
    --model "$MODEL" \
    --results-file "$RESULTS_FILE" \
    --circuit-file "$CIRCUIT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-type "$DATASET_TYPE" \
    --max-correct "$MAX_CORRECT" \
    --max-incorrect "$MAX_INCORRECT"

echo ""
echo "Grad-CAM analysis complete. Check $OUTPUT_DIR/"
