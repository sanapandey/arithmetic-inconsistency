#!/bin/bash
# Full analysis pipeline for counterfactual arithmetic circuits

# Configuration
MODEL="${1:-meta-llama/Llama-2-7b-hf}"
MAX_SAMPLES="${2:-100}"
SPLIT="${3:-test}"

echo "=========================================="
echo "FULL COUNTERFACTUAL CIRCUIT ANALYSIS"
echo "=========================================="
echo "Model: $MODEL"
echo "Max samples per dataset: $MAX_SAMPLES"
echo "Split: $SPLIT"
echo ""

# Step 1: Evaluate on all counterfactual datasets
echo "STEP 1: Evaluating on counterfactual datasets..."
python3 eval_counterfactual.py \
    --model "$MODEL" \
    --dataset all \
    --split "$SPLIT" \
    --max-samples "$MAX_SAMPLES" \
    --log-every 20

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

# Step 2: Run activation patching for each dataset
echo ""
echo "STEP 2: Running activation patching experiments..."

MODEL_SAFE=$(echo "$MODEL" | sed 's/\//__/g')

# activation_patching.py writes to results/patching_results.json (not a subdir)
mkdir -p results/patching_results

for dataset in numeric english spanish italian; do
    RESULTS_FILE="results/$MODEL_SAFE/counterfactual/${dataset}_counterfactual_results.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Patching for $dataset..."
        python3 activation_patching.py \
            --model "$MODEL" \
            --results-file "$RESULTS_FILE" \
            --max-samples "$MAX_SAMPLES" \
            --format-eap-ig
        
        # Save dataset-specific copy (activation_patching writes to results/patching_results.json)
        if [ -f "results/patching_results.json" ]; then
            cp "results/patching_results.json" \
               "results/patching_results/${dataset}_patching_results.json"
            echo "  Saved results/patching_results/${dataset}_patching_results.json"
        fi
    fi
done

# Step 3: Run comprehensive analysis
echo ""
echo "STEP 3: Running comprehensive analysis..."
python3 comprehensive_analysis.py \
    --model "$MODEL" \
    --datasets numeric english spanish italian \
    --split "$SPLIT" \
    --max-samples "$MAX_SAMPLES" \
    --skip-evaluation \
    --skip-patching \
    --output-dir "analysis_output"

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo "Check analysis_output/ for:"
echo "  - ANALYSIS_REPORT.md: Full written report"
echo "  - C_arith.json: Established arithmetic circuit (for Grad-CAM)"
echo "  - *.png: All visualizations"
echo "  - analysis_results.json: Raw analysis data"
echo ""
echo "Next: Run Grad-CAM with ./run_gradcam.sh $MODEL numeric"
echo ""

