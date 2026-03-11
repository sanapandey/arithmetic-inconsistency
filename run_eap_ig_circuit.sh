#!/bin/bash
# Run EAP-IG circuit analysis (aligned with https://github.com/hannamw/EAP-IG ioi.ipynb)

# Requires: pip install transformer_lens; pip install git+https://github.com/hannamw/EAP-IG
# Optional: pip install pygraphviz  (for circuit graph visualization)

# Default: Llama 3 8B (set MODEL=gpt2-small for a quick CPU-friendly run)
MODEL="${1:-meta-llama/Meta-Llama-3-8B}"
DATASET="${2:-numeric}"
MAX_SAMPLES="${3:-100}"

echo "=========================================="
echo "EAP-IG Circuit Analysis"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Max samples: $MAX_SAMPLES"
echo ""

python3 circuit_analysis_eap_ig.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --max-samples "$MAX_SAMPLES" \
  --method EAP-IG-inputs \
  --ig-steps 5 \
  --top-n 20000 \
  --batch-size 10 \
  --output-dir analysis_output

echo ""
echo "Output: analysis_output/<dataset>/C_arith.json, eap_graph.json"
