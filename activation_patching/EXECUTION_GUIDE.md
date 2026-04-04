# Execution Guide: Comprehensive Counterfactual Analysis

## Functional Difference: Script Creation vs. Direct Execution

**When I create a script for you:**
- ✅ Script is ready to run on your machine with full GPU/resources
- ✅ You control execution timing and resource allocation
- ✅ Can debug/iterate locally
- ✅ Full access to your environment (eap-viz, models, etc.)

**When I try to run commands here:**
- ❌ Sandbox limitations prevent model loading (PyTorch segfaults)
- ❌ No GPU access
- ❌ No network access (can't download models)
- ❌ Limited to file operations and code generation

**Conclusion:** You need to run the script on your machine. I've created a complete, ready-to-run script.

## Quick Start

### Option 1: One-Command Execution (Recommended)

```bash
./run_full_analysis.sh meta-llama/Llama-2-7b-hf 200 test
```

This will:
1. Evaluate on all counterfactual datasets (numeric, english, spanish, italian)
2. Run activation patching for each
3. Generate comprehensive analysis with visualizations
4. Create report answering all 6 questions

### Option 2: Step-by-Step (For Debugging)

```bash
# Step 1: Evaluate on counterfactual datasets
python eval_counterfactual.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset all \
    --split test \
    --max-samples 200

# Step 2: Run activation patching (for each dataset)
for dataset in numeric english spanish italian; do
    python activation_patching.py \
        --model meta-llama/Llama-2-7b-hf \
        --results-file results/meta-llama__Llama-2-7b-hf/counterfactual/${dataset}_counterfactual_results.json \
        --max-samples 200 \
        --format-eap-ig
done

# Step 3: Comprehensive analysis
python comprehensive_analysis.py \
    --model meta-llama/Llama-2-7b-hf \
    --datasets numeric english spanish italian \
    --split test \
    --max-samples 200 \
    --skip-evaluation \
    --skip-patching \
    --output-dir analysis_output
```

## What the Analysis Will Answer

### Question 1: Which layers are most causally important for arithmetic?
- **Output**: Ranked list of layers by patching effect
- **Visualization**: Bar chart of layer importance

### Question 2: Do early vs late layers matter differently?
- **Output**: Average importance for early/middle/late layers
- **Visualization**: Comparison bar chart

### Question 3: Are circuits different for numeric vs verbal formats?
- **Output**: Correlation matrix between formats, layer importance heatmap
- **Visualization**: Heatmap showing layer importance across formats

### Question 4: Can you fix errors by patching at specific layers?
- **Output**: Error fix rates per layer
- **Visualization**: Top error-fixing layers chart

### Question 5: Which circuits differentiate correct from incorrect? Same across languages?
- **Output**: Layers that show different effects for correct vs incorrect predictions
- **Visualization**: Side-by-side comparison charts for each format

### Question 6: How does performance change when patching those circuits?
- **Output**: Accuracy comparison (original vs with patching)
- **Visualization**: Performance comparison bar chart

## Output Structure

After running, you'll have:

```
analysis_output/
├── ANALYSIS_REPORT.md          # Comprehensive written report
├── analysis_results.json        # Raw analysis data
├── layer_analysis.png           # Q1-Q2 visualizations
├── format_comparison_heatmap.png # Q3 visualization
├── differentiating_circuits.png  # Q5 visualization
├── patching_performance.png      # Q6 visualization
└── eap_*.png/html               # EAP-viz visualizations (if available)

results/
└── meta-llama__Llama-2-7b-hf/
    └── counterfactual/
        ├── numeric_counterfactual_results.json
        ├── english_counterfactual_results.json
        ├── spanish_counterfactual_results.json
        └── italian_counterfactual_results.json

results/
└── patching_results/
    ├── numeric_patching_results.json
    ├── english_patching_results.json
    ├── spanish_patching_results.json
    └── italian_patching_results.json
```

## Expected Runtime

- **Evaluation**: ~10-30 minutes (depending on GPU and sample size)
- **Activation Patching**: ~30-60 minutes (most time-consuming step)
- **Analysis & Visualization**: ~1-2 minutes

**Total**: ~1-2 hours for 100 samples per dataset

## Troubleshooting

### Model Loading Issues
- Ensure you have enough GPU memory (8B model needs ~16GB)
- Try `--dtype float16` or `--dtype bfloat16` to reduce memory

### EAP-viz Not Found
- The script will fall back to matplotlib/seaborn automatically
- Check eap-viz installation: `python -c "import eap_viz; print('OK')"`

### Out of Memory
- Reduce `--max-samples` (start with 20-50)
- Process one dataset at a time
- Use smaller batch sizes in the code

## Next Steps After Running

1. **Review `ANALYSIS_REPORT.md`** for key findings
2. **Examine visualizations** to understand layer importance patterns
3. **Check `analysis_results.json`** for detailed metrics
4. **Use EAP-viz visualizations** for interactive exploration (if generated)

## Questions?

The script is designed to be self-contained and will provide detailed output at each step. If you encounter issues, check the error messages - they're designed to be informative.

