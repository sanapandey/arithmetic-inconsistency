# Sample Size Analysis for Activation Patching

## Statistical Considerations

### Effect Size Detection

For activation patching, we're measuring:
- **Binary outcome**: Does patching at layer L change the prediction? (Yes/No)
- **Effect size**: Proportion of samples where patching has an effect

**Power analysis for binary outcomes:**

To detect a difference with:
- **Power = 0.8** (80% chance of detecting true effect)
- **α = 0.05** (5% false positive rate)
- **Effect size = 0.2** (20% of samples show patching effect)

**Minimum sample size needed: ~100-150 samples**

However, this assumes:
- Single comparison
- Medium effect size
- No multiple comparisons correction

### Multiple Comparisons Problem

We're testing **many layers** (typically 20-32 for 8B models):
- If testing 20 layers with α=0.05, expect ~1 false positive by chance
- Need Bonferroni correction: α_adj = 0.05/20 = 0.0025 per layer
- This **increases** required sample size

### Recommended Sample Sizes

#### **Minimum Viable (Quick Test)**
- **50-100 samples**: Can detect large effects (>30% change rate)
- **Use case**: Initial exploration, debugging pipeline
- **Limitation**: May miss subtle but important layers

#### **Recommended (Balanced)**
- **200-300 samples**: Good balance of statistical power and runtime
- **Can detect**: Medium effects (15-20% change rate)
- **Multiple comparisons**: Reasonable protection
- **Runtime**: ~2-4 hours for full pipeline

#### **Robust (Publication Quality)**
- **500-1000 samples**: High statistical power
- **Can detect**: Small but meaningful effects (10-15% change rate)
- **Multiple comparisons**: Well-protected
- **Runtime**: ~5-10 hours for full pipeline

## Practical Recommendations

### For Your Analysis

Given you want to:
1. Answer 6 questions comprehensively
2. Compare across 4 formats (numeric, english, spanish, italian)
3. Identify differentiating circuits
4. Generate publication-quality visualizations

**I recommend: 200-300 samples per dataset**

**Reasoning:**
- **200 samples**: ~95% power to detect 20% effect size per layer
- **300 samples**: ~95% power to detect 15% effect size per layer
- **Across 4 datasets**: 800-1200 total samples (manageable runtime)
- **Statistical rigor**: Sufficient for meaningful conclusions

### Sample Size by Question

**Q1-Q2 (Layer Importance):**
- **200 samples**: Good for ranking layers
- **300 samples**: Better for detecting subtle differences

**Q3 (Format Comparison):**
- **200 samples per format**: Sufficient for correlation analysis
- **300 samples**: More robust cross-format comparisons

**Q4 (Error Fixing):**
- **Depends on error rate**: If model is 80% accurate, need ~200 samples to get ~40 errors
- **Recommendation**: 200-300 samples (ensures ~40-60 errors to analyze)

**Q5 (Differentiating Circuits):**
- **200 samples**: Minimum for correct vs incorrect comparison
- **300 samples**: Better separation, more reliable

**Q6 (Performance with Patching):**
- **200 samples**: Good for accuracy estimates (±5% confidence interval)
- **300 samples**: Tighter confidence intervals (±4%)

## Runtime Estimates

For **meta-llama/Llama-2-7b-hf** on typical GPU (A100/V100):

| Samples | Evaluation Time | Patching Time | Total Time |
|---------|----------------|---------------|------------|
| 50      | ~5-10 min      | ~15-30 min    | ~20-40 min |
| 100     | ~10-20 min     | ~30-60 min     | ~40-80 min |
| 200     | ~20-40 min     | ~60-120 min    | ~1.5-3 hrs |
| 300     | ~30-60 min     | ~90-180 min    | ~2-4 hrs   |
| 500     | ~50-100 min    | ~150-300 min   | ~3.5-7 hrs |

## My Specific Recommendation

**Start with 200 samples per dataset, then scale up if needed:**

```bash
# Initial run (200 samples)
./run_full_analysis.sh meta-llama/Llama-2-7b-hf 200 test

# If results look promising but need more precision:
./run_full_analysis.sh meta-llama/Llama-2-7b-hf 300 test
```

**Why 200 is a good starting point:**
1. **Statistical power**: 95% power to detect 20% effects
2. **Practical runtime**: ~2-3 hours total (manageable)
3. **Error analysis**: With ~80% accuracy, get ~40 errors per dataset (enough for Q4)
4. **Format comparison**: 200 per format = 800 total (good for correlations)
5. **Can always increase**: If effects are borderline, easy to scale up

**When to use more (300-500):**
- If you see interesting but borderline effects at 200
- If preparing for publication (need higher confidence)
- If model accuracy is very high (>90%) - need more samples to get enough errors

**When 100 might be enough:**
- Quick exploratory analysis
- Testing the pipeline
- If you only care about very large effects (>30%)
- Time-constrained scenarios

## Conclusion

**For your comprehensive analysis: 200-300 samples per dataset**

- **200**: Good starting point, balanced power/runtime
- **300**: More robust, better for subtle effects
- **100**: Minimum viable, may miss important layers

I'd suggest starting with **200**, reviewing results, and scaling to **300** if you need more precision for specific questions.

