# Comprehensive Counterfactual Analysis Report

## Executive Summary

This report analyzes activation patching experiments on counterfactual arithmetic datasets to identify causal circuits responsible for arithmetic reasoning across multiple formats.

## Key Findings

### 1. Which layers are most causally important for arithmetic?

**Top 5 Most Important Layers:**

- model.layers.0: 0.970 patching effect
- model.layers.1: 0.970 patching effect
- model.layers.10: 0.970 patching effect
- model.layers.11: 0.970 patching effect
- model.layers.12: 0.970 patching effect

### 2. Do early vs late layers matter differently?

- **Early layers**: 0.970 average effect
- **Middle layers**: 0.970 average effect
- **Late layers**: 0.970 average effect

### 3. Are circuits different for numeric vs verbal formats?

**Format Correlations:**

- numeric_vs_english: nan
- numeric_vs_spanish: nan
- numeric_vs_italian: nan
- english_vs_spanish: nan
- english_vs_italian: nan
- spanish_vs_italian: nan

### 4. Can you fix errors by patching at specific layers?

**Top 5 Error-Fixing Layers:**

- model.layers.0: 0.000 fix rate
- model.layers.1: 0.000 fix rate
- model.layers.10: 0.000 fix rate
- model.layers.11: 0.000 fix rate
- model.layers.12: 0.000 fix rate

Total errors analyzed: 3200

### 5. Which circuits differentiate correct from incorrect?

**Common differentiating layers across formats:** 10

Layers:
- model.layers.7
- model.layers.0
- model.layers.13
- model.layers.16
- model.layers.28
- model.layers.26
- model.layers.14
- model.layers.21
- model.layers.9
- model.layers.15

### 6. How does performance change when patching those circuits?

## Methodology

1. Evaluated model on counterfactual pairs (x, x') where x' is x with signs flipped
2. Performed activation patching: patched activations from x' into x at each layer
3. Measured causal effect: how much patching changes predictions
4. Identified important circuits: layers with highest patching effects
5. Compared across formats: numeric, English, Spanish, Italian
6. Tested intervention: evaluated performance when patching at key layers

## Visualizations

See generated PNG files for detailed visualizations:
- `layer_analysis.png`: Overall layer importance and early/late analysis
- `format_comparison_heatmap.png`: Layer importance across formats
- `differentiating_circuits.png`: Circuits that differentiate correct/incorrect
- `patching_performance.png`: Performance with circuit patching

