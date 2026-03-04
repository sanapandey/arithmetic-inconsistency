#!/usr/bin/env python3
"""
Comprehensive Counterfactual Analysis with Activation Patching

This script answers 6 key questions:
1. Which layers are most causally important for arithmetic?
2. Do early vs late layers matter differently?
3. Are circuits different for numeric vs verbal formats?
4. Can you fix errors by patching at specific layers?
5. Which circuits differentiate correct from incorrect? Same across languages?
6. How does performance change when patching those circuits during evaluation?

Generates comprehensive visualizations and analysis report.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import eap-viz (try multiple possible names)
EAP_VIZ_AVAILABLE = False
eap_viz = None
try:
    import eap_viz
    EAP_VIZ_AVAILABLE = True
except ImportError:
    try:
        import eap_ig as eap_viz
        EAP_VIZ_AVAILABLE = True
    except ImportError:
        try:
            from eap import viz as eap_viz
            EAP_VIZ_AVAILABLE = True
        except ImportError:
            print("Warning: eap-viz not found. Will use matplotlib/seaborn for visualizations.")

# Import our evaluation and patching modules
import sys
sys.path.append('.')
from eval_counterfactual import (
    load_counterfactual_dataset,
    evaluate_counterfactual_pair,
    COUNTERFACTUAL_DATASETS
)
from activation_patching import (
    patch_activations,
    get_layer_names
)


def load_patching_results(results_file: str) -> List[Dict]:
    """Load activation patching results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('results', [])


def analyze_layer_importance(patching_results: List[Dict]) -> Dict[str, float]:
    """
    Question 1: Which layers are most causally important?

    Uses two signals and takes the mean so scores vary by layer:
    - Causal effect: fraction of items where patching changed the prediction.
    - Correct-after-patch: fraction of items where patched output matched the expected answer.
    """
    layer_effects = defaultdict(list)
    layer_correct = defaultdict(list)

    for item in patching_results:
        y_expected = item.get('y_expected', '').strip()
        original_pred = item.get('original_predicted_x', '').strip()

        for layer_name, result in item.get('patching_results', {}).items():
            if 'error' in result:
                continue

            patched_output = result.get('patched_output', '').strip()

            # Effect: did patching change the prediction?
            effect = 1.0 if (original_pred != patched_output) else 0.0
            layer_effects[layer_name].append(effect)

            # Correct after patch: did patched output match expected?
            correct = 1.0 if (patched_output == y_expected) else 0.0
            layer_correct[layer_name].append(correct)

    # Per-layer importance = mean of (effect, correct_after_patch) so layers differ
    all_layers = set(layer_effects.keys()) | set(layer_correct.keys())
    layer_importance = {}
    for layer in all_layers:
        effects = layer_effects.get(layer, [])
        corrects = layer_correct.get(layer, [])
        if effects or corrects:
            mean_effect = np.mean(effects) if effects else 0.0
            mean_correct = np.mean(corrects) if corrects else 0.0
            layer_importance[layer] = 0.5 * mean_effect + 0.5 * mean_correct
        else:
            layer_importance[layer] = 0.0

    return layer_importance


def analyze_early_vs_late_layers(layer_importance: Dict[str, float]) -> Dict[str, float]:
    """
    Question 2: Do early vs late layers matter differently?
    
    Returns: Dict with 'early', 'middle', 'late' importance scores
    """
    # Sort layers and divide into thirds
    sorted_layers = sorted(layer_importance.items(), 
                          key=lambda x: int(x[0].split('.')[-1]) if x[0].split('.')[-1].isdigit() else 0)
    
    n_layers = len(sorted_layers)
    early_end = n_layers // 3
    middle_end = 2 * n_layers // 3
    
    early_layers = sorted_layers[:early_end]
    middle_layers = sorted_layers[early_end:middle_end]
    late_layers = sorted_layers[middle_end:]
    
    return {
        'early': np.mean([imp for _, imp in early_layers]) if early_layers else 0,
        'middle': np.mean([imp for _, imp in middle_layers]) if middle_layers else 0,
        'late': np.mean([imp for _, imp in late_layers]) if late_layers else 0,
        'early_layers': [name for name, _ in early_layers],
        'middle_layers': [name for name, _ in middle_layers],
        'late_layers': [name for name, _ in late_layers],
    }


def compare_formats(all_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Question 3: Are circuits different for numeric vs verbal formats?
    
    Args:
        all_results: Dict mapping dataset_type -> patching_results
    
    Returns: Comparison analysis
    """
    format_importance = {}
    
    for dataset_type, results in all_results.items():
        layer_importance = analyze_layer_importance(results)
        format_importance[dataset_type] = layer_importance
    
    # Find common important layers across formats
    all_layers = set()
    for imp in format_importance.values():
        all_layers.update(imp.keys())
    
    # Calculate correlation between formats
    correlations = {}
    format_names = list(format_importance.keys())
    
    for i, fmt1 in enumerate(format_names):
        for fmt2 in format_names[i+1:]:
            # Get common layers
            common_layers = set(format_importance[fmt1].keys()) & set(format_importance[fmt2].keys())
            if len(common_layers) > 1:
                scores1 = [format_importance[fmt1][l] for l in common_layers]
                scores2 = [format_importance[fmt2][l] for l in common_layers]
                corr = np.corrcoef(scores1, scores2)[0, 1]
                correlations[f"{fmt1}_vs_{fmt2}"] = corr
    
    return {
        'format_importance': format_importance,
        'correlations': correlations,
        'common_layers': list(all_layers)
    }


def analyze_error_fixing(patching_results: List[Dict]) -> Dict:
    """
    Question 4: Can you fix errors by patching at specific layers?
    
    Analyzes cases where original prediction was wrong but patching fixed it
    """
    error_fixes = defaultdict(list)
    
    for item in patching_results:
        original_correct = item.get('correct_x', False)
        y_expected = item.get('y_expected', '').strip()
        original_pred = item.get('original_predicted_x', '').strip()
        
        # Only analyze cases where original was wrong
        if original_correct:
            continue
        
        for layer_name, result in item.get('patching_results', {}).items():
            if 'error' in result:
                continue
            
            patched_output = result.get('patched_output', '').strip()
            
            # Check if patching fixed the error
            patched_correct = (patched_output == y_expected)
            
            if patched_correct:
                error_fixes[layer_name].append(1.0)  # Fixed!
            else:
                error_fixes[layer_name].append(0.0)  # Still wrong
    
    # Calculate fix rate per layer
    fix_rates = {
        layer: np.mean(fixes) if fixes else 0.0
        for layer, fixes in error_fixes.items()
    }
    
    return {
        'fix_rates': fix_rates,
        'total_errors_analyzed': sum(len(fixes) for fixes in error_fixes.values())
    }


def identify_differentiating_circuits(all_results: Dict[str, List[Dict]]) -> Dict:
    """
    Question 5: Which circuits differentiate correct from incorrect?
    Are these circuits the same across languages?
    
    Identifies layers that are more important for correct vs incorrect predictions
    """
    differentiating_layers = {}
    
    for dataset_type, results in all_results.items():
        correct_importance = defaultdict(list)
        incorrect_importance = defaultdict(list)
        
        for item in results:
            original_correct = item.get('correct_x', False)
            
            for layer_name, result in item.get('patching_results', {}).items():
                if 'error' in result:
                    continue
                
                original_pred = item.get('original_predicted_x', '').strip()
                patched_output = result.get('patched_output', '').strip()
                effect = 1.0 if (original_pred != patched_output) else 0.0
                
                if original_correct:
                    correct_importance[layer_name].append(effect)
                else:
                    incorrect_importance[layer_name].append(effect)
        
        # Calculate average importance for correct vs incorrect
        correct_avg = {layer: np.mean(effects) for layer, effects in correct_importance.items()}
        incorrect_avg = {layer: np.mean(effects) for layer, effects in incorrect_importance.items()}
        
        # Find layers that differentiate (high effect for one, low for other)
        differentiating = {}
        all_layers = set(correct_avg.keys()) | set(incorrect_avg.keys())
        
        for layer in all_layers:
            c_imp = correct_avg.get(layer, 0)
            i_imp = incorrect_avg.get(layer, 0)
            diff = abs(c_imp - i_imp)
            differentiating[layer] = {
                'correct_importance': c_imp,
                'incorrect_importance': i_imp,
                'difference': diff
            }
        
        differentiating_layers[dataset_type] = differentiating
    
    # Find common differentiating layers across languages
    if len(differentiating_layers) > 1:
        # Get top differentiating layers for each format
        top_layers_per_format = {}
        for fmt, layers in differentiating_layers.items():
            sorted_layers = sorted(layers.items(), 
                                 key=lambda x: x[1]['difference'], 
                                 reverse=True)
            top_layers_per_format[fmt] = [name for name, _ in sorted_layers[:10]]
        
        # Find intersection
        if top_layers_per_format:
            common_differentiating = set.intersection(*[set(layers) for layers in top_layers_per_format.values()])
        else:
            common_differentiating = set()
    else:
        common_differentiating = set()
    
    return {
        'differentiating_layers': differentiating_layers,
        'common_differentiating_layers': list(common_differentiating)
    }


def evaluate_with_patching(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_type: str,
    layers_to_patch: List[str],
    max_new_tokens: int = 8
) -> Dict:
    """
    Question 6: How does performance change when patching those circuits during evaluation?
    
    Evaluates model while patching at identified important layers
    """
    results = []
    correct = 0
    total = len(dataset)
    
    for item in tqdm(dataset, desc=f"Evaluating with patching ({dataset_type})"):
        x_prompt = item['x']
        x_prime_prompt = item['x_prime']
        y_expected = item['y_expected']
        
        # Patch at each important layer and measure effect
        best_patched_output = None
        best_layer = None
        
        for layer_name in layers_to_patch:
            try:
                patched_output, _ = patch_activations(
                    model, tokenizer,
                    source_prompt=x_prime_prompt,
                    target_prompt=x_prompt,
                    layer_name=layer_name,
                    max_new_tokens=max_new_tokens
                )
                
                # Check if this patching improves prediction
                if best_patched_output is None:
                    best_patched_output = patched_output
                    best_layer = layer_name
                else:
                    # Prefer patching that gives correct answer
                    if patched_output.strip() == y_expected.strip():
                        best_patched_output = patched_output
                        best_layer = layer_name
            except:
                continue
        
        # Evaluate best patched output
        from eval_counterfactual import extract_answer, check_answer
        predicted = extract_answer(best_patched_output or '', dataset_type)
        is_correct = check_answer(predicted, y_expected, dataset_type)
        
        if is_correct:
            correct += 1
        
        results.append({
            'id': item['id'],
            'original_predicted': item.get('original_predicted_x', ''),
            'patched_predicted': predicted,
            'expected': y_expected,
            'correct': is_correct,
            'best_layer': best_layer
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def create_eap_viz_visualizations(analysis_results: Dict, output_dir: Path):
    """Create visualizations using eap-viz if available"""
    if not EAP_VIZ_AVAILABLE:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Prepare data for eap-viz
        if 'layer_importance' in analysis_results:
            layer_data = analysis_results['layer_importance']
            
            # Format for eap-viz (adjust based on actual eap-viz API)
            if hasattr(eap_viz, 'plot_layer_importance'):
                eap_viz.plot_layer_importance(layer_data, 
                                             save_path=str(output_dir / 'eap_layer_importance.png'))
            elif hasattr(eap_viz, 'visualize'):
                eap_viz.visualize(layer_data, 
                                 output_path=str(output_dir / 'eap_visualization.html'))
        
        print("EAP-viz visualizations created")
    except Exception as e:
        print(f"Warning: Could not create eap-viz visualizations: {e}")
        print("Falling back to matplotlib visualizations")


def create_visualizations(analysis_results: Dict, output_dir: Path):
    """Create comprehensive visualizations using matplotlib/seaborn and eap-viz if available"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Try eap-viz first
    if EAP_VIZ_AVAILABLE:
        create_eap_viz_visualizations(analysis_results, output_dir)
    
    # 1. Layer Importance Heatmap
    if 'layer_importance' in analysis_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall layer importance
        ax = axes[0, 0]
        layer_imp = analysis_results['layer_importance']
        layers = sorted(layer_imp.keys(), 
                       key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        importances = [layer_imp[l] for l in layers]
        
        ax.barh(range(len(layers)), importances)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([l.split('.')[-1] if '.' in l else l for l in layers], fontsize=8)
        ax.set_xlabel('Patching Effect (fraction of predictions changed)', fontsize=12)
        ax.set_title('Layer Importance for Arithmetic Reasoning', fontsize=14, weight='bold')
        ax.invert_yaxis()
        
        # Plot 2: Early vs Middle vs Late
        ax = axes[0, 1]
        early_late = analysis_results.get('early_vs_late', {})
        if early_late:
            categories = ['Early', 'Middle', 'Late']
            values = [early_late.get('early', 0), early_late.get('middle', 0), early_late.get('late', 0)]
            ax.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Average Patching Effect', fontsize=12)
            ax.set_title('Layer Importance by Position', fontsize=14, weight='bold')
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
        
        # Plot 3: Format Comparison
        ax = axes[1, 0]
        format_comparison = analysis_results.get('format_comparison', {})
        if format_comparison and 'format_importance' in format_comparison:
            formats = list(format_comparison['format_importance'].keys())
            # Average importance per format
            avg_importance = {
                fmt: np.mean(list(imp.values())) 
                for fmt, imp in format_comparison['format_importance'].items()
            }
            ax.bar(formats, [avg_importance[f] for f in formats], color='steelblue')
            ax.set_ylabel('Average Layer Importance', fontsize=12)
            ax.set_title('Circuit Importance by Format', fontsize=14, weight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Error Fixing Rates
        ax = axes[1, 1]
        error_fixing = analysis_results.get('error_fixing', {})
        if error_fixing and 'fix_rates' in error_fixing:
            fix_rates = error_fixing['fix_rates']
            top_layers = sorted(fix_rates.items(), key=lambda x: x[1], reverse=True)[:10]
            if top_layers:
                layers, rates = zip(*top_layers)
                ax.barh(range(len(layers)), rates, color='coral')
                ax.set_yticks(range(len(layers)))
                ax.set_yticklabels([l.split('.')[-1] if '.' in l else l for l in layers], fontsize=8)
                ax.set_xlabel('Error Fix Rate', fontsize=12)
                ax.set_title('Top Layers for Error Correction', fontsize=14, weight='bold')
                ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'layer_analysis.png', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'layer_analysis.png'}")
    
    # 2. Format Comparison Heatmap
    if 'format_comparison' in analysis_results:
        format_imp = analysis_results['format_comparison'].get('format_importance', {})
        if format_imp:
            # Create matrix of layer importance across formats
            all_layers = set()
            for imp in format_imp.values():
                all_layers.update(imp.keys())
            
            formats = list(format_imp.keys())
            layers = sorted(all_layers, 
                          key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
            
            matrix = []
            for fmt in formats:
                row = [format_imp[fmt].get(layer, 0) for layer in layers]
                matrix.append(row)
            
            plt.figure(figsize=(max(12, len(layers) * 0.3), 6))
            sns.heatmap(matrix, 
                       xticklabels=[l.split('.')[-1] if '.' in l else l for l in layers],
                       yticklabels=formats,
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Patching Effect'})
            plt.title('Layer Importance Across Formats', fontsize=16, weight='bold', pad=20)
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Format', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / 'format_comparison_heatmap.png', bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'format_comparison_heatmap.png'}")
    
    # 3. Differentiating Circuits
    if 'differentiating_circuits' in analysis_results:
        diff_circuits = analysis_results['differentiating_circuits']
        if 'differentiating_layers' in diff_circuits:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            dataset_types = list(diff_circuits['differentiating_layers'].keys())
            
            for idx, dataset_type in enumerate(dataset_types[:4]):
                ax = axes[idx // 2, idx % 2]
                layers_data = diff_circuits['differentiating_layers'][dataset_type]
                
                # Get top 10 differentiating layers
                sorted_layers = sorted(layers_data.items(), 
                                     key=lambda x: x[1]['difference'], 
                                     reverse=True)[:10]
                
                if sorted_layers:
                    layers, data = zip(*sorted_layers)
                    correct_imp = [d['correct_importance'] for d in data]
                    incorrect_imp = [d['incorrect_importance'] for d in data]
                    
                    x = np.arange(len(layers))
                    width = 0.35
                    ax.bar(x - width/2, correct_imp, width, label='Correct', color='green', alpha=0.7)
                    ax.bar(x + width/2, incorrect_imp, width, label='Incorrect', color='red', alpha=0.7)
                    
                    ax.set_ylabel('Patching Effect', fontsize=10)
                    ax.set_title(f'{dataset_type.capitalize()}: Correct vs Incorrect', fontsize=12, weight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([l.split('.')[-1] if '.' in l else l for l in layers], 
                                      rotation=45, ha='right', fontsize=8)
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'differentiating_circuits.png', bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'differentiating_circuits.png'}")
    
    # 4. Performance with Patching
    if 'patching_performance' in analysis_results:
        perf_data = analysis_results['patching_performance']
        
        plt.figure(figsize=(10, 6))
        formats = []
        original_acc = []
        patched_acc = []
        
        for fmt, data in perf_data.items():
            formats.append(fmt)
            original_acc.append(data.get('original_accuracy', 0) * 100)
            patched_acc.append(data.get('patched_accuracy', 0) * 100)
        
        x = np.arange(len(formats))
        width = 0.35
        plt.bar(x - width/2, original_acc, width, label='Original', color='steelblue', alpha=0.7)
        plt.bar(x + width/2, patched_acc, width, label='With Patching', color='coral', alpha=0.7)
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Performance: Original vs With Circuit Patching', fontsize=14, weight='bold')
        plt.xticks(x, formats, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'patching_performance.png', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'patching_performance.png'}")


def generate_report(analysis_results: Dict, output_dir: Path):
    """Generate comprehensive analysis report"""
    report_path = output_dir / 'ANALYSIS_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Counterfactual Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes activation patching experiments on counterfactual arithmetic datasets ")
        f.write("to identify causal circuits responsible for arithmetic reasoning across multiple formats.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Question 1
        f.write("### 1. Which layers are most causally important for arithmetic?\n\n")
        if 'layer_importance' in analysis_results:
            layer_imp = analysis_results['layer_importance']
            top_layers = sorted(layer_imp.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("**Top 5 Most Important Layers:**\n\n")
            for layer, importance in top_layers:
                f.write(f"- {layer}: {importance:.3f} patching effect\n")
            f.write("\n")
        
        # Question 2
        f.write("### 2. Do early vs late layers matter differently?\n\n")
        if 'early_vs_late' in analysis_results:
            evl = analysis_results['early_vs_late']
            f.write(f"- **Early layers**: {evl.get('early', 0):.3f} average effect\n")
            f.write(f"- **Middle layers**: {evl.get('middle', 0):.3f} average effect\n")
            f.write(f"- **Late layers**: {evl.get('late', 0):.3f} average effect\n\n")
        
        # Question 3
        f.write("### 3. Are circuits different for numeric vs verbal formats?\n\n")
        if 'format_comparison' in analysis_results:
            fc = analysis_results['format_comparison']
            if 'correlations' in fc:
                f.write("**Format Correlations:**\n\n")
                for pair, corr in fc['correlations'].items():
                    f.write(f"- {pair}: {corr:.3f}\n")
                f.write("\n")
        
        # Question 4
        f.write("### 4. Can you fix errors by patching at specific layers?\n\n")
        if 'error_fixing' in analysis_results:
            ef = analysis_results['error_fixing']
            if 'fix_rates' in ef:
                top_fixers = sorted(ef['fix_rates'].items(), key=lambda x: x[1], reverse=True)[:5]
                f.write("**Top 5 Error-Fixing Layers:**\n\n")
                for layer, rate in top_fixers:
                    f.write(f"- {layer}: {rate:.3f} fix rate\n")
                f.write(f"\nTotal errors analyzed: {ef.get('total_errors_analyzed', 0)}\n\n")
        
        # Question 5
        f.write("### 5. Which circuits differentiate correct from incorrect?\n\n")
        if 'differentiating_circuits' in analysis_results:
            dc = analysis_results['differentiating_circuits']
            if 'common_differentiating_layers' in dc:
                common = dc['common_differentiating_layers']
                f.write(f"**Common differentiating layers across formats:** {len(common)}\n\n")
                if common:
                    f.write("Layers:\n")
                    for layer in common[:10]:
                        f.write(f"- {layer}\n")
                    f.write("\n")
        
        # Question 6
        f.write("### 6. How does performance change when patching those circuits?\n\n")
        if 'patching_performance' in analysis_results:
            pp = analysis_results['patching_performance']
            f.write("**Performance Comparison:**\n\n")
            for fmt, data in pp.items():
                orig = data.get('original_accuracy', 0) * 100
                patched = data.get('patched_accuracy', 0) * 100
                change = patched - orig
                f.write(f"- **{fmt}**: Original: {orig:.2f}%, With Patching: {patched:.2f}% ")
                f.write(f"(Change: {change:+.2f}%)\n")
            f.write("\n")
        
        f.write("## Methodology\n\n")
        f.write("1. Evaluated model on counterfactual pairs (x, x') where x' is x with signs flipped\n")
        f.write("2. Performed activation patching: patched activations from x' into x at each layer\n")
        f.write("3. Measured causal effect: how much patching changes predictions\n")
        f.write("4. Identified important circuits: layers with highest patching effects\n")
        f.write("5. Compared across formats: numeric, English, Spanish, Italian\n")
        f.write("6. Tested intervention: evaluated performance when patching at key layers\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("See generated PNG files for detailed visualizations:\n")
        f.write("- `layer_analysis.png`: Overall layer importance and early/late analysis\n")
        f.write("- `format_comparison_heatmap.png`: Layer importance across formats\n")
        f.write("- `differentiating_circuits.png`: Circuits that differentiate correct/incorrect\n")
        f.write("- `patching_performance.png`: Performance with circuit patching\n\n")
    
    print(f"Report saved to: {report_path}")


def extract_arithmetic_circuit(analysis_results: Dict) -> Dict:
    """
    Extract C_arith (established arithmetic circuit) from analysis results.
    
    The circuit is defined as top layers by importance across numeric, English,
    Spanish, and Italian formats. Used by Grad-CAM for D_sal computation.
    
    Returns:
        Dict with per_format circuits, combined circuit, and layer importance
    """
    c_arith = {
        "per_format": {},
        "combined": [],
        "layer_importance_per_format": {},
        "top_k": 10,  # Number of top layers per format
    }
    
    # Per-format circuit from format_comparison
    if "format_comparison" in analysis_results:
        fc = analysis_results["format_comparison"]
        format_importance = fc.get("format_importance", {})
        top_k = c_arith["top_k"]
        
        all_format_layers = []
        for fmt, layer_imp in format_importance.items():
            # Top K layers by importance (descending)
            sorted_layers = sorted(
                layer_imp.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            circuit_layers = [name for name, _ in sorted_layers]
            c_arith["per_format"][fmt] = circuit_layers
            c_arith["layer_importance_per_format"][fmt] = dict(sorted_layers)
            all_format_layers.append(set(circuit_layers))
        
        # Combined circuit: intersection of format circuits (layers common to all)
        if all_format_layers:
            c_arith["combined"] = list(set.intersection(*all_format_layers))
            # If intersection is empty, use union of top layers from each format
            if not c_arith["combined"] and all_format_layers:
                c_arith["combined"] = list(set.union(*all_format_layers))
    
    # Fallback: use layer_importance and common_differentiating_layers
    if not c_arith["per_format"] and "layer_importance" in analysis_results:
        layer_imp = analysis_results["layer_importance"]
        sorted_layers = sorted(
            layer_imp.items(), key=lambda x: x[1], reverse=True
        )[:c_arith["top_k"]]
        c_arith["combined"] = [name for name, _ in sorted_layers]
        c_arith["per_format"]["numeric"] = c_arith["combined"]
        c_arith["layer_importance_per_format"]["numeric"] = dict(sorted_layers)
    
    if "differentiating_circuits" in analysis_results:
        dc = analysis_results["differentiating_circuits"]
        common = dc.get("common_differentiating_layers", [])
        if common and not c_arith["combined"]:
            c_arith["combined"] = common
    
    return c_arith


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive counterfactual analysis with activation patching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['numeric', 'english', 'spanish', 'italian'],
                        help='Datasets to analyze')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max samples per dataset (for speed)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory with existing results (if any)')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Output directory for analysis')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation, use existing results')
    parser.add_argument('--skip-patching', action='store_true',
                        help='Skip patching, use existing patching results')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Max samples: {args.max_samples}")
    print()
    
    # Load model if needed
    model = None
    tokenizer = None
    
    if not args.skip_evaluation or not args.skip_patching:
        print("Loading model...")
        dtype_map = {
            'auto': 'auto',
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # Use device_map="auto" if accelerate is available (uses GPU/MPS when available)
        try:
            import accelerate
            device_map = "auto"
        except ImportError:
            print("Warning: accelerate not found. Using CPU. Install with: pip install accelerate")
            device_map = None
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map=device_map,
            trust_remote_code=True,
        )
        
        if device_map is None:
            model = model.to('cpu')
        
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded.\n")
    
    # Run analysis for each dataset
    all_patching_results = {}
    all_eval_results = {}
    
    for dataset_type in args.datasets:
        print(f"\n{'='*70}")
        print(f"Analyzing: {dataset_type}")
        print(f"{'='*70}\n")
        
        dataset_path = COUNTERFACTUAL_DATASETS.get(dataset_type)
        if not dataset_path or not Path(dataset_path).exists():
            print(f"Warning: Dataset {dataset_type} not found, skipping...")
            continue
        
        # Load or run evaluation
        results_file = Path(args.results_dir) / args.model.replace('/', '__') / 'counterfactual' / f'{dataset_type}_counterfactual_results.json'
        
        if args.skip_evaluation and results_file.exists():
            print(f"Loading existing results from {results_file}")
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
            all_eval_results[dataset_type] = eval_results
        else:
            print("Running evaluation...")
            from eval_counterfactual import evaluate_counterfactual_dataset
            dataset = load_counterfactual_dataset(dataset_path, args.split, args.max_samples)
            eval_results = evaluate_counterfactual_dataset(
                model, tokenizer, dataset, dataset_type,
                max_new_tokens=8,
                collect_activations=False,
                log_every=20
            )
            all_eval_results[dataset_type] = eval_results
        
        # Load or run patching
        patching_file = Path(args.results_dir) / 'patching_results' / f'{dataset_type}_patching_results.json'
        
        if args.skip_patching and patching_file.exists():
            print(f"Loading existing patching results from {patching_file}")
            with open(patching_file, 'r') as f:
                patching_results = json.load(f)
            all_patching_results[dataset_type] = patching_results
        else:
            print("Running activation patching...")
            from activation_patching import run_patching_experiment
            layer_names = get_layer_names(model)
            patching_results = run_patching_experiment(
                model, tokenizer,
                eval_results['results'],
                layer_names[:min(10, len(layer_names))],  # Limit layers for speed
                max_samples=args.max_samples,
                max_new_tokens=8
            )
            all_patching_results[dataset_type] = patching_results
            
            # Save patching results
            patching_file.parent.mkdir(parents=True, exist_ok=True)
            with open(patching_file, 'w') as f:
                json.dump(patching_results, f, indent=2)
    
    # Run comprehensive analysis
    print(f"\n{'='*70}")
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print(f"{'='*70}\n")
    
    analysis_results = {}
    
    # Question 1 & 2: Layer importance (use first dataset as reference)
    if all_patching_results:
        first_dataset = list(all_patching_results.keys())[0]
        first_results = all_patching_results[first_dataset]
        
        print("Q1-Q2: Analyzing layer importance...")
        layer_importance = analyze_layer_importance(first_results)
        analysis_results['layer_importance'] = layer_importance
        
        early_vs_late = analyze_early_vs_late_layers(layer_importance)
        analysis_results['early_vs_late'] = early_vs_late
    
    # Question 3: Format comparison
    if len(all_patching_results) > 1:
        print("Q3: Comparing circuits across formats...")
        format_comparison = compare_formats(all_patching_results)
        analysis_results['format_comparison'] = format_comparison
    
    # Question 4: Error fixing
    if all_patching_results:
        print("Q4: Analyzing error fixing...")
        error_fixing = analyze_error_fixing(first_results)
        analysis_results['error_fixing'] = error_fixing
    
    # Question 5: Differentiating circuits
    if all_patching_results:
        print("Q5: Identifying differentiating circuits...")
        differentiating_circuits = identify_differentiating_circuits(all_patching_results)
        analysis_results['differentiating_circuits'] = differentiating_circuits
    
    # Question 6: Performance with patching
    if model and tokenizer and all_patching_results:
        print("Q6: Evaluating performance with circuit patching...")
        patching_performance = {}
        
        # Get top important layers
        if 'layer_importance' in analysis_results:
            top_layers = sorted(analysis_results['layer_importance'].items(),
                              key=lambda x: x[1], reverse=True)[:5]
            layers_to_patch = [layer for layer, _ in top_layers]
        else:
            layers_to_patch = []
        
        for dataset_type in args.datasets:
            dataset_path = COUNTERFACTUAL_DATASETS.get(dataset_type)
            if not dataset_path:
                continue
            
            dataset = load_counterfactual_dataset(dataset_path, args.split, min(50, args.max_samples))
            eval_results = all_eval_results.get(dataset_type, {})
            original_accuracy = eval_results.get('accuracy_x', 0)
            
            if layers_to_patch:
                patched_results = evaluate_with_patching(
                    model, tokenizer, dataset, dataset_type, layers_to_patch
                )
                patched_accuracy = patched_results['accuracy']
            else:
                patched_accuracy = original_accuracy
            
            patching_performance[dataset_type] = {
                'original_accuracy': original_accuracy,
                'patched_accuracy': patched_accuracy,
                'layers_patched': layers_to_patch
            }
        
        analysis_results['patching_performance'] = patching_performance
    
    # Save analysis results (output_dir already created at start of main)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / 'analysis_results.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"Saved analysis results to {analysis_path.absolute()}")
    
    # Extract and save C_arith (established arithmetic circuit) for Grad-CAM
    c_arith = extract_arithmetic_circuit(analysis_results)
    circuit_path = output_dir / 'C_arith.json'
    with open(circuit_path, 'w') as f:
        json.dump(c_arith, f, indent=2)
    print(f"Saved arithmetic circuit C_arith to {circuit_path.absolute()}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(analysis_results, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(analysis_results, output_dir)
    print(f"Saved report to {(output_dir / 'ANALYSIS_REPORT.md').absolute()}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"- analysis_results.json: Full analysis data")
    print(f"- *.png: Visualizations")
    print(f"- ANALYSIS_REPORT.md: Comprehensive report")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

