#!/usr/bin/env python3
"""
Visualize layer importance from EAP-IG formatted data and patching results.

Loads results/eap_ig_data.json (and per-dataset patching results) and produces:
- Layer importance bar chart
- Per-sample effect heatmap
- Format comparison (if multiple datasets)
- Early vs late layer analysis

Optionally uses EAP-IG / eap_viz if installed.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try EAP-IG / eap_viz (pip install eap[viz] provides eap.viz)
EAP_VIZ_AVAILABLE = False
eap_viz = None
for mod in ['eap_viz', 'eap_ig', 'eap.viz', 'eap']:
    try:
        if mod == 'eap_viz':
            import eap_viz
        elif mod == 'eap_ig':
            import eap_ig as eap_viz
        elif mod == 'eap.viz':
            from eap import viz as eap_viz
        else:  # eap - try eap.viz submodule from pip install eap[viz]
            import eap
            eap_viz = getattr(eap, 'viz', None) or getattr(eap, 'visualization', None)
            if eap_viz is None:
                raise ImportError("eap has no viz submodule")
        EAP_VIZ_AVAILABLE = True
        break
    except (ImportError, AttributeError):
        pass


def load_eap_ig_data(path: Path) -> dict:
    """Load EAP-IG formatted JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_layer_importance_from_eap(eap_data: dict) -> dict:
    """Compute layer importance from EAP-IG format (prediction_changed, correct after patch)."""
    layer_effects = defaultdict(list)
    layer_correct = defaultdict(list)

    for item in eap_data.get('results', []):
        y_expected = item.get('y_expected', '').strip()
        original_pred = item.get('original_prediction_x', '').strip()

        for layer_name, le in item.get('layer_effects', {}).items():
            patched = le.get('patched_output', '').strip()
            changed = le.get('prediction_changed', (patched != original_pred))

            layer_effects[layer_name].append(1.0 if changed else 0.0)
            layer_correct[layer_name].append(1.0 if (patched == y_expected) else 0.0)

    all_layers = set(layer_effects.keys()) | set(layer_correct.keys())
    importance = {}
    for layer in all_layers:
        eff = np.mean(layer_effects[layer]) if layer_effects[layer] else 0.0
        corr = np.mean(layer_correct[layer]) if layer_correct[layer] else 0.0
        importance[layer] = 0.5 * eff + 0.5 * corr
    return importance


def plot_layer_importance_bar(layer_importance: dict, output_path: Path):
    """Bar chart of layer importance."""
    sorted_items = sorted(
        layer_importance.items(),
        key=lambda x: int(x[0].split('.')[-1]) if x[0].split('.')[-1].isdigit() else 0
    )
    layers = [x[0].replace('model.layers.', 'L') for x in sorted_items]
    scores = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.viridis(np.array(scores))
    ax.bar(range(len(layers)), scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Layer importance (0.5 × effect + 0.5 × correct_after_patch)')
    ax.set_xlabel('Layer')
    ax.set_title('Layer importance from activation patching (EAP-IG data)')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_effect_heatmap(eap_data: dict, output_path: Path):
    """Heatmap: samples (rows) × layers (cols), value = prediction_changed (1/0)."""
    results = eap_data.get('results', [])
    if not results:
        return

    all_layers = set()
    for item in results:
        all_layers.update(item.get('layer_effects', {}).keys())
    layers = sorted(all_layers, key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)

    matrix = []
    for item in results[:100]:  # Limit to 100 samples for readability
        row = []
        for layer in layers:
            le = item.get('layer_effects', {}).get(layer, {})
            val = 1.0 if le.get('prediction_changed', False) else 0.0
            row.append(val)
        matrix.append(row)

    if not matrix:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(matrix, xticklabels=[L.replace('model.layers.', 'L') for L in layers],
                yticklabels=[r.get('id', str(i)) for i, r in enumerate(results[:100])],
                cmap='RdYlGn', vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Prediction changed (1=yes)'})
    ax.set_xlabel('Layer')
    ax.set_ylabel('Sample')
    ax.set_title('Per-sample effect: does patching this layer change the prediction?')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_format_comparison(patching_dir: Path, output_path: Path):
    """Compare layer importance across datasets (numeric, english, etc.)."""
    datasets = ['numeric', 'english', 'spanish', 'italian']
    format_importance = {}

    for ds in datasets:
        # Try EAP-IG style in patching results
        pf = patching_dir / f'{ds}_patching_results.json'
        if not pf.exists():
            continue
        with open(pf, 'r') as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get('results', [])

        layer_effects = defaultdict(list)
        layer_correct = defaultdict(list)
        for item in results:
            y_exp = item.get('y_expected', '').strip()
            orig = item.get('original_predicted_x', '').strip()
            for ln, pr in item.get('patching_results', {}).items():
                if 'error' in pr:
                    continue
                patched = pr.get('patched_output', '').strip()
                layer_effects[ln].append(1.0 if (patched != orig) else 0.0)
                layer_correct[ln].append(1.0 if (patched == y_exp) else 0.0)

        imp = {}
        for ln in set(layer_effects.keys()) | set(layer_correct.keys()):
            e = np.mean(layer_effects[ln]) if layer_effects[ln] else 0
            c = np.mean(layer_correct[ln]) if layer_correct[ln] else 0
            imp[ln] = 0.5 * e + 0.5 * c
        format_importance[ds] = imp

    if len(format_importance) < 2:
        return

    all_layers = set()
    for imp in format_importance.values():
        all_layers.update(imp.keys())
    layers = sorted(all_layers, key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
    layer_labels = [L.replace('model.layers.', 'L') for L in layers]

    matrix = []
    for ds in format_importance:
        row = [format_importance[ds].get(L, 0) for L in layers]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(matrix, xticklabels=layer_labels, yticklabels=list(format_importance.keys()),
                cmap='viridis', vmin=0, vmax=1, ax=ax, annot=False, cbar_kws={'label': 'Importance'})
    ax.set_xlabel('Layer')
    ax.set_ylabel('Format / dataset')
    ax.set_title('Layer importance by format (numeric vs verbal)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_early_vs_late(layer_importance: dict, output_path: Path):
    """Early vs middle vs late layer importance."""
    sorted_items = sorted(
        layer_importance.items(),
        key=lambda x: int(x[0].split('.')[-1]) if x[0].split('.')[-1].isdigit() else 0
    )
    n = len(sorted_items)
    if n < 3:
        return
    early = sorted_items[:n // 3]
    mid = sorted_items[n // 3:2 * n // 3]
    late = sorted_items[2 * n // 3:]

    fig, ax = plt.subplots(figsize=(8, 4))
    regions = ['Early', 'Middle', 'Late']
    means = [np.mean([x[1] for x in early]), np.mean([x[1] for x in mid]), np.mean([x[1] for x in late])]
    bars = ax.bar(regions, means, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax.set_ylabel('Mean importance')
    ax.set_title('Early vs middle vs late layers')
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{v:.2f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def try_eap_viz(eap_data: dict, output_dir: Path):
    """Call EAP-IG / eap_viz if available."""
    if not EAP_VIZ_AVAILABLE or eap_viz is None:
        print("EAP-IG / eap_viz not installed. Using matplotlib visualizations.")
        return

    try:
        layer_imp = compute_layer_importance_from_eap(eap_data)
        if hasattr(eap_viz, 'plot_layer_importance'):
            p = output_dir / 'eap_viz_layer_importance.png'
            eap_viz.plot_layer_importance(layer_imp, save_path=str(p))
            print(f"Saved (EAP-viz): {p}")
        elif hasattr(eap_viz, 'visualize'):
            p = output_dir / 'eap_viz_output.html'
            eap_viz.visualize(layer_imp, output_path=str(p))
            print(f"Saved (EAP-viz): {p}")
    except Exception as e:
        print(f"EAP-viz failed: {e}. Using matplotlib.")


def main():
    parser = argparse.ArgumentParser(description='Visualize layer importance from EAP-IG data')
    parser.add_argument('--eap-ig-file', type=str, default='results/eap_ig_data.json',
                        help='Path to eap_ig_data.json')
    parser.add_argument('--patching-dir', type=str, default='results/patching_results',
                        help='Directory with *_patching_results.json')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Output directory for plots')
    args = parser.parse_args()

    eap_path = Path(args.eap_ig_file)
    output_dir = Path(args.output_dir)
    patching_dir = Path(args.patching_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Layer importance visualization")
    print("=" * 60)

    if not eap_path.exists():
        print(f"EAP-IG data not found: {eap_path}")
        print("Run activation patching with --format-eap-ig first.")
        return 1

    eap_data = load_eap_ig_data(eap_path)
    layer_importance = compute_layer_importance_from_eap(eap_data)

    # Try EAP-IG native viz first
    try_eap_viz(eap_data, output_dir)

    # Matplotlib plots
    plot_layer_importance_bar(layer_importance, output_dir / 'eap_layer_importance_bar.png')
    plot_effect_heatmap(eap_data, output_dir / 'eap_effect_heatmap.png')
    plot_early_vs_late(layer_importance, output_dir / 'eap_early_vs_late.png')
    if patching_dir.exists():
        plot_format_comparison(patching_dir, output_dir / 'eap_format_comparison.png')

    print("\nDone. Check output in:", output_dir.absolute())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
