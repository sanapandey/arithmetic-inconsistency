#!/usr/bin/env python3
"""
Activation Patching for Counterfactual Arithmetic Analysis

This script performs activation patching experiments to identify which layers/components
are causally involved in arithmetic reasoning. Results are formatted for EAP-IG visualization.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_counterfactual_results(results_path: str) -> List[Dict]:
    """Load counterfactual evaluation results"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def patch_activations(
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    layer_name: str,
    patch_position: Optional[int] = None,
    max_new_tokens: int = 8
) -> Tuple[str, Dict]:
    """
    Patch activations from source into target at specified layer
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        source_prompt: Source prompt (x')
        target_prompt: Target prompt (x)
        layer_name: Name of layer to patch
        patch_position: Position to patch (None = all positions)
        max_new_tokens: Max tokens to generate
    
    Returns:
        Tuple of (generated_text, patched_activations_dict)
    """
    # Tokenize both prompts
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    
    source_len = source_inputs['input_ids'].shape[1]
    target_len = target_inputs['input_ids'].shape[1]
    
    # Store source activations
    source_activations = {}
    
    def source_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                source_activations[name] = output[0].detach()
            else:
                source_activations[name] = output.detach()
        return hook
    
    # Store target activations and patching hook
    target_activations = {}
    patch_applied = False
    
    def target_hook(name):
        def hook(module, input, output):
            nonlocal patch_applied
            if name == layer_name and not patch_applied:
                # Patch: replace target activation with source activation
                if isinstance(output, tuple):
                    output_tensor = output[0]
                    if name in source_activations:
                        source_act = source_activations[name]
                        if patch_position is None:
                            # Patch all positions (truncate/pad as needed)
                            min_len = min(output_tensor.shape[1], source_act.shape[1])
                            output_tensor[:, :min_len, :] = source_act[:, :min_len, :].to(output_tensor.device)
                        else:
                            # Patch specific position
                            if patch_position < output_tensor.shape[1] and patch_position < source_act.shape[1]:
                                output_tensor[:, patch_position, :] = source_act[:, patch_position, :].to(output_tensor.device)
                        patch_applied = True
                    target_activations[name] = output_tensor.detach()
                    return (output_tensor,) + output[1:]
                else:
                    if isinstance(output, tuple):
                        target_activations[name] = output[0].detach()
                    else:
                        target_activations[name] = output.detach()
            else:
                if isinstance(output, tuple):
                    target_activations[name] = output[0].detach()
                else:
                    target_activations[name] = output.detach()
            return output
        return hook
    
    # Register hooks
    source_hooks = []
    target_hooks = []
    
    for name, module in model.named_modules():
        if layer_name in name or name == layer_name:
            source_hook_obj = module.register_forward_hook(source_hook(name))
            source_hooks.append((name, source_hook_obj))
            target_hook_obj = module.register_forward_hook(target_hook(name))
            target_hooks.append((name, target_hook_obj))
    
    # Forward pass on source to collect activations
    with torch.no_grad():
        _ = model(**source_inputs)
    
    # Forward pass on target with patching (patch_applied stays False so the hook can apply the patch)
    with torch.no_grad():
        outputs = model.generate(
            **target_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode generated text
    generated = tokenizer.decode(outputs[0][target_len:], skip_special_tokens=True)
    
    # Clean up hooks
    for name, hook in source_hooks:
        hook.remove()
    for name, hook in target_hooks:
        hook.remove()
    
    # Convert activations to CPU and prepare for serialization
    patched_info = {
        'layer_name': layer_name,
        'patch_position': patch_position,
        'source_len': source_len,
        'target_len': target_len,
    }
    
    return generated, patched_info


def run_patching_experiment(
    model,
    tokenizer,
    results: List[Dict],
    layer_names: List[str],
    max_samples: Optional[int] = None,
    max_new_tokens: int = 8
) -> List[Dict]:
    """
    Run activation patching experiment across multiple layers
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        results: List of counterfactual evaluation results
        layer_names: List of layer names to patch
        max_samples: Maximum number of samples to process
        max_new_tokens: Max tokens for generation
    
    Returns:
        List of patching results
    """
    if max_samples:
        results = results[:max_samples]
    
    patching_results = []
    
    for item in tqdm(results, desc="Running patching experiments"):
        x_prompt = item['x']
        x_prime_prompt = item['x_prime']
        y_expected = item['y_expected']
        y_prime_expected = item['y_prime_expected']
        
        item_results = {
            'id': item['id'],
            'x': x_prompt,
            'x_prime': x_prime_prompt,
            'y_expected': y_expected,
            'y_prime_expected': y_prime_expected,
            'original_predicted_x': item.get('predicted_x', ''),
            'original_predicted_x_prime': item.get('predicted_x_prime', ''),
            'patching_results': {},
        }
        
        # Patch at each layer
        for layer_name in layer_names:
            try:
                # Patch x' activations into x
                patched_output, patching_info = patch_activations(
                    model, tokenizer,
                    source_prompt=x_prime_prompt,
                    target_prompt=x_prompt,
                    layer_name=layer_name,
                    max_new_tokens=max_new_tokens
                )
                
                item_results['patching_results'][layer_name] = {
                    'patched_output': patched_output,
                    'patching_info': patching_info,
                }
            except Exception as e:
                print(f"Error patching layer {layer_name} for item {item['id']}: {e}")
                item_results['patching_results'][layer_name] = {
                    'error': str(e)
                }
        
        patching_results.append(item_results)
    
    return patching_results


def get_layer_names(model) -> List[str]:
    """Extract layer names from model architecture"""
    layer_names = []
    
    for name, module in model.named_modules():
        # Common patterns for transformer layers
        if any(keyword in name.lower() for keyword in ['layer', 'block', 'transformer', 'h.']):
            if '.' in name:  # Only get actual layers, not submodules
                layer_names.append(name)
    
    # Filter to get unique layer groups (e.g., "model.layers.0" not "model.layers.0.self_attn")
    unique_layers = []
    seen_prefixes = set()
    
    for name in sorted(layer_names):
        # Extract layer prefix (e.g., "model.layers.0" from "model.layers.0.self_attn")
        parts = name.split('.')
        if len(parts) >= 3:
            prefix = '.'.join(parts[:3])  # e.g., "model.layers.0"
            if prefix not in seen_prefixes:
                unique_layers.append(prefix)
                seen_prefixes.add(prefix)
    
    return unique_layers if unique_layers else layer_names[:20]  # Fallback to first 20


def format_for_eap_ig(patching_results: List[Dict], output_path: str):
    """
    Format patching results for EAP-IG visualization
    
    EAP-IG typically expects:
    - Layer-wise patching effects
    - Original vs patched predictions
    - Causal attribution scores
    """
    eap_data = {
        'experiment_type': 'activation_patching',
        'num_samples': len(patching_results),
        'results': [],
    }
    
    for item in patching_results:
        eap_item = {
            'id': item['id'],
            'x': item['x'],
            'x_prime': item['x_prime'],
            'y_expected': item['y_expected'],
            'y_prime_expected': item['y_prime_expected'],
            'original_prediction_x': item.get('original_predicted_x', ''),
            'original_prediction_x_prime': item.get('original_predicted_x_prime', ''),
            'layer_effects': {},
        }
        
        # Calculate effect of patching at each layer
        for layer_name, patching_result in item['patching_results'].items():
            if 'error' in patching_result:
                continue
            
            patched_output = patching_result['patched_output']
            
            # Simple effect metric: does patching change the prediction?
            original_pred = item.get('original_predicted_x', '')
            prediction_changed = (patched_output.strip() != original_pred.strip())
            
            eap_item['layer_effects'][layer_name] = {
                'patched_output': patched_output,
                'prediction_changed': prediction_changed,
                'layer_index': layer_name.split('.')[-1] if '.' in layer_name else layer_name,
            }
        
        eap_data['results'].append(eap_item)
    
    # Save formatted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eap_data, f, ensure_ascii=False, indent=2)
    
    print(f"EAP-IG formatted data saved to {output_path}")
    return eap_data


def main():
    parser = argparse.ArgumentParser(
        description='Run activation patching experiments on counterfactual datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to counterfactual evaluation results JSON file')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Specific layer names to patch (default: auto-detect)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                        help='Max new tokens for generation')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for patching results')
    parser.add_argument('--format-eap-ig', action='store_true',
                        help='Format results for EAP-IG visualization')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype for loading')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Activation Patching Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Results file: {args.results_file}")
    print()
    
    # Load model
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
    
    print("Model loaded successfully")
    print()
    
    # Get layer names
    if args.layers:
        layer_names = args.layers
    else:
        print("Auto-detecting layer names...")
        layer_names = get_layer_names(model)
        print(f"Found {len(layer_names)} layers: {layer_names[:5]}... (showing first 5)")
    
    print()
    
    # Load results
    print("Loading counterfactual results...")
    results = load_counterfactual_results(args.results_file)
    print(f"Loaded {len(results)} results")
    
    if args.max_samples:
        results = results[:args.max_samples]
        print(f"Processing {len(results)} samples (limited)")
    
    print()
    
    # Run patching experiments
    print("Running activation patching experiments...")
    patching_results = run_patching_experiment(
        model, tokenizer, results, layer_names,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'patching_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(patching_results, f, ensure_ascii=False, indent=2)
    
    print(f"Patching results saved to {results_file}")
    
    # Format for EAP-IG if requested
    if args.format_eap_ig:
        print("\nFormatting for EAP-IG visualization...")
        eap_file = output_dir / 'eap_ig_data.json'
        format_for_eap_ig(patching_results, str(eap_file))
    
    print("\n" + "=" * 60)
    print("Activation patching complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    if args.format_eap_ig:
        print(f"EAP-IG data: {output_dir / 'eap_ig_data.json'}")
    print("\nNext steps:")
    print("1. Review patching_results.json to see layer-wise effects")
    print("2. Load eap_ig_data.json into EAP-IG visualization tool")
    print("3. Analyze which layers are causally important for arithmetic reasoning")


if __name__ == '__main__':
    main()

