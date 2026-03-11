#!/usr/bin/env python3
"""
Evaluate models on counterfactual arithmetic datasets and collect activations for EAP-IG

This script:
1. Evaluates a model on counterfactual pairs (x, x')
2. Collects activations at each layer
3. Saves results in format compatible with activation patching and EAP-IG visualization
"""

import argparse
import json
import os
import re
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional


# Counterfactual dataset configurations
COUNTERFACTUAL_DATASETS = {
    'numeric': 'data/json/arith_dataset_numeric_counterfactual.json',
    'english': 'data/json/arith_dataset_english_counterfactual.json',
    'spanish': 'data/json/arith_dataset_spanish_counterfactual.json',
    'italian': 'data/json/arith_dataset_italian_counterfactual.json',
}


def load_counterfactual_dataset(dataset_path: str, split: str = 'test', max_samples: Optional[int] = None) -> List[Dict]:
    """Load counterfactual dataset and filter by split"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered = [item for item in data if item['split'] == split]
    
    if max_samples:
        filtered = filtered[:max_samples]
    
    return filtered


def normalize_answer(answer: str, dataset_type: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.strip()
    
    if dataset_type == 'numeric':
        # Extract just the number
        match = re.search(r'-?\d+', answer)
        if match:
            return match.group()
        return answer
    else:
        # For verbal, lowercase and clean
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s-]', '', answer)
        answer = ' '.join(answer.split())
        return answer


def extract_answer(text: str, dataset_type: str) -> str:
    """Extract the answer from model output"""
    text = text.strip()
    
    if dataset_type == 'numeric':
        match = re.search(r'-?\d+', text)
        if match:
            return match.group()
        return text
    else:
        text = text.split('\n')[0]
        text = re.split(r'[.!?]', text)[0]
        return text.strip()


def check_answer(predicted: str, expected: str, dataset_type: str) -> bool:
    """Check if predicted answer matches expected"""
    pred_norm = normalize_answer(predicted, dataset_type)
    exp_norm = normalize_answer(expected, dataset_type)
    
    if dataset_type == 'numeric':
        return pred_norm == exp_norm
    else:
        return pred_norm == exp_norm or pred_norm.startswith(exp_norm)


class ActivationCollector:
    """Hook to collect activations at each layer"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to collect activations"""
        self.activations = {}
        
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle tuple outputs)
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        # Collect all transformer layers
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower() or 'transformer' in name.lower():
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(make_hook(name))
                    self.hooks.append((name, hook))
        
        print(f"Registered {len(self.hooks)} activation hooks")
    
    def clear_hooks(self):
        """Remove all hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get collected activations"""
        return self.activations.copy()
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations = {}


def evaluate_counterfactual_pair(
    model,
    tokenizer,
    item: Dict,
    dataset_type: str,
    collector: Optional[ActivationCollector] = None,
    max_new_tokens: int = 8,
    collect_activations: bool = False
) -> Dict:
    """
    Evaluate a single counterfactual pair (x, x') and collect activations
    
    Returns:
        Dictionary with predictions, correctness, and activations (if collected)
    """
    results = {}
    
    # Process original (x, y)
    x_prompt = item['x']
    y_expected = item['y']
    
    device = next(model.parameters()).device
    inputs_x = tokenizer(x_prompt, return_tensors="pt").to(device)
    
    if collect_activations and collector:
        collector.clear_activations()
    
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_attentions": False,
            "output_hidden_states": collect_activations,
        }
        outputs_x = model.generate(**inputs_x, **gen_kwargs)
    
    input_len_x = inputs_x['input_ids'].shape[1]
    generated_x = tokenizer.decode(outputs_x[0][input_len_x:], skip_special_tokens=True)
    predicted_x = extract_answer(generated_x, dataset_type)
    correct_x = check_answer(predicted_x, y_expected, dataset_type)
    
    activations_x = None
    if collect_activations and collector:
        activations_x = collector.get_activations()
        # Convert to lists for JSON serialization
        activations_x = {k: v.tolist() for k, v in activations_x.items()}
    
    # Process corrupted (x', y')
    x_prime_prompt = item['x_prime']
    y_prime_expected = item['y_prime']
    
    # Ensure inputs are on the right device
    device = next(model.parameters()).device
    inputs_x_prime = tokenizer(x_prime_prompt, return_tensors="pt").to(device)
    
    if collect_activations and collector:
        collector.clear_activations()
    
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "output_attentions": False,
            "output_hidden_states": collect_activations,
        }
        outputs_x_prime = model.generate(**inputs_x_prime, **gen_kwargs)
    
    input_len_x_prime = inputs_x_prime['input_ids'].shape[1]
    generated_x_prime = tokenizer.decode(outputs_x_prime[0][input_len_x_prime:], skip_special_tokens=True)
    predicted_x_prime = extract_answer(generated_x_prime, dataset_type)
    correct_x_prime = check_answer(predicted_x_prime, y_prime_expected, dataset_type)
    
    activations_x_prime = None
    if collect_activations and collector:
        activations_x_prime = collector.get_activations()
        # Convert to lists for JSON serialization
        activations_x_prime = {k: v.tolist() for k, v in activations_x_prime.items()}
    
    results = {
        'id': item['id'],
        'x': x_prompt,
        'x_prime': x_prime_prompt,
        'y_expected': y_expected.strip(),
        'y_prime_expected': y_prime_expected.strip(),
        'predicted_x': predicted_x,
        'predicted_x_prime': predicted_x_prime,
        'correct_x': correct_x,
        'correct_x_prime': correct_x_prime,
        'ground_truth': item['ground_truth'],
        'ground_truth_prime': item['ground_truth_prime'],
    }
    
    if collect_activations:
        results['activations_x'] = activations_x
        results['activations_x_prime'] = activations_x_prime
    
    return results


def evaluate_counterfactual_dataset(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_type: str,
    collector: Optional[ActivationCollector] = None,
    max_new_tokens: int = 8,
    collect_activations: bool = False,
    log_every: int = 100
) -> Dict:
    """
    Evaluate model on counterfactual dataset
    
    Returns:
        Dictionary with accuracy metrics and detailed results
    """
    results = []
    correct_x = 0
    correct_x_prime = 0
    total = len(dataset)
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_type}")):
        result = evaluate_counterfactual_pair(
            model, tokenizer, item, dataset_type,
            collector=collector,
            max_new_tokens=max_new_tokens,
            collect_activations=collect_activations
        )
        
        if result['correct_x']:
            correct_x += 1
        if result['correct_x_prime']:
            correct_x_prime += 1
        
        results.append(result)
        
        # Periodic logging
        if (idx + 1) % log_every == 0:
            acc_x = correct_x / (idx + 1)
            acc_x_prime = correct_x_prime / (idx + 1)
            print(f"[progress] {dataset_type}: {idx + 1}/{total} processed, "
                  f"acc_x={acc_x*100:.2f}%, acc_x_prime={acc_x_prime*100:.2f}%", flush=True)
    
    accuracy_x = correct_x / total if total > 0 else 0
    accuracy_x_prime = correct_x_prime / total if total > 0 else 0
    
    return {
        'accuracy_x': accuracy_x,
        'accuracy_x_prime': accuracy_x_prime,
        'correct_x': correct_x,
        'correct_x_prime': correct_x_prime,
        'total': total,
        'results': results,
    }


def save_results(results: Dict, model_name: str, dataset_type: str, output_dir: str = 'results'):
    """Save evaluation results"""
    model_safe = model_name.replace('/', '__')
    result_dir = Path(output_dir) / model_safe / 'counterfactual'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    result_file = result_dir / f'{dataset_type}_counterfactual_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        'model': model_name,
        'dataset': dataset_type,
        'mode': 'counterfactual',
        'accuracy_x': results['accuracy_x'],
        'accuracy_x_prime': results['accuracy_x_prime'],
        'correct_x': results['correct_x'],
        'correct_x_prime': results['correct_x_prime'],
        'total': results['total'],
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_file = result_dir / f'{dataset_type}_counterfactual_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {result_dir}")
    return result_dir


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models on counterfactual arithmetic datasets with activation collection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'numeric', 'english', 'spanish', 'italian'],
                        help='Counterfactual dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate (for quick testing)')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                        help='Max new tokens for generation')
    parser.add_argument('--collect-activations', action='store_true',
                        help='Collect activations at each layer (for activation patching)')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Log interim accuracy every N samples')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Model dtype for loading')
    
    args = parser.parse_args()
    
    # Determine datasets to evaluate
    if args.dataset == 'all':
        datasets_to_eval = list(COUNTERFACTUAL_DATASETS.keys())
    else:
        datasets_to_eval = [args.dataset]
    
    print("=" * 60)
    print("Counterfactual Arithmetic Dataset Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Datasets: {', '.join(datasets_to_eval)}")
    print(f"Split: {args.split}")
    print(f"Collect activations: {args.collect_activations}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()
    
    # Load model and tokenizer
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
    
    print(f"Model loaded successfully")
    print()
    
    # Set up activation collector if needed
    collector = None
    if args.collect_activations:
        collector = ActivationCollector(model)
        collector.register_hooks()
    
    # Evaluate each dataset
    all_summaries = []
    
    for dataset_type in datasets_to_eval:
        print("-" * 40)
        print(f"Evaluating: {dataset_type}")
        print("-" * 40)
        
        # Load dataset
        dataset_path = COUNTERFACTUAL_DATASETS[dataset_type]
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found at {dataset_path}, skipping...")
            continue
        
        dataset = load_counterfactual_dataset(dataset_path, args.split, args.max_samples)
        print(f"Loaded {len(dataset)} samples from {args.split} split")
        
        # Evaluate
        results = evaluate_counterfactual_dataset(
            model,
            tokenizer,
            dataset,
            dataset_type,
            collector=collector,
            max_new_tokens=args.max_new_tokens,
            collect_activations=args.collect_activations,
            log_every=args.log_every,
        )
        
        # Save results
        save_results(results, args.model, dataset_type, args.output_dir)
        
        acc_x_pct = results['accuracy_x'] * 100
        acc_x_prime_pct = results['accuracy_x_prime'] * 100
        print(f"Accuracy on x: {acc_x_pct:.2f}% ({results['correct_x']}/{results['total']})")
        print(f"Accuracy on x': {acc_x_prime_pct:.2f}% ({results['correct_x_prime']}/{results['total']})")
        print()
        
        all_summaries.append({
            'dataset': dataset_type,
            'accuracy_x': results['accuracy_x'],
            'accuracy_x_prime': results['accuracy_x_prime'],
            'correct_x': results['correct_x'],
            'correct_x_prime': results['correct_x_prime'],
            'total': results['total'],
        })
    
    # Clean up hooks
    if collector:
        collector.clear_hooks()
    
    # Print final summary
    print("=" * 60)
    print("Final Summary")
    print("=" * 60)
    for s in all_summaries:
        print(f"  {s['dataset']:12s}: x={s['accuracy_x']*100:6.2f}%, x'={s['accuracy_x_prime']*100:6.2f}% "
              f"({s['correct_x']}/{s['total']}, {s['correct_x_prime']}/{s['total']})")
    
    print()
    print("=" * 60)
    print("Next steps for activation patching:")
    print("1. Review results in results/<model>/counterfactual/")
    print("2. Run activation_patching.py to perform patching experiments")
    print("3. Use EAP-IG visualization tools to analyze circuits")
    print("=" * 60)


if __name__ == '__main__':
    main()

