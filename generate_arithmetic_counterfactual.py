#!/usr/bin/env python3
"""
Counterfactual Arithmetic Problem Dataset Generator

Generates arithmetic problems with corrupted versions (x, x') and their corresponding
correct answers (y, y'). Uses sign switching as the corruption method.

Output: JSON files with counterfactual pairs in numeric, English, Spanish, and Italian formats
"""

import argparse
import json
import random
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import the generator and formatter classes from the original file
from generate_arithmetic_data import (
    ArithmeticProblemGenerator,
    DatasetFormatter,
    split_dataset
)


def corrupt_problem_by_sign_switching(problem: Dict) -> Dict:
    """
    Create a corrupted version of a problem by switching all signs.
    
    Args:
        problem: Original problem dictionary with 'operands' and 'operators'
    
    Returns:
        Corrupted problem dictionary with switched operators
    """
    corrupted = problem.copy()
    
    # Switch all operators: + becomes -, - becomes +
    corrupted_operators = []
    for op in problem['operators']:
        if op == '+':
            corrupted_operators.append('-')
        elif op == '-':
            corrupted_operators.append('+')
        else:
            corrupted_operators.append(op)  # Fallback (shouldn't happen)
    
    corrupted['operators'] = corrupted_operators
    
    # Calculate the result for the corrupted problem
    operands = corrupted['operands']
    result = operands[0]
    
    for i, op in enumerate(corrupted_operators):
        if op == '+':
            result += operands[i + 1]
        elif op == '-':
            result -= operands[i + 1]
    
    corrupted['result'] = result
    
    return corrupted


def generate_counterfactual_dataset(problems: List[Dict], language: str = 'numeric') -> List[Dict]:
    """
    Generate counterfactual dataset with x, x', y, y' for each problem.
    Filters out problems where y' == y (same answer for original and corrupted).
    
    Args:
        problems: List of original problem dictionaries
        language: Language format ('numeric', 'en', 'es', 'it')
    
    Returns:
        List of counterfactual entries with x, x', y, y'
    """
    formatter = DatasetFormatter()
    counterfactual_entries = []
    
    for problem in problems:
        # Create corrupted problem
        corrupted_problem = corrupt_problem_by_sign_switching(problem)
        
        # Skip if corrupted problem has the same result as original
        if problem['result'] == corrupted_problem['result']:
            continue
        
        # Format original problem (x, y) based on language
        if language == 'numeric':
            x_prompt, y_answer = formatter.format_numeric(problem)
            x_prime_prompt, y_prime_answer = formatter.format_numeric(corrupted_problem)
        else:
            x_prompt, y_answer = formatter.format_verbal(problem, language=language)
            x_prime_prompt, y_prime_answer = formatter.format_verbal(corrupted_problem, language=language)
        
        # Create counterfactual entry
        entry = {
            'id': problem['id'],
            'x': x_prompt,  # Original problem
            'x_prime': x_prime_prompt,  # Corrupted problem
            'y': y_answer,  # Correct answer to x
            'y_prime': y_prime_answer,  # Correct answer to x' (incorrect for x)
            'ground_truth': problem['result'],  # Original correct answer
            'ground_truth_prime': corrupted_problem['result'],  # Corrupted correct answer
            'split': problem['split'],
            'has_carry': problem['has_carry'],
            'n_terms': problem['n_terms'],
            'n_digits': problem['n_digits']
        }
        
        counterfactual_entries.append(entry)
    
    return counterfactual_entries


def save_counterfactual_dataset(entries: List[Dict], output_filename: str):
    """
    Save counterfactual dataset to JSON file.
    
    Args:
        entries: List of counterfactual entries
        output_filename: Output filename
    """
    # Ensure target directory exists: data/json/
    base_dir = os.path.join('data', 'json')
    os.makedirs(base_dir, exist_ok=True)
    
    output_path = os.path.join(base_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {output_path} ({len(entries)} counterfactual entries)")


def generate_problems_with_valid_counterfactuals(
    generator: ArithmeticProblemGenerator,
    n_problems: int,
    seen_problems: set = None
) -> List[Dict]:
    """
    Generate problems and filter out those where corrupted version has same answer.
    Continues generating until we have n_problems with valid counterfactuals.
    
    Args:
        generator: ArithmeticProblemGenerator instance
        n_problems: Target number of problems with valid counterfactuals
        seen_problems: Set of problem signatures already seen
    
    Returns:
        List of problem dictionaries with valid counterfactuals
    """
    if seen_problems is None:
        seen_problems = set()
    
    valid_problems = []
    max_attempts = n_problems * 20  # Allow more attempts to account for filtering
    attempts = 0
    
    while len(valid_problems) < n_problems and attempts < max_attempts:
        attempts += 1
        problem = generator.generate_problem()
        
        if problem is None:
            continue
        
        # Check uniqueness
        signature = generator._create_problem_signature(problem)
        if signature in seen_problems:
            continue
        
        # Create corrupted version and check if it has different result
        corrupted_problem = corrupt_problem_by_sign_switching(problem)
        
        # Skip if corrupted problem has the same result as original
        if problem['result'] == corrupted_problem['result']:
            continue
        
        # Valid problem - add it
        seen_problems.add(signature)
        valid_problems.append(problem)
        
        # Progress indicator
        if len(valid_problems) % 100 == 0:
            print(f"Generated {len(valid_problems)}/{n_problems} valid problems...")
    
    if len(valid_problems) < n_problems:
        print(f"Warning: Only generated {len(valid_problems)} valid problems out of {n_problems} requested")
        print(f"  (Some problems were skipped because corrupted versions had the same answer)")
    
    return valid_problems


def main():
    parser = argparse.ArgumentParser(
        description='Generate counterfactual arithmetic problem dataset with sign-switched corruptions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset size
    parser.add_argument('-n', '--num-problems', type=int, default=1000,
                        help='Number of problems to generate')
    
    # Difficulty parameters
    parser.add_argument('--terms', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('PROB_2', 'PROB_3'),
                        help='Probability distribution for 2-term and 3-term problems')
    parser.add_argument('--digits', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('PROB_2', 'PROB_3'),
                        help='Probability distribution for 2-digit and 3-digit numbers')
    parser.add_argument('--solution-digits', type=int, default=None,
                        help='Filter to keep only solutions with this many digits (None = no filter)')
    parser.add_argument('--carry-percentage', type=float, default=0.5,
                        help='Proportion of problems with carry/borrow operations')
    
    # Optional constraints
    parser.add_argument('--avoid-repeated-digits', action='store_true',
                        help='Avoid numbers with repeated digits (e.g., 121, 77, 88)')
    parser.add_argument('--avoid-clean-multiples', action='store_true',
                        help='Avoid clean multiples of 10 (e.g., 70, 80, 100)')
    parser.add_argument('--avoid-reverse-pairs', action='store_true',
                        help='Avoid reverse pairs for 2-term addition (if 34+21, skip 21+34)')
    
    # Output
    parser.add_argument('-o', '--output-prefix', type=str, default='arith_dataset',
                        help='Output filename prefix')
    
    # New argument to select specific formats
    parser.add_argument('--formats', type=str, nargs='+', default=['all'],
                        choices=['all', 'numeric', 'en', 'es', 'it'],
                        help='Specific problem formats to generate (default: all)')
    
    # Other
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate probability distributions
    if abs(sum(args.terms) - 1.0) > 1e-6:
        parser.error("--terms probabilities must sum to 1.0")
    if abs(sum(args.digits) - 1.0) > 1e-6:
        parser.error("--digits probabilities must sum to 1.0")
    
    print("=== Counterfactual Arithmetic Problem Dataset Generator ===\n")
    print(f"Generating {args.num_problems} problems with the following parameters:")
    print(f"  Terms distribution: 2-term={args.terms[0]:.1%}, 3-term={args.terms[1]:.1%}")
    print(f"  Digits distribution: 2-digit={args.digits[0]:.1%}, 3-digit={args.digits[1]:.1%}")
    print(f"  Solution digits filter: {args.solution_digits if args.solution_digits else 'None'}")
    print(f"  Carry/borrow percentage: {args.carry_percentage:.1%}")
    print(f"  Avoid repeated digits: {args.avoid_repeated_digits}")
    print(f"  Avoid clean multiples: {args.avoid_clean_multiples}")
    print(f"  Avoid reverse pairs: {args.avoid_reverse_pairs}")
    print(f"  Corruption method: Sign switching (all signs)")
    print(f"  Filter: Skipping problems where y' == y")
    print(f"  Target formats: {', '.join(args.formats)}")
    print(f"  Random seed: {args.seed if args.seed else 'None'}")
    print()
    
    # Initialize generator
    generator = ArithmeticProblemGenerator(
        terms_distribution=tuple(args.terms),
        digits_distribution=tuple(args.digits),
        solution_digits=args.solution_digits,
        carry_percentage=args.carry_percentage,
        avoid_repeated_digits=args.avoid_repeated_digits,
        avoid_clean_multiples=args.avoid_clean_multiples,
        avoid_reverse_pairs=args.avoid_reverse_pairs,
        seed=args.seed
    )
    
    # Generate problems with valid counterfactuals (filtering out y' == y)
    print("Generating problems with valid counterfactuals...")
    problems = generate_problems_with_valid_counterfactuals(generator, args.num_problems)
    
    if len(problems) == 0:
        print("Error: No problems generated. Try relaxing constraints.")
        return
    
    # Split into train/val/test
    print("\nSplitting into train/val/test...")
    problems = split_dataset(problems)
    
    # Assign unique IDs
    for i, problem in enumerate(problems):
        problem['id'] = f"prob_{i:06d}"
    
    # Determine which formats to generate
    target_formats = []
    if 'all' in args.formats:
        target_formats = ['numeric', 'en', 'es', 'it']
    else:
        target_formats = args.formats
    
    # Generate and save counterfactual datasets for each format
    print("\nGenerating counterfactual pairs (x, x', y, y')...")
    
    # Language names for filenames
    format_names = {
        'numeric': 'numeric',
        'en': 'english',
        'es': 'spanish',
        'it': 'italian'
    }
    
    # Store numeric entries for statistics
    numeric_entries = None
    
    for format_key in target_formats:
        # Generate counterfactual entries for this format
        counterfactual_entries = generate_counterfactual_dataset(problems, language=format_key)
        
        # Save dataset
        format_name = format_names[format_key]
        output_filename = f"{args.output_prefix}_{format_name}_counterfactual.json"
        save_counterfactual_dataset(counterfactual_entries, output_filename)
        
        # Store numeric entries for statistics
        if format_key == 'numeric':
            numeric_entries = counterfactual_entries
    
    # Print statistics (using numeric format as reference)
    print("\n=== Dataset Statistics ===")
    if numeric_entries is None:
        # Generate numeric entries if not already generated
        numeric_entries = generate_counterfactual_dataset(problems, language='numeric')
    print(f"Total counterfactual entries: {len(numeric_entries)}")
    
    # Count by split
    split_counts = defaultdict(int)
    for entry in numeric_entries:
        split_counts[entry['split']] += 1
    print(f"Train: {split_counts['train']} ({split_counts['train']/len(numeric_entries)*100:.1f}%)")
    print(f"Val: {split_counts['val']} ({split_counts['val']/len(numeric_entries)*100:.1f}%)")
    print(f"Test: {split_counts['test']} ({split_counts['test']/len(numeric_entries)*100:.1f}%)")
    
    # Count by characteristics
    terms_counts = defaultdict(int)
    digits_counts = defaultdict(int)
    carry_counts = defaultdict(int)
    
    for entry in numeric_entries:
        terms_counts[entry['n_terms']] += 1
        digits_counts[entry['n_digits']] += 1
        carry_counts[entry['has_carry']] += 1
    
    print(f"\n2-term problems: {terms_counts[2]} ({terms_counts[2]/len(numeric_entries)*100:.1f}%)")
    print(f"3-term problems: {terms_counts[3]} ({terms_counts[3]/len(numeric_entries)*100:.1f}%)")
    print(f"2-digit problems: {digits_counts[2]} ({digits_counts[2]/len(numeric_entries)*100:.1f}%)")
    print(f"3-digit problems: {digits_counts[3]} ({digits_counts[3]/len(numeric_entries)*100:.1f}%)")
    print(f"Problems with carry/borrow: {carry_counts[True]} ({carry_counts[True]/len(numeric_entries)*100:.1f}%)")
    print(f"Problems without carry/borrow: {carry_counts[False]} ({carry_counts[False]/len(numeric_entries)*100:.1f}%)")
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()

