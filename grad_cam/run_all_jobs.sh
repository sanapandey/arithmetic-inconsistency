#!/bin/bash
# Batch submission helper for arithmetic evaluations.
# Run from the repo root: bash run_all_jobs.sh

set -euo pipefail

sbatch eval_models.sh --modelname meta-llama/Llama-3.2-1B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname meta-llama/Llama-3.2-3B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname meta-llama/Llama-3.1-8B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname Qwen/Qwen3-8B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname Qwen/Qwen3-4B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname Qwen/Qwen3-0.6B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname Qwen/Qwen3-32B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname allenai/Olmo-3-1125-32B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname allenai/Olmo-3-1025-7B --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname Qwen/Qwen3-30B-A3B-Base --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname google/gemma-3-1b-pt --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname google/gemma-2-9b --one-shot --dataset embedded_verbal
# sbatch eval_models.sh --modelname moonshotai/Kimi-Linear-48B-A3B-Instruct --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname mistralai/Mistral-7B-v0.1 --one-shot --dataset embedded_verbal
# sbatch eval_models.sh --modelname mistralai/Ministral-3-3B-Base-2512 --one-shot --dataset embedded_verbal
# sbatch eval_models.sh --modelname mistralai/Ministral-3-8B-Base-2512 --one-shot --dataset embedded_verbal
# sbatch eval_models.sh --modelname mistralai/Ministral-3-14B-Base-2512 --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname google/gemma-2-2b --one-shot --dataset embedded_verbal
sbatch eval_models.sh --modelname microsoft/phi-4 --one-shot --dataset embedded_verbal

# too slow:
# sbatch eval_models.sh --modelname google/gemma-2-27b --one-shot
