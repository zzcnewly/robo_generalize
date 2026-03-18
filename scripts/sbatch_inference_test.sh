#!/bin/bash
#SBATCH --job-name=pi05_inference_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/inference_test_%j.out
#SBATCH --error=logs/inference_test_%j.err
#SBATCH --exclude=della-l06g12,della-i14g17,della-i14g10

set -e
cd /home/zz9706/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "=== Pi05 Subtask Inference Test ==="
echo "Node: $(hostname)"
echo "Start time: $(date)"

uv run python scripts/pi05_inference_subtask_inference.py

echo "End time: $(date)"
