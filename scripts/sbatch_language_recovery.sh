#!/bin/bash
#SBATCH --job-name=pi05_lang_recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/lang_recovery_%j.out
#SBATCH --error=logs/lang_recovery_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --exclude=della-l06g12,della-i14g17

# Pi05 Language Recovery Training
# Fine-tune pi05 base model with 5 hand-made examples to recover
# autoregressive language capability from PaliGemma backbone.

set -e

# Create log directory if it doesn't exist
mkdir -p logs

echo "=== Pi05 Language Recovery Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd /home/zz9706/openpi

# Allow JAX to use up to 90% of GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Run training
uv run scripts/train.py pi05_language_recovery

echo ""
echo "=== Training complete ==="
echo "End time: $(date)"
echo "Checkpoint: checkpoints/pi05_language_recovery/lang_recovery_v1/"
