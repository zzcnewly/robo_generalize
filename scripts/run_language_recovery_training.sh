#!/bin/bash
# Script to fine-tune pi05 model to recover autoregressive language capability.
# Uses 5 hand-made examples with high/low prompt pairs.
#
# Usage:
#   # On a GPU node (e.g. after salloc):
#   bash scripts/run_language_recovery_training.sh
#
#   # Or with uv:
#   uv run scripts/train.py pi05_language_recovery

set -e

# Allow JAX to use up to 90% of GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "=== Pi05 Language Recovery Training ==="
echo "Training with 5 hand-made examples to recover language output capability."
echo ""

# Run training with the language recovery config
uv run scripts/train.py pi05_language_recovery

echo ""
echo "=== Training complete ==="
echo "Checkpoint saved to: checkpoints/pi05_language_recovery/lang_recovery_v1/"
echo ""
echo "To test language output, run:"
echo "  uv run scripts/pi05_inference_subtask_inference.py"
echo "  (update the config name to pi05_language_recovery and checkpoint path)"
