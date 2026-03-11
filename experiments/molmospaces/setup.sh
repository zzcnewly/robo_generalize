#!/bin/bash
#
# One-command setup for MolmoSpaces evaluation with DreamZero.
#
# Usage:
#   bash experiments/molmospaces/setup.sh
#
# What this does:
#   1. Clones MolmoSpaces repo (if not already present)
#   2. Creates conda env 'mlspaces' with Python 3.10
#   3. Installs molmospaces and its dependencies (via conda run)
#   4. Sets up assets directory (asset download must be done on a GPU node)
#
# Prerequisites:
#   - conda (module load anaconda3/2024.10 on Princeton cluster)
#   - git
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOLMOSPACES_DIR="${REPO_ROOT}/external/molmospaces"
CONDA_ENV_NAME="mlspaces"
ASSETS_DIR="${REPO_ROOT}/external/molmospaces_assets"

echo "============================================"
echo "MolmoSpaces Setup for DreamZero Evaluation"
echo "============================================"
echo ""
echo "Repo root:       $REPO_ROOT"
echo "MolmoSpaces dir: $MOLMOSPACES_DIR"
echo "Assets dir:      $ASSETS_DIR"
echo "Conda env:       $CONDA_ENV_NAME"
echo ""

# Step 1: Clone MolmoSpaces if not present
echo "[Step 1/4] Checking MolmoSpaces repo..."
if [ -d "$MOLMOSPACES_DIR/.git" ]; then
    echo "  MolmoSpaces already exists at $MOLMOSPACES_DIR"
else
    echo "  Cloning MolmoSpaces..."
    mkdir -p "$(dirname "$MOLMOSPACES_DIR")"
    git clone https://github.com/allenai/molmospaces.git "$MOLMOSPACES_DIR"
    if [ $? -ne 0 ]; then
        echo "  ERROR: git clone failed."
        exit 1
    fi
    echo "  Done."
fi

# Step 2: Create conda environment + install
echo ""
echo "[Step 2/4] Setting up conda environment '$CONDA_ENV_NAME'..."
if command -v module &> /dev/null; then
    module load anaconda3/2024.10 2>/dev/null
fi

if conda env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Conda env '$CONDA_ENV_NAME' already exists. Skipping creation."
else
    echo "  Creating conda env '$CONDA_ENV_NAME' with Python 3.10..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.10
    if [ $? -ne 0 ]; then
        echo "  ERROR: conda create failed."
        exit 1
    fi
    echo "  Done."
fi

# Step 3: Install molmospaces into the env using conda run
echo ""
echo "[Step 3/4] Installing MolmoSpaces into '$CONDA_ENV_NAME' env..."
echo "  Running: pip install -e $MOLMOSPACES_DIR"
conda run -n "$CONDA_ENV_NAME" pip install -e "$MOLMOSPACES_DIR"
if [ $? -ne 0 ]; then
    echo "  ERROR: pip install molmospaces failed."
    exit 1
fi
echo "  Installing openpi-client..."
conda run -n "$CONDA_ENV_NAME" pip install openpi-client
if [ $? -ne 0 ]; then
    echo "  WARNING: pip install openpi-client failed (may not be critical)."
fi
# openpi-client pins numpy<2 but MolmoSpaces needs numpy>=2.2.
# The openpi_client code works fine with numpy 2.x (only uses msgpack_numpy).
echo "  Fixing numpy version (openpi-client downgrades it, but MolmoSpaces needs >=2.2)..."
conda run -n "$CONDA_ENV_NAME" pip install "numpy>=2.2.0,<3"
echo "  Done."

# Step 4: Set up assets directory and create version manifest
echo ""
echo "[Step 4/4] Setting up assets directory..."
mkdir -p "$ASSETS_DIR"

# Create version manifest so that MolmoSpaces doesn't try to auto-download
# assets at import time (which fails on login nodes without good connectivity).
# The actual asset download must be done separately on a GPU compute node.
MANIFEST_FILE="$ASSETS_DIR/mlspaces_installed_data_type_to_source_to_versions.json"
if [ ! -f "$MANIFEST_FILE" ]; then
    conda run -n "$CONDA_ENV_NAME" python -c "
import json, sys
sys.path.insert(0, '$MOLMOSPACES_DIR')
from molmo_spaces.molmo_spaces_constants import DATA_TYPE_TO_SOURCE_TO_VERSION
with open('$MANIFEST_FILE', 'w') as f:
    json.dump(DATA_TYPE_TO_SOURCE_TO_VERSION, f, indent=2)
print('  Version manifest created.')
"
fi

echo "  Assets directory: $ASSETS_DIR"
echo ""
echo "  IMPORTANT: Assets must be downloaded on a GPU compute node."
echo "  Get a GPU node first, then run:"
echo ""
echo "    salloc --gres=gpu:1 --mem=64G -t 02:00:00 --account=henderson"
echo "    module load anaconda3/2024.10"
echo "    export MLSPACES_ASSETS_DIR=$ASSETS_DIR"
echo "    export MLSPACES_FORCE_INSTALL=True"
echo "    conda run -n $CONDA_ENV_NAME python -c \\"
echo "      'from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()'"
echo ""
echo "  This downloads scene, robot, and object assets (~10GB)."
echo ""

echo "============================================"
echo "Setup complete!"
echo ""
echo "Quick start:"
echo ""
echo "  # Terminal 1: Start DreamZero server (in dreamzero env)"
echo "  conda activate dreamzero"
echo "  cd $REPO_ROOT"
echo "  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \\"
echo "      socket_test_optimized_AR.py --port 8000 --enable-dit-cache \\"
echo "      --model-path <path/to/checkpoint>"
echo ""
echo "  # Terminal 2: Run MolmoSpaces evaluation (in mlspaces env)"
echo "  conda activate $CONDA_ENV_NAME"
echo "  cd $REPO_ROOT"
echo "  bash experiments/molmospaces/run_eval.sh --port 8000"
echo "============================================"
