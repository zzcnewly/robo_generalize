#!/bin/bash
#SBATCH --job-name=mlspaces-download
#SBATCH --account=henderson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=logs/mlspaces_download_%j.out
#SBATCH --error=logs/mlspaces_download_%j.err

REPO_ROOT="/scratch/gpfs/HENDERSON/zs7353/robo_generalize"
MOLMOSPACES_DIR="${REPO_ROOT}/external/molmospaces"
CACHE_DIR="${REPO_ROOT}/external/molmospaces_cache"

cd "$REPO_ROOT"
mkdir -p logs

module purge
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate mlspaces

echo "========================================"
echo "MolmoSpaces Asset Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Cache dir: $CACHE_DIR"
echo "========================================"

# Download all mujoco assets using HuggingFace downloader (--versioned for direct cache use)
python "${MOLMOSPACES_DIR}/scripts/hf_download.py" "$CACHE_DIR" --source mujoco --versioned

echo ""
echo "[$(date)] Download complete."
echo ""

# Point MolmoSpaces cache to our download
export MLSPACES_CACHE_DIR="${CACHE_DIR}/mujoco"
export MLSPACES_ASSETS_DIR="${REPO_ROOT}/external/molmospaces_assets"

echo "To use these assets, set:"
echo "  export MLSPACES_CACHE_DIR=${CACHE_DIR}/mujoco"
echo "  export MLSPACES_ASSETS_DIR=${REPO_ROOT}/external/molmospaces_assets"
echo ""
echo "Done."
