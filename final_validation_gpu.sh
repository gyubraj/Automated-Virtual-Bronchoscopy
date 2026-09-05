#!/bin/bash
#SBATCH --job-name=airway_final_val
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=final_validation_%j.out
#SBATCH --error=final_validation_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ENV_ACTIVATE=${AVB_ENV_ACTIVATE:-$SCRIPT_DIR/idc_env/bin/activate}
if [[ -f "$ENV_ACTIVATE" ]]; then
  source "$ENV_ACTIVATE"
fi

DATA_ROOT=${AVB_DATA_ROOT:-$HOME/AMS_Project/datasets_new}
PYTHON_BIN=${AVB_PYTHON:-python}
"$PYTHON_BIN" -u run_final_validation_suite.py \
  --data-root "$DATA_ROOT" \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --device cuda \
  --limit 51 \
  --mask-threshold 0.2 \
  --skeleton-threshold 0.2 \
  --skeleton-low-threshold 0.08 \
  --prune-length 12 \
  --preserve-generations 8 \
  --output-root "$DATA_ROOT/final_validation/airrc"
