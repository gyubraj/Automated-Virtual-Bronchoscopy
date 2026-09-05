#!/bin/bash
#SBATCH --job-name=wingsnet
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=wingsnet_%j.out
#SBATCH --error=wingsnet_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ENV_ACTIVATE=${AVB_ENV_ACTIVATE:-$SCRIPT_DIR/idc_env/bin/activate}
if [[ -f "$ENV_ACTIVATE" ]]; then
  source "$ENV_ACTIVATE"
fi

PYTHON_BIN=${AVB_PYTHON:-python}
"$PYTHON_BIN" -u train.py \
  --resume saved_model/wingsnet_best_checkpoint.pth \
  --fine-tune \
  --save-dir saved_model_topology \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --distal-lumen-weight 3.0 \
  --cldice-weight 0.5 \
  --cldice-iterations 8 \
  --augment \
  --device cuda
