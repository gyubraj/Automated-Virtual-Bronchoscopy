#!/bin/bash
#SBATCH --job-name=airway_ablation
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=ablation_validation_%j.out
#SBATCH --error=ablation_validation_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ENV_ACTIVATE=${AVB_ENV_ACTIVATE:-$SCRIPT_DIR/idc_env/bin/activate}
if [[ -f "$ENV_ACTIVATE" ]]; then
  source "$ENV_ACTIVATE"
fi

DATA_ROOT=${AVB_DATA_ROOT:-$HOME/AMS_Project/datasets_new}
PYTHON_BIN=${AVB_PYTHON:-python}
"$PYTHON_BIN" -u run_validation_ablation.py \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --limit 10 \
  --output-root "$DATA_ROOT/final_validation/ablations"

"$PYTHON_BIN" -u summarize_validation_tables.py \
  "$DATA_ROOT/final_validation/ablations" \
  --output-md "$DATA_ROOT/final_validation/ablations/ablation_summary_table.md" \
  --output-json "$DATA_ROOT/final_validation/ablations/ablation_summary_table.json"
