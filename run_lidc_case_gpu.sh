#!/bin/bash
#SBATCH --job-name=lidc_case
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=lidc_case_%j.out
#SBATCH --error=lidc_case_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ENV_ACTIVATE=${AVB_ENV_ACTIVATE:-$SCRIPT_DIR/idc_env/bin/activate}
if [[ -f "$ENV_ACTIVATE" ]]; then
  source "$ENV_ACTIVATE"
fi

PYTHON_BIN=${AVB_PYTHON:-python}
"$PYTHON_BIN" -u run_lidc_case.py --device cuda --amp "$@"
