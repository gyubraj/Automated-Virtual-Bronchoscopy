#!/bin/bash
#SBATCH --job-name=airrc_patches
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=airrc_patches_%j.out
#SBATCH --error=airrc_patches_%j.err

cd /home/opat90op/project/Automated-Virtual-Bronchoscopy
source idc_env/bin/activate

python -u extract_airrc_patches.py \
  --patches-per-case 32 \
  --distal-patch-fraction 0.5 \
  --boundary-patch-fraction 0.3 \
  --distal-radius-percentile 35
