#!/bin/bash
#SBATCH --job-name=wingsnet
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=wingsnet_%j.out
#SBATCH --error=wingsnet_%j.err

cd /home/opat90op/project/Automated-Virtual-Bronchoscopy
source idc_env/bin/activate

python train.py \
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
