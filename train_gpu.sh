#!/bin/bash
#SBATCH --job-name=wingsnet
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=wingsnet_%j.out
#SBATCH --error=wingsnet_%j.err

cd /home/biqe46fe/Automated-Virtual-Bronchoscopy
python train.py