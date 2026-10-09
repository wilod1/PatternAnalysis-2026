#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/PatternAnalysis-2026/recognition/ConvNeXt_48849470
python train.py "$@"