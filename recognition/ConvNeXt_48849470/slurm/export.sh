#!/bin/bash
#SBATCH --job-name=export
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/PatternAnalysis-2026/recognition/ConvNeXt_48849470
python export_preds.py "$@"