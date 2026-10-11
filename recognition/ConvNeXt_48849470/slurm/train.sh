
#!/bin/bash
#SBATCH --job-name=convnext
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -e

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/ConvNeXt_48849470"

echo "Starting full ConvNeXt training..."
python -u train.py --epochs 100 --patience 20 "$@"

echo "Job finished at $(date)"
