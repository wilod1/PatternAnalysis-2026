#!/bin/bash
#SBATCH --job-name=final
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# One job for everything the report still needs: predictions for the six
# provided-split models, the benchmark / calibration / reject-option / failure
# analysis, and the resource profile. Submit from the repo folder:
#     sbatch slurm/final.sh

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/ConvNeXt_48849470"

echo "== predictions"
for r in prov_rc prov_mix prov_tiny_a prov_mini_a prov_cnn prov_resnet; do
    for s in val test; do
        echo "-- $r $s"
        python -u per_patient.py "results/checkpoints/${r}_best.pt" --split "$s" --workers 4 \
            || echo "EXPORT FAILED: $r $s"
    done
done

echo "== analysis"
python -u final_analysis.py || echo "ANALYSIS FAILED"

echo "== profiling"
python -u profile_all.py || echo "PROFILING FAILED"

echo "Job finished at $(date)"