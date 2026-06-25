#!/bin/bash
#SBATCH --job-name=mnm2_lax_t15_pred
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/mnm2_lax_t15_debug_predict_%j.out
#SBATCH --error=logs/slurm/mnm2_lax_t15_debug_predict_%j.err

cd /gpfs/work/aac/yifansun2302/cineCMR_SAM

mkdir -p logs/slurm

source scripts/activate_env.sh

echo "Prediction job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

python step2_predict_mnm2_lax_t15_debug.py

echo "Prediction job finished at: $(date)"
