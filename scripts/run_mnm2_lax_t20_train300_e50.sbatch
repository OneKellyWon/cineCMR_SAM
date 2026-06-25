#!/bin/bash
#SBATCH --job-name=mnm2_lax_t20_e50
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/mnm2_lax_t20_train300_e50_%j.out
#SBATCH --error=logs/slurm/mnm2_lax_t20_train300_e50_%j.err

cd /gpfs/work/aac/yifansun2302/cineCMR_SAM

mkdir -p logs/slurm

source scripts/activate_env.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

python step1_train_mnm2_lax_t20_split.py

echo "Job finished at: $(date)"