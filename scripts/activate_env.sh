#!/bin/bash
# Source before notebooks/training:  source scripts/activate_env.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge 2>/dev/null || true
module load python/3.10.10-gcc-8.5.0-cjgjmm3
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
echo "cineCMR_SAM ready | torch $(python -c 'import torch; print(torch.__version__)') | cuda built: $(python -c 'import torch; print(torch.backends.cuda.is_built())')"
