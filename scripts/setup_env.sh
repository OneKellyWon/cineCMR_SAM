#!/bin/bash
# Setup cineCMR_SAM runtime on HPC (no Docker). Mirrors docker/dockerfile + requirements.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQ_FILE="${REPO_ROOT}/docker/requirements_pip.txt"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "==> Repository: ${REPO_ROOT}"
echo "==> Virtualenv: ${VENV_DIR}"

module purge 2>/dev/null || true
module load python/3.10.10-gcc-8.5.0-cjgjmm3
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
# Spack py-torch on login nodes is often CPU-only; install CUDA wheels into venv.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
export PYTHONNOUSERSITE=1

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating venv..."
  python -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel

echo "==> Installing PyTorch 2.0.1 + CUDA 11.8 (matches docker CUDA 11.7)..."
python -m pip install --no-cache-dir \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118 \
  2>&1 | tee -a "${LOG_DIR}/pip_install.log"

echo "==> Installing Python dependencies (see ${REQ_FILE})..."
python -m pip install --no-cache-dir -r "${REQ_FILE}" 2>&1 | tee "${LOG_DIR}/pip_install.log"

echo "==> Installing CTProjector (optional, from dockerfile)..."
python -m pip install --no-cache-dir "git+https://github.com/zhennongchen/CTProjector.git" \
  2>&1 | tee -a "${LOG_DIR}/pip_install.log" || echo "WARN: CTProjector install failed (optional)."

echo "==> Verifying imports..."
python - <<'PY'
import sys
checks = [
    "torch", "torchvision", "numpy", "scipy", "h5py",
    "matplotlib", "SimpleITK", "skimage", "pydicom", "nibabel",
    "pandas", "cv2", "einops", "transformers", "diffusers",
    "segment_anything", "tensorboard", "jupyterlab",
]
failed = []
for m in checks:
    try:
        __import__(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        failed.append(m)
print("Python:", sys.version)
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if failed:
    raise SystemExit(f"Missing modules: {failed}")
PY

cat > "${REPO_ROOT}/scripts/activate_env.sh" <<'ACTIVATE'
#!/bin/bash
# Source this before running notebooks or training:  source scripts/activate_env.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge 2>/dev/null || true
module load python/3.10.10-gcc-8.5.0-cjgjmm3
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
echo "cineCMR_SAM env ready: $(python -c 'import torch; print(torch.__version__)')"
ACTIVATE
chmod +x "${REPO_ROOT}/scripts/activate_env.sh"

echo ""
echo "Done. Activate with:"
echo "  source ${REPO_ROOT}/scripts/activate_env.sh"
echo "On GPU nodes, verify CUDA: python -c 'import torch; print(torch.cuda.is_available())'"
