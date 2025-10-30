#!/usr/bin/env bash
# Merged installer: sets up Debian/Ubuntu prerequisites, creates .venv, installs
# base scientific stack, CUDA PyTorch (prefers cu126), then your pinned/project deps.
# Packages pinned in the second block (requirements.txt) override earlier ones.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

# Use sudo if not root (for apt only)
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; else SUDO=""; fi

# PyPI mirror for speed (Aliyun). Change/clear if you prefer the default PyPI.
ALIYUN_INDEX="https://mirrors.aliyun.com/pypi/simple/"

echo "==> [Apt] Installing minimal build prerequisites (no CUDA/driver changes)…"
# $SUDO apt-get update -yq
# $SUDO apt-get install -yq python3-venv python3-pip python3-dev build-essential git curl wget pkg-config

echo "==> [Venv] Creating and activating .venv…"
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [Pip] Upgrading pip/setuptools/wheel inside venv…"
python -m pip install -U --index-url "$ALIYUN_INDEX" pip setuptools wheel

echo "==> [Base Sci/Tools] Installing core scientific stack into venv…"
python -m pip install -U --index-url "$ALIYUN_INDEX" \
  numpy scipy pandas scikit-learn matplotlib pillow tqdm requests seaborn rich

echo "==> [Jupyter] Installing Jupyter tooling into venv…"
python -m pip install -U --index-url "$ALIYUN_INDEX" jupyterlab ipykernel

echo "==> [NLP base] Installing common NLP deps (excluding transformers/accelerate; pinned later)…"
python -m pip install -U --index-url "$ALIYUN_INDEX" datasets sentencepiece

echo "==> [PyTorch] Installing CUDA wheels (try cu126 -> cu124 -> cu121)…"
set +e
PT_OK=0
for CU_TAG in cu126 cu124 cu121; do
  echo "Attempting PyTorch wheels for $CU_TAG …"
  if python -m pip install -U \
       --index-url "https://download.pytorch.org/whl/${CU_TAG}" \
       torch torchvision torchaudio; then
    PT_OK=1
    break
  fi
done
set -e

if [[ "$PT_OK" -ne 1 ]]; then
  echo "GPU wheels not found; installing CPU-only PyTorch from mirror…"
  python -m pip install -U --index-url "$ALIYUN_INDEX" torch torchvision torchaudio
fi

echo "==> [Project requirements] Installing your project deps (pinned versions override)…"
cat > requirements.txt <<'REQ'
omegaconf
iopath
timm
opencv-python
webdataset
scikit-image
decord
wandb
visual-genome
peft==0.9.0
transformers==4.37.2
accelerate==0.21.0
bitsandbytes
diffusers
huggingface_hub[cli]
gdown
REQ

# Use eager strategy so transitive deps are adjusted to satisfy pinned versions
python -m pip install --upgrade --upgrade-strategy eager --index-url "$ALIYUN_INDEX" -r requirements.txt

echo "==> [Verify] Quick checks…"
python - <<'PY'
import sys, subprocess
print("Python:", sys.version.split()[0])
try:
    import torch
    print("PyTorch:", torch.__version__,
          "| torch.version.cuda:", getattr(torch.version, "cuda", None),
          "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    try:
        print("cuDNN version:", torch.backends.cudnn.version())
    except Exception as e:
        print("cuDNN version check failed:", e)
except Exception as e:
    print("PyTorch check failed:", e)
for pkg in ("transformers","accelerate","peft","bitsandbytes","diffusers"):
    try:
        mod = __import__(pkg.replace("-", "_"))
        print(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{pkg}: not importable ({e})")
PY

echo "==> Done. Everything installed inside $(python -c 'import sys, pathlib; print(pathlib.Path(sys.executable).parent.parent)')"
echo "   To use later: source .venv/bin/activate"