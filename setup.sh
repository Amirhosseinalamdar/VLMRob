#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

# Use sudo if not root
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; else SUDO=""; fi

ALIYUN_INDEX="https://mirrors.aliyun.com/pypi/simple/"

# 1) Essentials (no CUDA/driver packages)
$SUDO apt-get update -yq
$SUDO apt-get install -yq python3-pip python3-dev build-essential git curl wget pkg-config

# 2) Upgrade pip tooling (system-wide) from Aliyun
$SUDO -H python3 -m pip install -U --index-url "$ALIYUN_INDEX" pip setuptools wheel

# 3) Core scientific + tooling from Aliyun
$SUDO -H python3 -m pip install -U --index-url "$ALIYUN_INDEX" \
  numpy scipy pandas scikit-learn matplotlib pillow tqdm requests seaborn rich

# 4) Jupyter (optional) from Aliyun
$SUDO -H python3 -m pip install -U --index-url "$ALIYUN_INDEX" jupyterlab ipykernel

# 5) NLP stack from Aliyun
$SUDO -H python3 -m pip install -U --index-url "$ALIYUN_INDEX" \
  transformers datasets accelerate sentencepiece

# 6) PyTorch GPU wheels: try CUDA 12.6 first, then 12.4, 12.1 (no system CUDA changes)
set +e
PT_OK=0
for CU_TAG in cu126 cu124 cu121; do
  echo "Attempting PyTorch wheels for $CU_TAG ..."
  if $SUDO -H python3 -m pip install --upgrade \
       --index-url "https://download.pytorch.org/whl/${CU_TAG}" \
       torch torchvision torchaudio; then
    PT_OK=1
    break
  fi
done
set -e

# Fallback: CPU-only PyTorch from Aliyun
if [[ "$PT_OK" -ne 1 ]]; then
  echo "GPU wheels not found; installing CPU-only PyTorch from Aliyun..."
  $SUDO -H python3 -m pip install -U --index-url "$ALIYUN_INDEX" torch torchvision torchaudio
fi

# 7) Quick verification (does not modify anything)
python3 - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("PyTorch:", torch.__version__,
          "| torch.version.cuda:", getattr(torch.version, "cuda", None),
          "| CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch check failed:", e)
PY

echo "Done. Installed system-wide with pip (no virtualenv, no CUDA/driver changes)."
