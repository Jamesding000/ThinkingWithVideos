#!/usr/bin/env bash
set -euxo pipefail

# === 0. Init conda ===
# Adjust this path if your miniconda is elsewhere.
source "$HOME/miniconda/etc/profile.d/conda.sh"

# === 1. Create a fresh env ===
# If you want to recreate, uncomment the remove line.
# conda remove -n verl --all -y || true

conda create -y -n verl python=3.10
conda activate verl

# IMPORTANT: Do NOT install pytorch from conda in this env.
# We'll use official PyTorch wheels via pip (CUDA 12.4).

# === 2. Core PyTorch + CUDA stack (GPU) ===
# Uses the official cu129 wheels (babel)
pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

# Tensordict version used by the original script
pip install tensordict==0.6.2

# Quick sanity check
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
EOF

# === 3. vLLM + HF ecosystem + VERL deps ===
# latest versions of vllm and transformers to support Qwen3
pip install \
  "vllm==0.11.0" \
  "transformer==4.57.1" \
  "accelerate==1.11.0" \
  datasets \
  "peft==0.17.1" \
  hf-transfer \
  "numpy<2.0.0" \
  "pyarrow>=15.0.0" \
  pandas \
  "ray[default]" \
  codetiming \
  hydra-core \
  pylatexenc \
  qwen-vl-utils \
  wandb \
  dill \
  pybind11 \
  liger-kernel \
  mathruler \
  pytest \
  py-spy \
  pre-commit \
  ruff \
  "huggingface-hub[cli]"

# === 4. Other infra deps ===
pip install \
  "nvidia-ml-py>=12.560.30" \
  "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" \
  "pydantic>=2.9" \
  "grpcio>=1.62.1"

# === 5. FlashAttention + OpenCV ===
# Super slow build using pip install or clone repo + python setup.py install
# Fastest way to build is https://github.com/Dao-AILab/flash-attention/issues/945#issuecomment-2948520692
# Find the wheel file at: https://github.com/Dao-AILab/flash-attention/releases
# python version: 3.10
# torch version: 2.8.0
# cuda version: 12.9 (babel)
# cxx11abi: TRUE
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# MAX_JOBS=1 pip install "flash-attn==2.7.4.post1" --no-build-isolation --no-cache-dir
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install opencv-python opencv-fixer

# === 6. Extra basics: plotting / notebooks / analysis ===
pip install \
  decord \
  rouge_score \
  matplotlib \
  seaborn \
  scipy \
  ipykernel \
  notebook

# (Optional) register this env as a Jupyter kernel
python -m ipykernel install --user --name "verl" --display-name "Python (verl)"

# === 7. Install VERL (your repo) ===
# Adjust path if your repo lives somewhere else.
cd "$HOME/James/ThinkingWithVideos/verl"

# Install the package itself without pulling in conflicting deps
pip install --no-cache-dir --no-deps -e .

# If the repo has its own requirements.txt you still want:
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# === 8. Final sanity check ===
python - << 'EOF'
import torch, transformers, vllm
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
import flash_attn
print("flash-attn imported OK")
EOF
