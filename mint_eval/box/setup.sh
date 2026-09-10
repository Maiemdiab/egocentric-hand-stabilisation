#!/bin/bash
# MINT setup on a g6e (L40S). System python is 3.10.12 which is exactly what MINT pins
# (>=3.10,<3.11), so a plain venv replaces their conda env and we avoid their CN pip mirror.
exec > /opt/mint/setup.log 2>&1
set -x
set -e
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg python3.10-venv git curl
cd /opt/mint
[ -d repo ] || git clone --depth 1 https://github.com/wuji-technology/wuji-ego-mint.git repo
python3 -m venv venv
./venv/bin/pip install -q -U pip wheel setuptools
# torch first, from the official index; driver 580 handles cu128
./venv/bin/pip install -q torch==2.8.0 torchvision==0.23.0
# inference deps, minus the robotics extras the single-video CLI never imports
./venv/bin/pip install -q decord==0.6.0 einops==0.8.2 huggingface-hub==1.20.1 \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 PyYAML==6.0.3 safetensors==0.8.0 \
  scipy==1.15.3 smplx==0.1.28 tqdm==4.68.3 Flask==3.1.3 pyarrow==24.0.0
cd repo
./../venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# checkpoint: 4.56 GB, public, sha256-pinned by their script
mkdir -p checkpoints
if [ ! -s checkpoints/model.safetensors ]; then
  curl -fL --retry 5 --retry-all-errors -o checkpoints/model.safetensors \
    "https://huggingface.co/ZZJAsher/mint_v1/resolve/main/model.safetensors"
fi
echo "7b2f0aa5dfd00c271bb2f12c841ccfcc70e81e4052d413eacb5fb42a1bcc36c8  checkpoints/model.safetensors" | sha256sum --check
echo "SETUP_COMPLETE"
