#!/usr/bin/env bash
# Install MINT (wuji-ego-mint) for inference on an Ubuntu 22.04 GPU box.
#
# Deliberately does NOT use the project's scripts/create_env.sh: that builds a conda env and defaults
# its pip index to a mirror that may be slow or unreachable outside CN. Ubuntu 22.04 ships Python
# 3.10, which is exactly what the project pins (>=3.10,<3.11), so a plain venv is equivalent and
# simpler.
#
# Installs the inference subset only. mujoco / nlopt / pin are robotics extras that the single-video
# CLI never imports and that are the most likely to fail a source build.
exec > "${MINT_ROOT:-/opt/mint}/setup.log" 2>&1
set -x
set -e
ROOT="${MINT_ROOT:-/opt/mint}"
CKPT_URL="https://huggingface.co/ZZJAsher/mint_v1/resolve/main/model.safetensors"
CKPT_SHA="7b2f0aa5dfd00c271bb2f12c841ccfcc70e81e4052d413eacb5fb42a1bcc36c8"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg python3.10-venv git curl
cd "$ROOT"
[ -d repo ] || git clone --depth 1 https://github.com/wuji-technology/wuji-ego-mint.git repo
python3 -m venv venv
./venv/bin/pip install -q -U pip wheel setuptools
./venv/bin/pip install -q torch==2.8.0 torchvision==0.23.0
./venv/bin/pip install -q decord==0.6.0 einops==0.8.2 huggingface-hub==1.20.1 \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 PyYAML==6.0.3 safetensors==0.8.0 \
  scipy==1.15.3 smplx==0.1.28 tqdm==4.68.3 Flask==3.1.3 pyarrow==24.0.0
cd repo
../venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
mkdir -p checkpoints
if [ ! -s checkpoints/model.safetensors ]; then
  curl -fL --retry 5 --retry-all-errors -o checkpoints/model.safetensors "$CKPT_URL"
fi
echo "$CKPT_SHA  checkpoints/model.safetensors" | sha256sum --check
echo "SETUP_COMPLETE"
