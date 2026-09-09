#!/usr/bin/env bash
# Run MINT inference on one video at full frame rate, without exhausting host RAM.
#
#   ./run_mint.sh /path/to/clip.mp4 /path/to/outdir [target_fps]
#
# THE POINT OF THE PRE-RESIZE (see README): mint/../model_train/data/transforms.py does
#     x = frames_uint8.permute(0,3,1,2).float() / 255.0     # at the ORIGINAL resolution
#     x = F.interpolate(x, size=(378,518), ...)             # only then shrinks
# so host RAM scales with the resolution you feed it, not with the model's input. A 1823-frame clip
# costs ~20 GB at 720p and ~45 GB at 1080p, which OOMs a 30 GB box. Because the interpolate is a
# plain resize with NO letterboxing, pre-scaling to exactly 518x378 yields the same tensor the model
# would have built, at ~4 GB.
set -euo pipefail
IN="${1:?usage: run_mint.sh input.mp4 outdir [fps]}"
OUT="${2:?usage: run_mint.sh input.mp4 outdir [fps]}"
FPS="${3:-30}"
ROOT="${MINT_ROOT:-/opt/mint}"
W="${MINT_W:-518}"; H="${MINT_H:-378}"

SMALL="${IN%.*}_${W}x${H}.mp4"
if [ ! -s "$SMALL" ]; then
  ffmpeg -v error -y -i "$IN" -vf "scale=${W}:${H}" -c:v libx264 -crf 10 -preset veryfast -an "$SMALL"
fi

cd "$ROOT/repo"
"$ROOT/venv/bin/python" -m mint infer \
  --input "$SMALL" \
  --checkpoint checkpoints/model.safetensors \
  --output "$OUT" \
  --target-fps "$FPS" \
  --max-frames "${MINT_MAX_FRAMES:-100000}" \
  --no-render          # rendering needs MANO_LEFT.pkl + MANO_RIGHT.pkl under assets/mano/
