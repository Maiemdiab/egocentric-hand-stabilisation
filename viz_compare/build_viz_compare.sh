#!/bin/bash
# Re-render an EXISTING delivery visualisation with new hand labels, and stack it against the
# original so the two can be compared frame for frame.
#
# The point is that only ONE thing differs between the two panels: the hand label file. Same
# renderer, same head pose, same flags, same web conversion. Anything you see is the hand pipeline.
#
# Paths are environment variables so this carries no site-specific locations:
#   CPLUS_DIR     new hand labels           <CPLUS_DIR>/<clip>.npz
#   VIDEO_DIR     source video              <VIDEO_DIR>/<clip>.mp4
#   HEAD_DIR      6-DoF head pose           <HEAD_DIR>/<clip>.npz
#   IMU_DIR       optional accelerometer    <IMU_DIR>/<clip>.csv
#   DELIVERED_DIR the already-rendered originals <DELIVERED_DIR>/<clip>.mp4
#   RENDERER      path to the delivery renderer (see render_v4_panels.patch)
#   OUT_DIR       where results are written
# Optional: SHARD=k/n runs worker k of n over a deterministic split.
set -u
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
CPLUS_DIR=${CPLUS_DIR:?}; VIDEO_DIR=${VIDEO_DIR:?}; HEAD_DIR=${HEAD_DIR:?}
DELIVERED_DIR=${DELIVERED_DIR:?}; RENDERER=${RENDERER:?}; OUT_DIR=${OUT_DIR:?}
IMU_DIR=${IMU_DIR:-}
SHARD=${SHARD:-0/1}; SAFE=$(echo "$SHARD" | tr '/' '_')
mkdir -p "$OUT_DIR"/{done,enh,web,pairs}

$PY - "$CPLUS_DIR" "$OUT_DIR" "$SHARD" <<'PYEOF' > "/tmp/viz_todo.$SAFE.tsv"
import os, sys
src, out, shard = sys.argv[1], sys.argv[2], sys.argv[3]
k, n = (int(x) for x in shard.split('/'))
clips = sorted(f[:-4] for f in os.listdir(src) if f.endswith('.npz'))
for c in clips[k::n]:                       # sorted, so the split is deterministic and disjoint
    if not os.path.exists(f'{out}/done/{c}.ok'):
        print(c)
PYEOF

while read -r CLIP <&3; do
  [ -z "${CLIP:-}" ] && continue
  W=$(mktemp -d); mkdir -p "$W/out"
  CACHED="$OUT_DIR/web/${CLIP}_web.mp4"

  # Rendering is minutes; pairing is seconds. Cache the render so a layout change is cheap.
  if [ ! -s "$CACHED" ]; then
    # 1. new labels -> the schema the delivery renderer expects (a field rename, nothing recomputed)
    "$PY" "$HERE/h21_to_enhanced.py" --src "$CPLUS_DIR/$CLIP.npz" \
        --out "$OUT_DIR/enh/${CLIP}_enhanced_keypoints.npz" >/dev/null || { echo "[$CLIP] convert failed"; rm -rf "$W"; continue; }
    IMUARG=()
    [ -n "$IMU_DIR" ] && [ -f "$IMU_DIR/$CLIP.csv" ] && IMUARG=(--imu "$IMU_DIR/$CLIP.csv")
    # 2. same renderer, same flags the delivery used
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    "$PY" "$RENDERER" --video "$VIDEO_DIR/$CLIP.mp4" \
        --npz "$OUT_DIR/enh/${CLIP}_enhanced_keypoints.npz" --head "$HEAD_DIR/$CLIP.npz" \
        --out "$W/out" --future-sec 3 --past-sec 0 --clip-foot "${IMUARG[@]}" > "$W/render.log" 2>&1
    MADE=$(ls "$W"/out/*_wrist_traj_panels.mp4 2>/dev/null | head -1)
    [ -n "$MADE" ] && [ -s "$MADE" ] || { echo "[$CLIP] render failed: $(tail -2 "$W/render.log" | tr '\n' ' ')"; rm -rf "$W"; continue; }
    # 3. the delivered panels were published through a web conversion. Apply the IDENTICAL step,
    #    or we would be comparing a full-resolution render against a downscaled one.
    ffmpeg -nostdin -v error -y -i "$MADE" \
        -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
        -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart "$CACHED" \
      || { echo "[$CLIP] web conversion failed"; rm -f "$CACHED"; rm -rf "$W"; continue; }
  fi

  # 4. stack the original above the new one, letterbox cropped, time-locked
  "$PY" "$HERE/viz_pair.py" --left "$DELIVERED_DIR/$CLIP.mp4" --left-title "DELIVERED" \
      --right "$CACHED" --right-title "C+  rescue + smoothing + MINT-bridge + rigid" \
      --out "$OUT_DIR/pairs/${CLIP}_compare.mp4" >/dev/null \
    && { touch "$OUT_DIR/done/$CLIP.ok"; echo "[$CLIP] -> $OUT_DIR/pairs/${CLIP}_compare.mp4"; } \
    || echo "[$CLIP] pairing failed"
  rm -rf "$W"
done 3< "/tmp/viz_todo.$SAFE.tsv"
