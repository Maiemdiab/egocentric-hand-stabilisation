#!/bin/bash
# C+ pipeline driver: handedness rescue -> temporal smoothing -> MINT bridge -> exact rigidity,
# then a time-locked review strip. CPU only; no re-detection and no GPU.
#
# Everything is addressed by environment variable so this carries no site-specific paths:
#   OURS_DIR   per-clip detector output   <OURS_DIR>/<clip>.npz
#   MINT_DIR   per-clip MINT keypoints    <MINT_DIR>/<clip>.npz   (optional; no bridge without it)
#   VIDEO_DIR  per-clip source video      <VIDEO_DIR>/<clip>.mp4  (optional; needed only to render)
#   BASE_DIR   per-clip baseline labels   <BASE_DIR>/<clip>.npz   (optional; the "before" panel)
#   OUT_DIR    where results are written
# Tunables (defaults are the measured ones -- see METHOD.md for how each was chosen):
#   MAX_GAP=120  SMOOTH_H=6  SMOOTH_ORDER=3  POSE_H=16  POSE_ORDER=3
#
#   OURS_DIR=out/ours MINT_DIR=out/mint VIDEO_DIR=out/video OUT_DIR=out/cplus ./build_cplus.sh
set -u
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
OURS_DIR=${OURS_DIR:?set OURS_DIR}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
MINT_DIR=${MINT_DIR:-}; VIDEO_DIR=${VIDEO_DIR:-}; BASE_DIR=${BASE_DIR:-}
MAX_GAP=${MAX_GAP:-120}; SMOOTH_H=${SMOOTH_H:-6}; SMOOTH_ORDER=${SMOOTH_ORDER:-3}
POSE_H=${POSE_H:-16}; POSE_ORDER=${POSE_ORDER:-3}
HEAD_DIR=${HEAD_DIR:-}; IMU_DIR=${IMU_DIR:-}
mkdir -p "$OUT_DIR/npz" "$OUT_DIR/strips"

for SRC in "$OURS_DIR"/*.npz; do
  [ -e "$SRC" ] || { echo "no .npz in $OURS_DIR"; exit 1; }
  CLIP=$(basename "$SRC" .npz)
  [ -f "$OUT_DIR/npz/${CLIP}_cplus.npz" ] && continue
  W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

  # 1. recover detections that were found but never given a left/right label
  RESCUE=("$PY" "$HERE/rescue_handedness.py" --npz "$SRC" --out "$W/1_rescued.npz")
  # the gravity gate needs head pose + IMU; without them the source gate still applies
  [ -n "$HEAD_DIR" ] && [ -f "$HEAD_DIR/$CLIP.npz" ] && [ -n "$IMU_DIR" ] && [ -f "$IMU_DIR/$CLIP.csv" ] \
    && RESCUE+=(--head "$HEAD_DIR/$CLIP.npz" --imu "$IMU_DIR/$CLIP.csv")
  "${RESCUE[@]}" > "$W/1_rescue.json" || { echo "[$CLIP] rescue failed"; continue; }

  # 2. zero-phase smoothing: the estimator is re-run per frame with no memory of the last one
  "$PY" "$HERE/temporal_smooth.py" --npz "$W/1_rescued.npz" --out "$W/2_smoothed.npz" \
      --h "$SMOOTH_H" --order "$SMOOTH_ORDER" > "$W/2_smooth.json" || { echo "[$CLIP] smooth failed"; continue; }

  STAGE="$W/2_smoothed.npz"
  # 3. fill gaps with MINT's MOTION, anchored on our own detections at both ends
  if [ -n "$MINT_DIR" ] && [ -f "$MINT_DIR/$CLIP.npz" ]; then
    "$PY" "$HERE/mint_bridge2.py" --ours "$STAGE" --mint "$MINT_DIR/$CLIP.npz" \
        --out "$W/3_bridged.npz" --max-gap "$MAX_GAP" > "$W/3_bridge.json" \
      && STAGE="$W/3_bridged.npz" || echo "[$CLIP] bridge failed, continuing without it"
  fi

  # 4. exact bone rigidity + articulation smoothed in POSE space
  "$PY" "$HERE/rigidify.py" --npz "$STAGE" --out "$OUT_DIR/npz/${CLIP}_cplus.npz" \
      --h "$POSE_H" --order "$POSE_ORDER" > "$W/4_rigid.json" || { echo "[$CLIP] rigidify failed"; continue; }
  for J in 1_rescue 2_smooth 3_bridge 4_rigid; do
    [ -f "$W/$J.json" ] && cp "$W/$J.json" "$OUT_DIR/npz/${CLIP}.$J.json"
  done

  # 5. review strip: baseline | C+ | MINT, time-locked on the source video
  if [ -n "$VIDEO_DIR" ] && [ -f "$VIDEO_DIR/$CLIP.mp4" ]; then
    # the "before" panel: an explicit baseline if given, otherwise this pipeline's own input,
    # which is the comparison that actually matters -- detector output vs detector output + C+
    BEFORE="$SRC"
    [ -n "$BASE_DIR" ] && [ -f "$BASE_DIR/$CLIP.npz" ] && BEFORE="$BASE_DIR/$CLIP.npz"
    STRIP=("$PY" "$HERE/abc_strip.py" --video "$VIDEO_DIR/$CLIP.mp4"
           --delivered "$BEFORE" --hand21kp "$OUT_DIR/npz/${CLIP}_cplus.npz"
           --c-title "C+  rescue + smoothing + MINT-bridge + rigid"
           --out "$OUT_DIR/strips/${CLIP}_review.mp4")
    [ -n "$MINT_DIR" ] && [ -f "$MINT_DIR/$CLIP.npz" ] && STRIP+=(--mint "$MINT_DIR/$CLIP.npz" --mint-title "MINT (raw)")
    "${STRIP[@]}" >/dev/null || echo "[$CLIP] render failed"
  fi

  echo "[$CLIP] done -> $OUT_DIR/npz/${CLIP}_cplus.npz"
  rm -rf "$W"; trap - EXIT
done
