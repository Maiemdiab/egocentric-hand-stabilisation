#!/bin/bash
# TOP = the delivered visualisation as shipped.
# BOTTOM = C+ hands + SMOOTHED head pose, same renderer, same flags, same web conversion.
#
# Env: BUCKET MANIFEST CPLUS_DIR RENDERER OUT_DIR [DEST_PREFIX DELIVERED_PREFIX WIN SHARD]
# BOTTOM = C+ hands + smoothed head pose, through the same renderer and the same web conversion.
set -u
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
B=${BUCKET:?set BUCKET, e.g. s3://my-bucket}
DST=${DEST_PREFIX:-compare/A_vs_Cplus_smoothhead}
OUT=${OUT_DIR:?set OUT_DIR}; mkdir -p "$OUT/done" "$OUT/enh"
CPD=${CPLUS_DIR:?set CPLUS_DIR}
WIN=${WIN:-4}
SHARD=${SHARD:-0/1}; SAFE=$(echo "$SHARD" | tr "/" "_")
# fd 3: ffmpeg inside the renderer eats stdin and would truncate this work list
$PY - "${1:-/tmp/ten.txt}" "$SHARD" > "/tmp/render_todo.$SAFE.txt" <<'PYEOF'
import sys
src, shard = sys.argv[1], sys.argv[2]
k, n = (int(x) for x in shard.split('/'))
clips = sorted(l.strip() for l in open(src) if l.strip())
for c in clips[k::n]: print(c)      # sorted -> deterministic and disjoint
PYEOF
while read -r CLIP <&3; do
  [ -z "${CLIP:-}" ] && continue
  [ -f "$OUT/done/$CLIP.ok" ] && continue
  [ -s "$CPD/${CLIP}_Cplus.npz" ] || { echo "[$CLIP] no C+ yet, skipping"; continue; }
  FREE=$(df -g "$OUT" | tail -1 | awk "{print \$4}")
  [ "${FREE:-99}" -lt 4 ] && { echo "STOP: only ${FREE}G free"; break; }
  W=$(mktemp -d); mkdir -p "$W/out"
  read -r IND HK < <($PY -c "
import json,sys,os
m=json.load(open(os.environ['MANIFEST']))
r=[x for x in m if x['clip']=='$CLIP'][0]; print(r['input_dir'].rstrip('/'), r['head_key'])")
  aws s3 cp "$B/$IND/left_eye.mp4" "$W/left_eye.mp4" --quiet || { echo "[$CLIP] no video"; rm -rf $W; continue; }
  aws s3 cp "$B/$HK" "$W/head.npz" --quiet || { echo "[$CLIP] no head pose"; rm -rf $W; continue; }
  aws s3 cp "$B/${DELIVERED_PREFIX:-labelling_results/viz_delivery}/${CLIP}_wrist_traj_panels.mp4" "$W/A.mp4" --quiet \
    || { echo "[$CLIP] no delivered viz"; rm -rf $W; continue; }
  $PY "$HERE/h21_to_enhanced.py" --src "$CPD/${CLIP}_Cplus.npz" --out "$OUT/enh/${CLIP}_enh.npz" >/dev/null \
    || { echo "[$CLIP] convert failed"; rm -rf $W; continue; }
  $PY "$HERE/../smooth_head_pose.py" --head "$W/head.npz" --out "$W/head_sm.npz" --win $WIN > "$W/sm.json" \
    || { echo "[$CLIP] head smoothing failed"; rm -rf $W; continue; }
  IMU=(); aws s3 cp "$B/$IND/imu_accel.csv" "$W/imu.csv" --quiet 2>/dev/null && IMU=(--imu "$W/imu.csv")
  OMP_NUM_THREADS=1 MPLBACKEND=Agg $PY "${RENDERER:?set RENDERER}" --video "$W/left_eye.mp4" \
      --npz "$OUT/enh/${CLIP}_enh.npz" --head "$W/head_sm.npz" --out "$W/out" \
      --future-sec 3 --past-sec 0 --clip-foot "${IMU[@]}" > "$W/r.log" 2>&1
  MADE=$(ls "$W"/out/*_wrist_traj_panels.mp4 2>/dev/null | head -1)
  [ -n "$MADE" ] || { echo "[$CLIP] render failed: $(tail -2 "$W/r.log"|tr '\n' ' ')"; rm -rf $W; continue; }
  # identical web conversion to the one the delivered panels went through
  ffmpeg -nostdin -v error -y -i "$MADE" -vf \
     "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
     -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart "$W/B.mp4" \
     || { echo "[$CLIP] web conv failed"; rm -rf $W; continue; }
  $PY "$HERE/viz_pair.py" --left "$W/A.mp4" --left-title "A  DELIVERED" \
      --right "$W/B.mp4" --right-title "C+ hands + SMOOTHED head pose (win=$WIN)" \
      --out "$OUT/${CLIP}_A_vs_Cplus_smoothhead.mp4" > "$W/p.log" 2>&1 \
    && aws s3 cp "$OUT/${CLIP}_A_vs_Cplus_smoothhead.mp4" "$B/$DST/${CLIP}_A_vs_Cplus_smoothhead.mp4" --quiet \
    && { touch "$OUT/done/$CLIP.ok"; echo "[$CLIP] published"; rm -f "$OUT/${CLIP}_A_vs_Cplus_smoothhead.mp4"; } \
    || echo "[$CLIP] pairing failed"
  rm -rf "$W"
done 3< "/tmp/render_todo.$SAFE.txt"
echo "done: $(ls $OUT/done 2>/dev/null | wc -l | tr -d ' ')"
