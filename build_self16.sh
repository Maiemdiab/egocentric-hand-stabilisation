#!/bin/bash
# No-MINT build for the clips whose MINT bridges were ALL short, so dropping MINT costs nothing.
#   C -> self_bridge (own-motion Hermite, max-gap 15) -> rigidify
# Renders  A DELIVERED | C+ (MINT bridge) | SELF (no MINT)  and publishes to its own S3 prefix.
#
# Work-stealing: every worker reads the same list and claims a clip with `mkdir`, which is atomic,
# so N copies of this script share one queue with no coordinator. Resumable -- a clip whose .ok
# exists is skipped.
set -u
cd /Users/maiediab/Documents/POG/hand_jitter_validation
export AWS_PROFILE=stage AWS_DEFAULT_REGION=ap-south-1
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true
PY=/Users/maiediab/Documents/POG/.golden_venv/bin/python3
NPZ=/Users/maiediab/hand_abc_out/_npz
MINT=/Users/maiediab/hand_abc_out/_A_vs_Cplus_vs_MINT/npz
OUT=/Users/maiediab/hand_abc_out/_selfbridge
B=s3://stage-humyn-egocentric-stereo-data
D=$B/labelling_results/hand_pose_mint/_A_vs_MINT_vs_SELF
W=${W:-0}
MAXGAP=${MAXGAP:-15}; BETA=${BETA:-0.5}; H=${H:-16}; ORDER=${ORDER:-3}
mkdir -p "$OUT/npz" "$OUT/done" "$OUT/claim" "$OUT/log"

N=0; F=0
# fd 3: ffmpeg inside the renderer reads stdin and would otherwise eat the work list
while IFS=$'\t' read -r CLIP IND NPZKEY <&3; do
  [ -z "${CLIP:-}" ] && continue
  [ -f "$OUT/done/${CLIP}.ok" ] && continue
  mkdir "$OUT/claim/${CLIP}" 2>/dev/null || continue     # atomic claim; loser moves on
  FREE=$(df -g /Users/maiediab | tail -1 | awk '{print $4}')
  if [ "${FREE:-99}" -lt 12 ]; then echo "[w$W] STOP: only ${FREE}G free"; rmdir "$OUT/claim/${CLIP}"; break; fi

  T=/tmp/self16_w$W; rm -rf $T; mkdir -p $T
  LOG="$OUT/log/${CLIP}.log"; : > "$LOG"
  echo "[w$W] start $CLIP"

  $PY fix/self_bridge.py --ours "$NPZ/${CLIP}_C.npz" --out $T/sb.npz \
      --max-gap $MAXGAP --beta $BETA > $T/self.json 2>>"$LOG" \
      || { echo "[w$W] SELF-FAIL $CLIP"; F=$((F+1)); rm -rf $T; continue; }
  $PY fix/rigidify.py --npz $T/sb.npz --out $T/self_final.npz --h $H --order $ORDER \
      > $T/rigid.json 2>>"$LOG" \
      || { echo "[w$W] RIGID-FAIL $CLIP"; F=$((F+1)); rm -rf $T; continue; }
  cp $T/self_final.npz "$OUT/npz/${CLIP}_SELF.npz"
  cp $T/self.json "$OUT/npz/${CLIP}_self.json"; cp $T/rigid.json "$OUT/npz/${CLIP}_rigid.json"

  aws s3 cp "$B/$NPZKEY" $T/a.npz --quiet 2>>"$LOG" || { echo "[w$W] NO-A $CLIP"; F=$((F+1)); rm -rf $T; continue; }
  aws s3 cp "$B/$IND/left_eye.mp4" $T/v.mp4 --quiet 2>>"$LOG" || { echo "[w$W] NO-VIDEO $CLIP"; F=$((F+1)); rm -rf $T; continue; }

  MINTNPZ="$MINT/${CLIP}_Cplus.npz"
  MARG=""; [ -s "$MINTNPZ" ] && MARG="--fix $MINTNPZ"
  $PY fix/abc_strip.py --video $T/v.mp4 --delivered $T/a.npz $MARG \
      --hand21kp $T/self_final.npz \
      --c-title "SELF  no MINT (own-motion Hermite b=$BETA + rigid)" \
      --out "$OUT/${CLIP}_A_vs_MINT_vs_SELF.mp4" >>"$LOG" 2>&1 \
      || { echo "[w$W] RENDER-FAIL $CLIP"; F=$((F+1)); rm -rf $T; continue; }
  rm -f $T/v.mp4                                          # 336 MB each; drop it the moment it is used

  if [ -s "$OUT/${CLIP}_A_vs_MINT_vs_SELF.mp4" ]; then
    aws s3 cp "$OUT/${CLIP}_A_vs_MINT_vs_SELF.mp4" "$D/${CLIP}_A_vs_MINT_vs_SELF.mp4" --quiet 2>>"$LOG" \
      && aws s3 cp "$OUT/npz/${CLIP}_SELF.npz" "$D/npz/${CLIP}_SELF.npz" --quiet 2>>"$LOG" \
      && { touch "$OUT/done/${CLIP}.ok"; N=$((N+1))
           echo "[w$W] OK $CLIP  $($PY -c "import json;s=json.load(open('$T/self.json'));r=json.load(open('$T/rigid.json'));print(f\"filled={s['bridged']} skip_dis={s['skip_disagree']} tracks={r['tracks']} jit {r['jitter_artic_before']:.4f}->{r['jitter_artic_after']:.4f}\")" 2>/dev/null)"; }
  fi
  rm -rf $T
done 3< /tmp/self16.tsv
echo "[w$W] done: $N built, $F failed"
