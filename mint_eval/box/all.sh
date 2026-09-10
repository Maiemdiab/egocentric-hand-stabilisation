#!/bin/bash
# MINT over every ZED clip. Resumable, self-cleaning, one bad clip never stops it.
# Uses the instance profile for S3, so there is no session token to expire mid-run.
exec >> /opt/mint/all.log 2>&1
set -x
B=s3://$BUCKET
OUT=$B/$OUTPUT_PREFIX
R=/opt/mint
aws s3 cp $OUT/_clips.tsv $R/clips.tsv --quiet
TOTAL=$(wc -l < $R/clips.tsv); I=0; T_START=$(date +%s)
# read on fd 3: a child that consumes stdin (ffmpeg does) would otherwise eat clip-list
# bytes and leave `read` resuming MID-LINE -- truncated names and silently skipped clips
while IFS=$'\t' read -r NAME PATHK <&3; do
  I=$((I+1))
  [ -z "$NAME" ] && continue
  # resume: skip anything already uploaded
  if aws s3 ls "$OUT/$NAME/${NAME}_mint_prediction.npz" >/dev/null 2>&1; then continue; fi
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 20 ]; then echo "STOP: only ${FREE}G free"; break; fi
  T0=$(date +%s)
  W=$R/w; rm -rf $W; mkdir -p $W
  if ! aws s3 cp "$B/$PATHK/left_eye.mp4" $W/v.mp4 --quiet; then echo "NOVIDEO $NAME"; continue; fi
  # resize to the model's own input size: keeps preprocess_frames' float32 cast at ~4 GB
  if ! ffmpeg -nostdin -v error -y -i $W/v.mp4 -vf scale=518:378 -c:v libx264 -crf 10 -preset veryfast -an $W/s.mp4; then
    echo "FFMPEGFAIL $NAME"; rm -rf $W; continue; fi
  rm -f $W/v.mp4
  rm -rf $R/o; mkdir -p $R/o
  if ( cd $R/repo && timeout 1800 $R/venv/bin/python -m mint infer --input $W/s.mp4 \
        --checkpoint checkpoints/model.safetensors --output $R/o \
        --target-fps 30 --max-frames 100000 --no-render ); then
    ( cd $R && $R/venv/bin/python mint_to_kp.py --prediction $R/o/prediction.npz --out $R/o/mint_kp.npz ) || echo "DECODEFAIL $NAME"
    printf '{"source_s3":"%s","name":"%s","target_fps":30,"input":"518x378"}\n' "$PATHK" "$NAME" > $R/o/mint_meta.json
    for F in prediction.npz mint_kp.npz summary.json mint_meta.json; do
      [ -s "$R/o/$F" ] && aws s3 cp "$R/o/$F" "$OUT/$NAME/${NAME}_mint_${F}" --quiet
    done
    T1=$(date +%s); EL=$((T1-T_START))
    echo "OK $I/$TOTAL $NAME $((T1-T0))s elapsed_h=$((EL/3600))"
  else
    echo "INFERFAIL $NAME"
  fi
  rm -rf $W $R/o
done 3< $R/clips.tsv
echo "ALL_COMPLETE"
