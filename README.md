# Egocentric hand-tracking stabilisation

Three post-processing passes for **21-keypoint hand tracking on head-mounted (egocentric) video**,
plus a review renderer. They operate on the *output* of an existing detection pipeline — no GPU, no
re-detection — and each targets a defect measured rather than assumed.

Written against a pipeline that fuses **MediaPipe Hands** (2D) with **WiLoR/MANO** (3D), but the
inputs are generic: per-detection 2D keypoints, 3D keypoints, frame indices, a per-detection source
tag, and track/handedness labels.

## The three defects

**1. Discarded detections that were never labelled.** If handedness is assigned *per track*, a
detection landing in no surviving track never gets a left/right label and is dropped — even though
both models found it and fusion succeeded. On one measured clip, **100% of the hand-frames the
pipeline "lost" were of this kind.** The gaps sit at track seams: one hand fragmented into 8 tracks
while the other formed 1.

**2. False positives, mostly feet.** A bare foot's toes fit a plausible hand-shaped 21-keypoint
skeleton. Recovering discarded detections brings these back unless you gate first.

**3. Per-frame estimator noise.** Measured by splitting every frame-to-frame step by whether the
estimator changed between those two frames: **~90% of the motion happens with the *same* estimator on
both frames.** It is not source switching — it is that the estimator is re-run from scratch each
frame with no memory of the last one.

## Scripts

### `rescue_handedness.py` — recover unlabelled detections, non-causally

Two rules, applied in order:

- **A · complement.** The frame already shows one confidently labelled hand and the candidate sits on
  the correct side of it. A person has two hands, so the label is *determined*, not inferred. This
  does ~85% of the work.
- **B · temporal.** No anchor on this frame: take the nearest confident detection **before and after**
  — they must **agree**. A one-sided anchor is accepted only in a much tighter window.

Spatial gating scales with palm width plus a per-frame motion budget, so a distant hand can never
donate its label.

**Two quality gates run *before* any label reasoning**, which is the part that matters:

| gate | why |
|---|---|
| cross-model corroboration (`fused` only) | A blind review of ~100 sampled detections put the not-on-a-hand rate at **~7% for `fused`, ~32% for single-model, ~60% for depth-borrowed** rows. |
| gravity-aligned height below the head | Tilt-invariant. Real hands sit a median **0.45 m** below the head; a verified foot false-positive sat at **1.02 m**. |

The second gate is only meaningful on rows with *measured* depth — a row whose depth was borrowed
from a clip median defeats every depth-derived test, which is exactly how one false positive survived
an earlier, looser guard.

> A detection can be unlabelled because nobody worked out *which* hand it is (recoverable), or
> because it is **not a hand** (must stay out). Label rules cannot tell those apart. Gate first.

Measured on one clip: **+35 frames with both hands**, frames with no hand **11 → 2**, zero duplicate
L/R labels, zero frames where left appears to the right of right.

### `temporal_smooth.py` — zero-phase smoothing

Local weighted polynomial fit evaluated *at* the sample, using frames on both sides.

- **zero phase** — a symmetric window adds no lag. An EMA or any causal filter drags the skeleton
  behind the hand. Measured lag: **0.0 frames**.
- **gap aware** — fits over real frame indices inside one track, so a detection gap widens the
  neighbourhood instead of being silently treated as adjacent.
- **weighted** — per-source reliability, so corroborated detections pull harder.
- **robust** — two Tukey biweight passes, so one bad frame doesn't drag its neighbours.

| half-window / order | jitter reduction | p90 wrist speed retained |
|---|---|---|
| h=4, order 2 | −48% | 98% |
| **h=6, order 3 (default)** | **−61%** | **94%** |
| h=8, order 3 | −69% | 91% |

**A note on measuring over-smoothing.** A "displacement on fast frames ÷ displacement on slow frames"
ratio reads ~4.9 here at *every* window and *every* order — it is flat, because fast frames simply
carry more noise from motion blur. It does not measure over-smoothing. Use the **speed distribution**
(p90/p99 before vs after) instead.

Smoothed rows are flagged (`smoothed`) and originals retained (`kp2d_raw`). **A smoothed keypoint is
not a raw per-frame measurement and should not be presented as one.**

### `abc_strip.py` — side-by-side review renderer

One decode pass, OpenCV drawing only, downscaled. Renders 2 or 3 label sets over the same video,
time-locked, with per-panel track and detection counts. Built for triaging hundreds of clips:
**~19 s and ~15 MB per 60 s clip**, against minutes per clip for a 3D-panel renderer.

### `render_rescue_ab.py` — rescue validation renderer

Before/after for the handedness rescue, with rescued detections in **green** and tagged `L*`/`R*` at
the wrist — so you can check the *label*, not just that a skeleton appeared.

## Usage

```bash
python rescue_handedness.py --npz in.npz --out rescued.npz \
    --head head_pose_6dof.npz --imu imu_accel.csv        # gravity gate (optional but recommended)

python temporal_smooth.py  --npz rescued.npz --out final.npz --h 6 --order 3

python abc_strip.py --video clip.mp4 \
    --delivered baseline.npz --hand21kp final.npz --out review.mp4

python render_rescue_ab.py --video clip.mp4 \
    --original in.npz --rescued rescued.npz --out rescue_check.mp4
```

`--delivered` is the baseline label set; `--hand21kp` is the candidate (the flag is named for the
upstream 21-keypoint module). Add `--fix` for a third panel.

## Expected npz fields

`frame_idx (N,)` · `kp2d (N,21,2)` · `kp3d_cam (N,21,3)` · `source (N,)` · `hand (N,)` · `kept (N,)`

`source` values used for weighting and gating: `fused`, `wilor`, `wilor_pnpfail`, `lifted_2d`.
`hand`: `0` left, `1` right, `2` other/bystander, `-1` unlabelled.

## Also here

`mint_eval/` — deployment and inference scripts for evaluating
[MINT](https://github.com/wuji-technology/wuji-ego-mint) as an alternative to per-frame detection,
including the two footguns that cost the most time: host RAM scaling with input resolution rather
than the model's, and CLI defaults that process only ~10 s of a 60 s clip.

## Requirements

`numpy`, `scipy`, `opencv-python`, and `ffmpeg` on PATH.

## Status

Research/engineering code, validated on egocentric ZED footage. The false-positive gates are
heuristics validated against a small number of confirmed cases — **ranking them properly needs a
labelled hand/foot evaluation set, which does not appear to exist publicly.** Treat the thresholds as
starting points for your own footage, not as tuned constants.
