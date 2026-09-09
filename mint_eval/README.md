# Evaluating MINT on your own egocentric footage

Deployment and inference scripts for [wuji-ego-mint](https://github.com/wuji-technology/wuji-ego-mint)
— a 1.1B-parameter model that predicts camera motion and two-hand MANO pose from monocular egocentric
video — set up as a candidate against an existing per-frame hand-tracking pipeline.

Nothing here is specific to one dataset or account: instance id, region, profile and paths are all
environment variables.

## Why this is separate from the main scripts

MINT is a *replacement* for per-frame detection, where the rest of this repo *post-processes* it. It
is worth evaluating because it removes several defect classes structurally rather than filtering
them: one model instead of two fused (so no source-switching discontinuity), MANO output (so bone
lengths are constant by construction), explicit left/right heads (so no handedness inference), and
its own camera trajectory.

## Setup

```bash
export SSM_INSTANCE=i-xxxxxxxxxxxx AWS_PROFILE=... AWS_DEFAULT_REGION=...
./ssm_run.sh <<'CMD'
sudo mkdir -p /opt/mint && sudo chown "$USER" /opt/mint
CMD
# copy setup_mint_gpu.sh to the box, then:
./ssm_run.sh <<'CMD'
setsid nohup /opt/mint/setup_mint_gpu.sh >/dev/null 2>&1 </dev/null &
CMD
```

`ssm_run.sh` exists because an ops-provisioned GPU box often has **no public IP and a key you do not
hold**. If it runs the SSM agent with a suitable instance profile, you need neither.

## Two things that will bite you

### 1. Host RAM scales with your input resolution, not the model's

`model_train/data/transforms.py`:

```python
x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0   # at the ORIGINAL resolution
x = F.interpolate(x, size=(378, 518), ...)             # only then shrinks
```

The float32 conversion happens **before** the resize, so a whole clip is materialised at full
resolution in float32. For a 1823-frame (~60 s at 30 fps) clip:

| input | float32 tensor | result on a 30 GB box |
|---|---|---|
| 1920×1080 | ~45 GB | OOM |
| 1280×720 | ~20 GB | **OOM at 30.8 GB RSS** (measured) |
| 518×378 | ~4.3 GB | fine, ~6 GB total |

`F.interpolate(size=(378,518))` is a plain resize with **no letterboxing**, so pre-scaling the video
to exactly 518×378 with ffmpeg produces the same tensor the model would have built from 1080p.
`run_mint.sh` does this automatically.

Also note `read_video()` holds every decoded frame in a Python list and then `np.stack`s it, which
transiently doubles the uint8 buffer on top of the above.

### 2. The CLI defaults are much smaller than you expect

`mint infer` defaults to `--max-frames 160` at `--target-fps 15` — about **10 seconds** of a 60 s
clip. For a like-for-like comparison against a 30 fps pipeline, pass `--target-fps 30` and raise
`--max-frames`. MINT's native operating rate is 15 fps, so 30 fps is outside its training regime;
worth measuring both.

`--hand-mode` defaults to `smooth`, which runs a constant-velocity UKF with an unscented RTS backward
pass over the hand output. Worth knowing before attributing MINT's stability to the backbone alone.

## MANO

Rendering needs **both** `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` under `assets/mano/`, from
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) under its own licence (non-commercial research).
`run_mint.sh` passes `--no-render`, which produces `prediction.npz` (MANO parameters, camera
extrinsics, FOV, per-frame hand presence) without them. Keypoints can then be decoded with a single
right-hand MANO model, mirroring for the left, as WiLoR/HaMeR do.

## Measured throughput

NVIDIA L40S (g6e.xlarge, 46 GB VRAM, 4 vCPU, 30 GB RAM):

- **47 ms/frame** steady state, 72 ms/frame including model load, at the default 32-frame window
- checkpoint is 4.56 GB, public and un-gated, sha256-pinned by the project's own script

## Licence note

MINT's own code is MIT, but it interfaces with MANO (separate registration, non-commercial research)
and HaWoR (CC BY-NC-ND 4.0). Check these before any commercial use.
