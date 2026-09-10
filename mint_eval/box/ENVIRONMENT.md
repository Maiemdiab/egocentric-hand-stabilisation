# The box this actually ran on

Captured from the live instance on 2026-09-10, at 456 of 893 clips completed. This is the record
needed to reproduce the MINT outputs the bridge was measured against; without it, "we used MINT"
is not a reproducible statement.

| | |
|---|---|
| instance | `g6e.xlarge` — 1× NVIDIA L40S 46 GB, 4 vCPU, 30 GB RAM |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-1052-aws |
| driver | 580.126.09 |
| Python | 3.10.12 (venv at `/opt/mint/venv`) |
| torch | 2.8.0 / torchvision 0.23.0 |
| numpy | 1.26.4 |
| model | https://github.com/wuji-technology/wuji-ego-mint.git @ `ddb5d391f378d491d8353560dfc4644e95517654` (2026-09-09) |
| checkpoint | `model.safetensors`, 4,558,656,192 bytes, md5 `caf52bfac91ea7819d350a09ed01c8c6` |

**The model repo has no local modifications** (`git diff` is empty), so the vendored code is exactly
the upstream commit above. Everything that had to be worked around is below.

## Things that had to be worked around

**`chumpy` breaks on numpy ≥ 1.24.** MANO's pickles need chumpy to unpickle, and chumpy 0.70 does
`from numpy import bool, int, float, complex, object, unicode, str` — aliases numpy removed in 1.24.
Line 11 of `chumpy/__init__.py` becomes:

```python
from numpy import nan, inf
```

This is a site-packages edit, not a repo change, so it does **not** show up in `git diff` and will
not reproduce from `requirements-frozen.txt` alone. It has to be re-applied after any reinstall.

**Out of memory at 30.8 GB on 1080p input.** `preprocess_frames` casts to `.float()` at the
*original* resolution before `F.interpolate`, so a 1920×1080 clip allocates far more than the model
needs. Fixed upstream of the model by pre-resizing each clip to exactly the model's input size
(518×378) with ffmpeg, which produces an identical tensor because the resize is a plain scale with
no letterboxing. That is why `all.sh` transcodes before inference.

**`while read … done < file` loses data** when a child consumes stdin — ffmpeg does. It leaves
`read` resuming mid-line, which silently truncates clip names and skips clips. `all.sh` reads the
work list on **fd 3** and passes `-nostdin` to ffmpeg. This bug cost a full sweep once: 28 malformed
S3 prefixes and an unknown number of skipped clips before it was spotted.

## Performance, measured on this box

Batch size and FP8 make **no difference** — the L40S is already compute-bound at batch 1:

| config | forward | ms/frame | wall |
|---|---|---|---|
| baseline (batch 1) | 118.8 s | 48.1 | 147 s |
| `--window-batch-size 4` | 117.4 s | 48.2 | 143 s |
| `--window-batch-size 8` | 119.3 s | 48.1 | 144 s |
| batch 8 + `--fp8-mode auto` | 118.3 s | 48.2 | — |

Time per frame does not move from 48.1 ms between batch 1 and batch 8. If SMs were idle, an 8×
batch would cut it. (`--fp8-mode auto` silently does nothing without `torch.compile` — the log says
`FP8 auto -> off`.) The 8.8 GB VRAM figure is a red herring: VRAM is not the constraint, the SMs are.

Where a 165 s clip actually goes:

| stage | time | GPU |
|---|---|---|
| **MINT forward pass** | **118.8 s (72%)** | busy |
| internal decode + post-processing + 4 S3 uploads | ~25 s | idle |
| CUDA init + 1.1B model load, **once per clip** | 11.4 s | idle |
| `libx264` rescale to 518×378 | 8.0 s | idle |
| S3 download | 1.9 s | idle |

So 28% of each clip is GPU-idle, and that is the entire optimisation headroom — a 1.4× ceiling.
Worth having: overlap the download/transcode with inference (~10 s), keep the model resident across
clips instead of a fresh process per clip (11.4 s), and use NVDEC instead of libx264 (measured
8.0 s → 2.6 s; NVDEC is separate silicon and costs the SMs nothing).

The bigger levers are not utilisation at all. The job is interruption-safe — `all.sh` resumes by
skipping anything already uploaded — which makes it a good fit for **spot**, and that saves more than
every tuning item above combined.

## Files

| file | role |
|---|---|
| `setup.sh` | build the box from bare Ubuntu: venv, torch, the model repo, the checkpoint |
| `all.sh` | the sweep: resumable, self-cleaning, one bad clip never stops it |
| `mint_to_kp.py` | model output → 21 keypoints + 2D, using the model's own intrinsics |
| `requirements-frozen.txt` | exact package set from the running box |

Bucket and prefixes are `$BUCKET`, `$INPUT_PREFIX`, `$OUTPUT_PREFIX` — substitute your own.
