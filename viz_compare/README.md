# Comparing a new hand pipeline inside the delivery visualisation

The review strips in the parent directory draw the skeleton on the ego view, which is the right tool
for judging the labels. But the customer sees a different artefact: a multi-panel visualisation that
combines the hands with 6-DoF head pose and wrist trajectories. A fix can look convincing on a
review strip and still not answer *"is the delivered video still shaky?"*

This directory re-renders that delivery visualisation with new hand labels and stacks it against the
original, so the question can be answered in the form the customer actually complained about.

## The one rule that makes it a fair test

**Only the hand label file differs.** Same renderer, same head pose, same flags
(`--future-sec 3 --past-sec 0 --clip-foot`), same web conversion. Anything you see between the two
panels is the hand pipeline and nothing else.

That last item is easy to get wrong and it matters. The delivered panels were not published raw —
they went through `scale=1920:-2, pad=1920:1080, crf 18` on the way out. Comparing a fresh
full-resolution render against that downscaled original makes the new one look sharper and steadier
for reasons that have nothing to do with the labels. The driver applies the identical conversion to
the new render before pairing.

## Layout

Panels are stacked **vertically**, original on top. Side by side, two 2892-wide panels come to
3840+ px across and each half is unreadable on a normal screen; stacked, each keeps its full width
and the eye compares the same region by looking straight up and down.

The letterbox bars the web conversion padded in — about a third of each panel's height — are
detected (by scanning for pure-black rows over several frames, so it adapts rather than assuming a
fixed geometry) and cropped from both sides identically.

## Files

| file | role |
|---|---|
| `build_viz_compare.sh` | the driver: convert -> render -> web-convert -> stack. Resumable, sharded. |
| `h21_to_enhanced.py` | maps the new labels onto the delivery renderer's `_enhanced_keypoints` schema. A field rename; nothing is recomputed. |
| `viz_pair.py` | stacks two rendered panels, time-locked, with letterbox cropping. |
| `render_v4_panels.patch` | two fixes to the delivery renderer (see below). The renderer itself is not included here. |

### `render_v4_panels.patch`

The delivery renderer is your own code, so only the changes are published. Two fixes, +31 lines:

- **Honour the `drawn` mask.** The post-process picks one 2D source per track and flags the minority
  rows. Those rows stay in the label file but must not be drawn, or the skeleton teleports ~0.46 palm
  widths every time the renderer swaps between MediaPipe and WiLoR pixels inside one track.
- **Freeze handedness per track.** The stock `assign_handedness` writes a per-frame label, so the
  left/right colour swaps mid-track whenever the two hands cross in x. A track is one physical hand;
  its identity is decided once and held.

On a stock label file the patched renderer produces a byte-identical result, so it is safe to use for
both halves of the comparison.

## Running it

```bash
CPLUS_DIR=out/cplus/npz \
VIDEO_DIR=out/video \
HEAD_DIR=out/head \
IMU_DIR=out/imu \
DELIVERED_DIR=out/delivered_panels \
RENDERER=/path/to/render_v4_panels.py \
OUT_DIR=out/compare \
./build_viz_compare.sh
```

Parallelise with `SHARD=k/n` (deterministic, disjoint split over a sorted clip list):

```bash
for k in 0 1 2 3 4 5; do SHARD=$k/6 ./build_viz_compare.sh & done
```

## Two things worth knowing before you scale it up

**Rendering is minutes per clip, and it is disk-bound, not CPU-bound.** Each render writes a ~300 MB
uncompressed intermediate before re-encoding. Six concurrent workers hold ~4.5 GB of transient space
and generate enough write traffic that load average is dominated by I/O waiters — on a 14-core
machine, six workers showed a load average near 50 while only ~60% of CPU was busy. Adding workers
past that point makes it slower, not faster. Check free disk before choosing `n`.

**The render is cached.** `build_viz_compare.sh` keeps each converted panel under `web/`, so changing
the comparison layout re-pairs in seconds instead of re-rendering hours of panels. This exists
because the layout changed once already — side-by-side turned out to be unreadable — and re-rendering
everything to fix a title bar is a bad trade.

**A killed run leaves its work behind.** The render is a child process and outlives the shell that
started it, so stopping the driver mid-clip leaves a several-hundred-MB temp directory. The driver
uses `mktemp -d`; sweep for stale ones if you interrupt a large run.
