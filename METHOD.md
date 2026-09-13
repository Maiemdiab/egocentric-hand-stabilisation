# How C+ was built, and why each choice was made

This is the decision log for the pipeline in `build_cplus.sh`. Every parameter below was chosen from
a measurement, and the measurement is given alongside it. Several of the things that look like
obvious improvements were tried and **made results worse** — those are recorded too, because the
negative results are what justify the current shape.

Scale of the evidence: 176 clips of egocentric video (1920×1080, 30 fps, ~1800 frames each) for
which both our detector output and a second model's output exist.

---

## 0. The starting point

A delivered 21-keypoint hand dataset was rejected by the customer as **"shakey"** in the review
video. "Shakey" is not a metric, so the first job was to find out what was actually moving.

Splitting the frame-to-frame motion of the drawn skeleton over 25 clips gave a jitter budget:

| cause | share |
|---|---|
| per-frame detector noise | **83%** |
| teleports when the estimator source switches | 12% |
| re-appearance after a dropout | 5% |

That budget decided everything downstream. In particular it ruled out the fix the team had proposed
(smoothing bounding-box tracks before cropping), which addresses about **1.5%** of the visible shake.

A second measurement found a genuine bug rather than a model limitation: the delivered `fuse()` let
all 21 keypoints slide independently, so **bone-length coefficient of variation was 12.32% against a
0.11% reference — 112× inflated.** A hand whose bones change length every frame *cannot* look stable,
regardless of how good the detector is.

---

## 1. Recover detections that were found but never labelled

`rescue_handedness.py`

Handedness was assigned **per track**. A detection landing in no surviving track therefore never got
a left/right label and was dropped — even though both models found it and fusion succeeded. On the
diagnostic clip that was 167 rows, and **100% of the hand-frames the pipeline "lost" were of this
kind**: nothing was missing, it was discarded.

The pass looks both ways in time (it is offline, so acausal costs nothing) and recovers a row only
where the label is *forced* rather than guessed:

- **Rule A — complement.** The frame already shows one confidently-labelled hand, the candidate is on
  the correct side of it, and the opposite label is free. A person has two hands, so the label is
  determined.
- **Rule B — temporal.** No anchor on this frame: take the nearest confident detection before *and*
  after that is spatially compatible. If both exist they must **agree**. A single-sided anchor is
  accepted only inside a much tighter window.

### The false-positive regression, and what fixed it

The first version resurrected a **foot** — toes fit a plausible hand-shaped 21-keypoint skeleton.
Two gates were added, and the order in which they failed is instructive:

1. **Cross-model corroboration.** Rescue only rows both models agreed on. The team's blind review put
   the not-on-a-hand rate at ~7% for `fused`, ~32% for `wilor`-only, ~60% for `lifted_2d`.
2. **Gravity gate.** Height of the wrist below the head *along the IMU gravity vector*. An earlier
   attempt using image-y and camera-frame Z had failed because both are confounded by head tilt —
   look down and a hand's Z grows exactly like a foot's. Rotating into the world removes the
   confound.

A second foot survived gate 1 alone. The reason matters: its source was `lifted_2d`, whose depth is a
**borrowed clip median**, so every depth-derived test on it — including the gravity gate — is
meaningless. `lifted_2d` is the *worst* class, not a safe one.

> **Later correction.** The gate was written as `source == 'fused'`, which also excluded
> `wilor_pnpfail`. Reading the fusion source shows `wilor_pnpfail` is a **matched** MediaPipe/WiLoR
> pair whose PnP refit merely failed — the same two-model evidence as `fused`. It belongs inside the
> gate. Measured impact is small (55 rows over 3 clips) because the rescue only ever considers
> *unlabelled* rows, but the gate is wrong as written.

---

## 2. Zero-phase temporal smoothing

`temporal_smooth.py`

The residual vibration was tested against the obvious hypothesis — that it comes from the estimator
switching between models — and the hypothesis failed: **~90% of the frame-to-frame motion happens
with the same estimator on both frames.** It is not source switching. The estimator is simply re-run
from scratch every frame with no memory of the previous one, and nothing upstream filters that.

The filter is a local weighted polynomial fit evaluated *at* the sample, using frames on both sides:

- **Zero phase.** Offline, so a symmetric window is free and introduces **no lag** — unlike an EMA or
  any causal filter, which drags the skeleton behind the hand.
- **Gap aware.** Fitting happens over real frame indices inside one track, so a detection gap widens
  the neighbourhood instead of being silently treated as adjacent.
- **Source weighted.** `fused` is trusted above `wilor` and `lifted_2d`.
- **Robust.** Two Tukey biweight passes, so one bad frame bends the curve toward itself rather than
  dragging its neighbours.

Result: **−61.5% jitter, 0 frames of lag, 94% of p90 speed retained.**

> **A discarded test.** An earlier over-smoothing check — "ratio of fast-frame to slow-frame
> displacement ≤ 2.5" — reads ~4.9 at *every* window and order. It is flat because fast frames carry
> more motion-blur noise, so it measures nothing. The valid test is the **p90/p99 speed
> distribution**, used everywhere below.

---

## 3. Bridge detection gaps with the second model's *motion*

`mint_bridge2.py`

The second model (MINT, a 1.1B-parameter MANO-output transformer) was evaluated as a replacement and
**failed the pre-committed 2D-alignment criterion**: 0.81 palm widths off on average, only 27% of
frames within half a palm, and the bias *wanders* ±40 px over about a second.

But separating placement from motion gave the key result:

| | our detector | MINT |
|---|---|---|
| absolute 2D placement | good | **0.81 palm off, slowly wandering** |
| frame-to-frame motion | 12–17 px step | **6.9 px step** |

**MINT's motion is ~7× better than its placement.** So: anchor on *our* detection at each end of a
gap, carry MINT's frame-to-frame deltas inward from both sides, and blend by position. The shared
bias cancels at both ends and is linearly interpolated between.

Depth is interpolated from **our** anchors, never taken from MINT, whose metric scale runs ~47% deep.

### Held-out validation

Hide L real frames inside a run both models cover, reconstruct them, compare to what was hidden.
154 clips; median error in palm widths:

| gap | lerp | **two-sided** | window-anchor | one-sided |
|---|---|---|---|---|
| 10 | 0.264 | **0.150** | 0.212 | 0.318 |
| 30 | 0.605 | **0.262** | 0.299 | 0.441 |
| 60 | 0.958 | **0.366** | 0.387 | 0.555 |
| 90 | 1.114 | **0.407** | 0.426 | 0.613 |

Three conclusions, two of them against expectation:

1. **A single-frame anchor beats a window-median anchor.** The two-sided blend cancels the bias
   *exactly at* the anchors, so widening the anchor only mixes in frames where it has already
   drifted. The "more robust" version is worse at every gap length.
2. **One-sided extension is worse than two-sided at any length** — extending 10 frames off one anchor
   is worse than interpolating across a 90-frame hole between two. It stays off.
3. **Error grows slowly with gap length**, which is what makes a wide gap cap viable at all.

### Gates: what predicts a bad bridge, and what does not

**Anchor-offset disagreement** — the difference between the our-minus-MINT offset at the left anchor
and at the right anchor — is the useful signal. If MINT holds the same hand we do, its bias drifts
slowly and the two offsets agree; if it latched onto a *different* hand mid-gap, they diverge:

| anchor disagreement | n | median error | p90 |
|---|---|---|---|
| < 0.25 palm | 751 | **0.132** | 0.424 |
| 0.5 – 1.0 | 569 | 0.359 | 0.880 |
| 2 – 4 | 156 | 0.604 | 1.213 |
| > 4 | 83 | **0.864** | 2.183 |

**MINT's own confidence predicts nothing**, and both confidence gates were removed:

- correlation with bridge error **−0.064**, against **0.407** for anchor disagreement;
- gating at 0.80 removed 0.3% of surviving gaps and left median error unchanged (0.242 either way);
- worse, the per-frame floor fired exactly where it hurt — MINT dips toward its 0.5 floor around its
  *own* dropouts, vetoing precisely the gaps that hole-filling exists to recover.

The calibration that originally motivated the confidence gate measured whether MINT *agrees with us*,
which is a different question from whether the bridge lands. Reusing it was a mistake.

### Bridging across the second model's own dropouts

An all-or-nothing rule ("MINT must be present on every frame of the span") was rejecting more than
the gap cap did: gaps MINT covers only *partly* hold **48.8%** of our missing frames — more than the
fully-covered ones. Short MINT dropouts are now interpolated. Cost, measured at gap 30:

| MINT hole filled | median error | vs no hole |
|---|---|---|
| 2 frames | 0.255 | +0.1% |
| 5 frames | 0.257 | +1.2% |
| **10 frames** | 0.271 | +6.6% |
| 20 frames | 0.346 | +35.9% |
| 25 frames | 0.434 | +70.6% |

The cap sits at **20** because that is where filling stops beating the alternative: splitting the gap
and extending one-sided measures 0.441, a 20-frame fill measures 0.346. A second cap stops a hole
spanning more than half a gap, which would silently degenerate into plain interpolation while still
looking like a motion-model bridge.

### The known weakness of this validation

The held-out study measures **placement, never presence**. It can only sample gaps our detector *did*
cover at both ends, so it measures easier conditions than the gaps actually filled. Adjudicating
bridged rows visually by gap length:

| gap | on a real hand |
|---|---|
| 1–10 frames | 8/8 |
| 11–20 | 5/8 |
| 21–45 | 2/8 |
| 46+ | 1/8 |

Placement barely degrades (0.13 palm at gap 5, 0.41 at 45) but **the hand leaves the frame**. A wide
`--max-gap` therefore produces well-placed skeletons on nothing. `--max-gap 20` is the defensible
setting; the default is wider only because recall was explicitly prioritised. Every bridged row
carries `bridge_disagree` and `bridge_gap` so a stricter subset can be selected without re-running.

---

## 4. Exact rigidity, with articulation smoothed in pose space

`rigidify.py`, `kinematic.py`

Splitting the residual vibration into the wrist (global) and the fingers relative to the wrist
(articulation) showed where the remaining shake lives:

| | global | articulation |
|---|---|---|
| after stages 1–3 | 0.0563 | 0.0560 |
| MINT | 0.0361 | **0.0142** |

MINT wins articulation by **3.9×** because it emits MANO parameters: a low-dimensional articulated
model physically *cannot* make one fingertip shimmer independently of its neighbours. Our `fuse()`
lets all 21 points slide, so it can, and does.

The fix gives our data the same structural property. The hand is **parameterised**, not constrained:

```
palm  {0,5,9,13,17}   a rigid body, 6 DoF, in the canonical shape of this clip's own hand
thumb 1-2-3-4         a chain off the wrist: fixed link lengths, free directions
index 6-7-8           a chain off MCP 5    (likewise middle / ring / pinky)
```

38 parameters. Bone lengths cannot deviate no matter what the optimiser does, so **bone-length CV is
exactly 0.0000%**.

> Alternating projection — push lengths, pull back to the observed rays — was tried first and
> plateaus at **4–6% CV at every window and order**, because the observed rays and the true bone
> lengths are frequently not simultaneously satisfiable. Parameterisation makes rigidity structural
> instead of negotiated.

Then the crucial step: **smooth the parameters, not the keypoints.** Any smoothed parameter vector
still maps to an exactly-rigid hand; a smoothed set of keypoints does not. Two representations are
deliberately *not* smoothed as raw numbers:

- the palm rotation is smoothed as a **rotation matrix** and re-orthonormalised by SVD, because a
  rotation vector flips sign near π and would tear;
- each link is smoothed as a **unit direction vector** and renormalised, because the (θ, φ) angles
  wrap at ±π and averaging across the wrap points the finger backwards.

### Choosing the smoothing strength

Fitting alone is a *regression* — the per-frame least-squares fit is noisier than its input
(articulation 0.048 → 0.081). The smoothing is what makes the stage a win:

| pose-smooth h | global | articulation | p90 motion kept | p99 kept |
|---|---|---|---|---|
| none | 0.0776 | 0.0811 | — | — |
| 5 | 0.0466 | 0.0467 | 98% | 95% |
| **16** | 0.0298 | 0.0199 | **85%** | **94%** |
| 24 | 0.0233 | 0.0131 | 68% | 84% |
| *MINT, for reference* | *0.0361* | *0.0142* | ***60%*** | ***36%*** |

The reference row is the important one. **MINT buys its stability by suppressing real motion** — it
retains only 36% of p99 wrist speed. At h=24 this pipeline beats MINT on *both* jitter axes while
still keeping more real movement than MINT does. h=16 is the shipped default because this is training
data and motion fidelity matters; h=24 is available for maximum stability.

Rows in tracks too short to fit (< 4 frames) are dropped rather than left breathing at ~14%, so the
delivered set is exactly rigid. One canonical hand is estimated per hand side per clip — the wearer's
bones do not change length between tracks.

---

## 5. Things that were tried and rejected

Recorded because each looked like an obvious improvement.

**Using the second model as the base, with our detections correcting its drift.** The most attractive
alternative, and it fails on five measurements. A drift-corrected MINT base has a **flat error floor
of 0.51–0.57 palm** at every gap length and every smoothing bandwidth — it never depends on anchors,
so it never improves, and the anchored bridge still beats it at gap 150. The floor is MINT's
articulation, not a correctable camera bias: MediaPipe and WiLoR disagree with *each other* by 0.159
palm on the same rows, while MINT disagrees with us by 0.740 — **MINT is the outlier of the three.**
Worst of all, its 100% frame coverage is substantially fabricated: on frames our pipeline covers
nothing, **18 of 20 sampled MINT detections sit on a chair seat, a chair back, or a bare forearm**, at
confidence 0.70–0.99.

**Using the second model to referee our false positives.** Within the single-model classes, distance
to the nearest MINT hand separates on-hand from off-hand with **AUC 0.586** — chance. MINT
hallucinates on the same torso, forearm and chair that a single-model detection does. It is an
excellent motion model and a useless referee. Nothing else separated either: `mp_score` 0.476,
fusion residual 0.500, palm size 0.411.

**Row-level corroboration gating.** Dropping every single-model row lifts precision but costs far too
much: on one clip it discarded 578 rows while the false-positive rate stayed flat (1.9% → 2.0%).
Roughly 35% of single-model rows *are* real hands.

**Track-level corroboration gating** (drop a track only if *no* row along it was corroborated) is a
strictly better version of the same idea — on the worst clip it keeps 2.3× the rows of the row gate
at the same false-positive rate — but on clips whose false positives sit *inside* otherwise-good
tracks it buys nothing. It is available in this repo's history but is not part of C+.

**Loosening the tracker to bridge seams**: right hand +16 frames, left hand −600, and `other`
demotions rose 122 → 776. **Re-detecting with MINT-guided crops**: 8% recovery — a detector capability
limit, not a search limit. **Offset-correcting minority-source rows**: the spread (18.2 px) exceeds
the offset (21.4 px), so no constant offset exists to correct.

---

## 6. Running it

```bash
OURS_DIR=out/ours \
MINT_DIR=out/mint \
VIDEO_DIR=out/video \
BASE_DIR=out/baseline \
OUT_DIR=out/cplus \
./build_cplus.sh
```

Stages can also be run individually; each writes a JSON report and each keeps its input values so
the pass is reversible and auditable:

```bash
python rescue_handedness.py --npz clip.npz --out 1.npz --head head.npz --imu imu.csv
python temporal_smooth.py   --npz 1.npz   --out 2.npz --h 6 --order 3
python mint_bridge2.py --ours 2.npz --mint mint/clip.npz --out 3.npz --max-gap 20
python rigidify.py     --npz 3.npz --out cplus.npz --h 16 --order 3
python abc_strip.py --video clip.mp4 --delivered baseline.npz --hand21kp cplus.npz \
                    --mint mint/clip.npz --out review.mp4
```

`why_not_bridged.py` replays every gate for one frame and names the one that fired — the fastest way
to answer "why is there no hand here?":

```bash
python why_not_bridged.py --ours clip_C.npz --mint mint/clip.npz --frame 1246 --window 2
```

### Provenance fields on every output

| field | meaning |
|---|---|
| `hand_src_rescue` | which rescue rule recovered this row (`A_complement`, `B_both`, `B_single`) |
| `smoothed` | temporal smoothing applied; originals in `kp2d_raw` / `kp3d_cam_raw_pre` |
| `bridged`, `source='bridged'` | **inferred, not measured** |
| `bridge_disagree`, `bridge_gap` | predicted quality of a bridged row, for downstream filtering |
| `rigid` | rigid-fit applied; originals in `kp2d_prerigid` / `kp3d_cam_prerigid` |

---

## 7. Results

Against the delivered baseline, over 116 clips:

| | delivered | C+ | |
|---|---|---|---|
| jitter (palm units) | 0.185 | 0.068 | **−63%** |
| bone-length CV | 15.0% | **0.000%** | exact |
| skeleton jump at a source switch | 0.838 | 0.318 | −62% |
| left/right flips | 0.115 | 0.000 | −100% |

Articulation jitter lands at **0.014–0.021 against MINT's 0.0142**, while retaining 85–94% of real
motion where MINT retains 36–60%.

## 8. Limitations to carry into any delivery note

- **Bridged rows are inferred, not measured.** Beyond ~20-frame gaps their *placement* stays
  plausible while the hand may have left the frame entirely.
- **The held-out error figures are a floor, not a guarantee** — they can only be sampled from frames
  the detector already covered, which are easier than the frames actually being filled.
- **Smoothed keypoints are not raw per-frame measurements** and must not be presented as such.
- **Depth is inherited, not re-estimated.** The known ~12% WiLoR depth bias is left exactly as
  delivered, to be handled separately and disclosed.
- **The visual adjudications behind the precision numbers are small samples** (100 rows for source
  classes, 8 per gap band) taken from the three worst clips, so per-class rates are pessimistic
  fleet-wide.

---

## 9. Dropping MINT where it earns nothing (`self_bridge.py`)

### 9.1 The observation that started it

A reviewer watching `_A_vs_Cplus_smoothhead` rejected `Others_towel_making_87` for a hand that
floats free of the arm, and accepted `Hospitality_pouring_pink_yogurt_into_bowls_73` — guessing that
the difference was how much of each clip came from MINT. The bridge statistics confirm it exactly:

| clip | bridged rows | in gaps >20 frames | longest run |
|---|---|---|---|
| `Others_towel_making_87` | 869 / 3547 (24.5%) | 390 (11.0%) | 66 frames |
| `Hospitality_pouring_pink_yogurt_into_bowls_73` | 11 / 3644 (0.3%) | 0 | ≤10 frames |

The rejected frame is `bridged=True` and is the *only* row drawn in it — the detector saw nothing
there, so everything visible was MINT-carried. Long bridges do not jitter; they **slide**, because
past roughly 20 frames the anchors stop constraining MINT's slow bias.

Across the 106-clip set the median clip is 14.2% bridged / 5.6% long-bridged, and 16 clips have
**zero** long bridges. So "is this clip safe" is answerable from the bridge json alone, without
watching anything.

### 9.2 Two failed designs, recorded because they were plausible

The MINT bridge was originally justified against **linear interpolation of 21 independent
keypoints** — the weakest possible own-data opponent. Giving it a fair one took three attempts.

**Attempt 1 — blend two damped extrapolations.** Same structure as `mint_bridge2` but carrying our
own local velocity instead of MINT's deltas. Scored **0.273** at gap 10 against lerp's 0.273: no
gain at all. Blending two one-sided extrapolations does not cancel their error, it averages it in.

**Attempt 2 — carry the hand shape across by Procrustes rotation.** Split each hand into a wrist
trajectory and a wrist-centred shape, rotate/scale anchor A's shape onto anchor B's. Scored
**0.363** — materially *worse than doing nothing clever*. Hand articulation is not a rigid rotation
of the point set, so this drags fingers along arcs they never travelled.

**Attempt 3 — per-keypoint cubic Hermite.** The unique cubic hitting both anchor positions *and*
both anchor velocities. Tangents are blended between the chord slope and the measured velocity,
`T = (1-β)·chord + β·V`, so **β=0 reproduces linear interpolation exactly**.

That last property is not cosmetic — it is the test. A first cut used *zero tangents* and claimed
they degenerated to lerp. Zero tangents give **smoothstep**, not lerp, and the self-test caught it
at 9.5 px on a 7-frame gap. After re-parameterising, β=0 matches lerp to 3e-5 px, and the `hb0.0`
column below is identical to the `lerp` column at every gap — a permanent built-in check.

### 9.3 Held-out result (178 clips, median palm widths)

| gap | lerp | **MINT** | hb0.0 | hb0.25 | **hb0.5** | hb0.75 | hb1.0 |
|---|---|---|---|---|---|---|---|
| 2 | 0.072 | **0.054** | 0.072 | 0.065 | 0.063 | 0.068 | 0.078 |
| 5 | 0.159 | **0.104** | 0.159 | 0.147 | 0.140 | 0.143 | 0.155 |
| 10 | 0.273 | **0.153** | 0.273 | 0.253 | 0.244 | 0.247 | 0.261 |
| 15 | 0.364 | **0.183** | 0.364 | 0.341 | 0.331 | 0.336 | 0.355 |
| 30 | 0.576 | **0.246** | 0.576 | 0.554 | 0.552 | 0.577 | 0.621 |

β=0.5 is the best own-data estimator; β=1.0 is worse at every gap, because the local velocity
estimate is noisy and full-strength tangents overshoot.

**MINT still wins per frame — by 1.3× at gap 2, widening to 2.2× at gap 30.** It carries real
information we do not have. That is stated plainly because it decides where this may be used.

### 9.4 Why it is still worth having

Clip-level damage is **per-frame error × rows filled**, and the second factor varies by two orders
of magnitude. On `home_chopping_onion_87` — one gap, 7 frames, 7 rows of 3646:

| | C+ with MINT | no MINT |
|---|---|---|
| rows | 3646 | 3646 |
| tracks | 2 | 2 |
| articulation jitter | 0.0057 | 0.0057 |
| global jitter | 0.0072 | 0.0072 |

The two outputs are identical on 3639 rows (median 0.000 px, p99 0.014 px) and differ only on the 7
bridged rows, by 8.6 px median = **0.062 palm widths**. MINT — a 1.139B-parameter model and a GPU
box — was being run for that.

Fleet-wide, a no-MINT fill capped at 15 frames recovers **72.6% of MINT's short-gap rows but only
36.4% overall**; the missing 20,186 rows are all long bridges, which are the ones that slide. The
anchor-disagreement gate rejects 8.9% of gaps outright — where neither side's velocity predicts
what happened, the rows stay empty rather than being invented.

**Use it where the bridge json shows no long gaps. It is not a drop-in replacement at gap 120.**

### 9.5 Files

| file | role |
|---|---|
| `self_bridge.py` | own-motion gap fill, no external model. `--max-gap 15 --beta 0.5` |
| `eval_self_bridge.py` | held-out harness; attempts 1 and 2, kept as the negative result |
| `eval_self_bridge2.py` | held-out harness for the Hermite β sweep, incl. the β=0≡lerp check |
| `build_self16.sh` | parallel work-stealing build + render for the zero-long-bridge clips |
