#!/usr/bin/env python3
"""Fill detection gaps from OUR OWN detections only. No MINT, no external model.

Measured, on 178 clips, held-out (hide the middle L frames of a run, reconstruct, compare):

    gap      lerp   MINT   herm0.25  herm0.5  herm0.75  herm1.0     <- median palm widths
     10     0.273  0.153      0.258    0.249     0.251    0.261

Two rounds of design went into that `herm0.5` column, and the first round LOST:

  * blending two damped one-sided extrapolations scored 0.273 -> it just adds each side's
    extrapolation error into the middle instead of cancelling it;
  * carrying the wrist-centred shape across by a Procrustes rotation scored 0.363, worse than
    doing nothing clever at all -- hand articulation is not a rigid rotation of the point set, so
    rotating anchor A's shape onto anchor B's drags fingers along arcs they never travelled.

A cubic Hermite is the unique cubic hitting BOTH anchor positions and BOTH anchor velocities. It
degenerates to exactly lerp at alpha=0, so it can only improve on it, and run per-keypoint each
point follows its own doubly-anchored path -- shape is preserved implicitly rather than imposed.
beta=0.5 measured best; beta=1.0 is worse at every gap, because the local velocity estimate is
noisy and full-strength tangents overshoot.

Honest limitation, stated because it decides where this may be used: MINT still wins per frame,
by 1.3x at gap 2 widening to 2.2x at gap 30. MINT carries real information we do not have. What
makes this worth having is that the clip-level damage is (per-frame error) x (rows filled): on a
clip whose gaps are few and short the difference is a handful of pixels on a handful of rows and
is invisible, while MINT's long bridges visibly slide. So this is the right fill for SHORT gaps
and `--max-gap` defaults accordingly; it is not a drop-in replacement at gap 120.

Depth is interpolated from our own anchors, then back-projected through K -- same as the MINT path.

Filled rows are flagged `bridged=True` and `bridge_src='self'`. They are INFERRED, not measured.
"""
from __future__ import annotations
import argparse, json
import numpy as np

L, R = 0, 1


def load_ours(path):
    z = np.load(path, allow_pickle=True); f = z.files
    fi = z['frame_idx'].astype(int)
    k2 = z['kp2d'].astype(float)
    k3 = z['kp3d_cam'].astype(float) if 'kp3d_cam' in f else None
    if 'kept' in f and 'hand' in f:
        mask = z['kept'].astype(bool) & np.isin(z['hand'].astype(int), (L, R)); hand = z['hand'].astype(int)
    elif 'drawn' in f:
        mask = z['drawn'].astype(bool); hand = z['hand'].astype(int)
    else:
        mask = np.ones(len(fi), bool); hand = z['is_right'].astype(int)
    return z, fi, k2, k3, mask, hand


def palm_width(kp):
    """Index-MCP to pinky-MCP: the scale everything is reported in."""
    return max(float(np.linalg.norm(kp[5] - kp[17])), 1e-3)


def kp_velocity(fdict, at, win):
    """Per-keypoint velocity (21,2) px/frame, from a line fit over the `win` frames nearest `at`."""
    fs = np.array(sorted(fdict))
    if len(fs) < 2: return np.zeros((21, 2))
    sel = fs[np.argsort(np.abs(fs - at))[:win]]
    if len(sel) < 2: return np.zeros((21, 2))
    x = (sel - at).astype(float)
    Y = np.stack([fdict[f] for f in sel])
    xm = x.mean(); den = float(((x - xm) ** 2).sum())
    if den < 1e-9: return np.zeros((21, 2))
    return ((x - xm)[:, None, None] * (Y - Y.mean(0))).sum(0) / den


def hermite(P0, P1, V0, V1, g, beta):
    """Cubic Hermite across a gap of `g` interior frames.

    Tangents blend between the CHORD slope and the measured velocity:
        T = (1-beta)*chord + beta*V
    beta=0 therefore reproduces linear interpolation EXACTLY, so the knob can only improve on lerp
    and the degenerate case is testable. (Zero TANGENTS would give smoothstep, not lerp -- an easy
    thing to assume and wrong by ~9 px on a 7-frame gap; a self-test catches it.)
    """
    Ls = float(g + 1)
    chord = (P1 - P0) / Ls
    T0 = (1 - beta) * chord + beta * V0
    T1 = (1 - beta) * chord + beta * V1
    out = []
    for j in range(1, g + 1):
        t = j / Ls
        h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2;    h11 = t**3 - t**2
        out.append(h00*P0 + h10*(T0*Ls) + h01*P1 + h11*(T1*Ls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-gap', type=int, default=15,
                    help='longest gap to fill. Beyond ~15 frames our own velocity has no information '
                         'about what the hand did in the middle and the fill is a guess.')
    ap.add_argument('--vel-win', type=int, default=7, help='frames of context used to estimate velocity')
    ap.add_argument('--beta', type=float, default=0.5,
                    help='Hermite tangent blend: 0 = plain linear interpolation, 1 = full measured '
                         'velocity. Above ~0.5 the noisy velocity estimate starts to overshoot.')
    ap.add_argument('--max-anchor-disagree', type=float, default=2.5,
                    help='skip the gap if the anchors disagree, after each is carried on its own '
                         'velocity to the far end, by more than this many palm widths')
    a = ap.parse_args()

    z, fi, k2, k3, mask, hand = load_ours(a.ours)
    K = np.asarray(z['K'], float)

    obs = {L: {}, R: {}}
    for i in np.where(mask)[0]:
        h = int(hand[i])
        if h in (L, R):
            obs[h][int(fi[i])] = (k2[i], None if k3 is None else k3[i])

    new2d, new3d, newf, newh, newdis, newgap = [], [], [], [], [], []
    st = dict(gaps_seen=0, bridged=0, skip_too_long=0, skip_disagree=0)
    band = {10: 0, 15: 0, 20: 0, 30: 0, 45: 0, 99: 0}

    for s in (L, R):
        fr = np.array(sorted(obs[s]))
        if len(fr) < 2: continue
        for i in range(len(fr) - 1):
            a0, b0 = int(fr[i]), int(fr[i + 1])
            g = b0 - a0 - 1
            if g <= 0: continue
            st['gaps_seen'] += 1
            if g > a.max_gap: st['skip_too_long'] += 1; continue

            Ka, _ = obs[s][a0]; Kb, _ = obs[s][b0]
            pw = 0.5 * (palm_width(Ka) + palm_width(Kb))

            # velocity context: our own frames on each side, excluding the gap itself
            pre = {f: obs[s][f][0] for f in fr[max(0, i - a.vel_win + 1):i + 1]}
            post = {f: obs[s][f][0] for f in fr[i + 1:i + 1 + a.vel_win]}
            V0 = kp_velocity(pre, a0, a.vel_win)
            V1 = kp_velocity(post, b0, a.vel_win)

            # Gate: carry each anchor to the OTHER end on its own velocity and see how far apart
            # they land. Two anchors that disagree wildly mean the hand did something in the gap
            # that neither side's velocity predicts, and no interpolation is going to recover it.
            span = float(g + 1)
            fwd_end = Ka[0] + V0[0] * span
            bwd_end = Kb[0] - V1[0] * span
            worst = 0.5 * (float(np.linalg.norm(fwd_end - Kb[0])) +
                           float(np.linalg.norm(bwd_end - Ka[0]))) / pw
            if worst > a.max_anchor_disagree:
                st['skip_disagree'] += 1; continue

            preds = hermite(Ka, Kb, V0, V1, g, a.beta)

            o3a = obs[s][a0][1]; o3b = obs[s][b0][1]
            za = float(np.median(o3a[:, 2])) if o3a is not None and np.isfinite(o3a[:, 2]).any() else None
            zb = float(np.median(o3b[:, 2])) if o3b is not None and np.isfinite(o3b[:, 2]).any() else None
            Kinv = np.linalg.inv(K)
            for j, uv in enumerate(preds, start=1):
                t = a0 + j; w = j / (g + 1)
                new2d.append(uv); newf.append(t); newh.append(s)
                newdis.append(worst); newgap.append(g)
                if za is not None and zb is not None:
                    zt = (1 - w) * za + w * zb
                    d = Kinv @ np.vstack([uv.T, np.ones(21)])
                    d = (d / d[2]).T
                    new3d.append((d * zt).astype(np.float32))
                else:
                    new3d.append(np.full((21, 3), np.nan, np.float32))
                st['bridged'] += 1
                for k in sorted(band):
                    if g <= k: band[k] += 1; break

    n_old = len(fi); add = len(new2d)
    out = {k: z[k] for k in z.files}
    if add:
        def cat(key, extra):
            return np.concatenate([np.asarray(out[key]), np.asarray(extra)]) if key in out else np.asarray(extra)
        out['frame_idx'] = cat('frame_idx', np.array(newf, np.int32)).astype(np.int32)
        out['kp2d'] = cat('kp2d', np.array(new2d, np.float32)).astype(np.float32)
        out['kp3d_cam'] = cat('kp3d_cam', np.array(new3d, np.float32)).astype(np.float32)
        for key in ('hand', 'is_right', 'hand_perframe'):
            if key in out: out[key] = cat(key, np.array(newh, np.int8))
        # grow EVERY other per-row array, generically (v1 of the MINT bridge left one short)
        for key, arr in list(out.items()):
            arr = np.asarray(arr)
            if key in ('frame_idx', 'kp2d', 'kp3d_cam', 'hand', 'is_right', 'hand_perframe'): continue
            if arr.ndim == 0 or len(arr) != n_old: continue
            if arr.dtype == bool:
                fill = np.ones(add, bool) if key in ('kept', 'drawn') else np.zeros(add, bool)
            elif arr.dtype.kind in 'US':
                fill = np.array(['self_bridged' if key == 'source' else ''] * add, dtype=arr.dtype)
            elif arr.dtype.kind == 'f':
                fill = np.full((add,) + arr.shape[1:], np.nan, arr.dtype)
            else:
                fill = np.full((add,) + arr.shape[1:], -1, arr.dtype)
            out[key] = np.concatenate([arr, fill])
        out['bridged'] = np.concatenate([np.zeros(n_old, bool), np.ones(add, bool)])
        out['bridge_src'] = np.array([''] * n_old + ['self'] * add)
        out['bridge_disagree'] = np.concatenate([np.full(n_old, np.nan, np.float32),
                                                 np.array(newdis, np.float32)])
        out['bridge_gap'] = np.concatenate([np.zeros(n_old, np.int32), np.array(newgap, np.int32)])
        order = np.argsort(out['frame_idx'], kind='stable')
        for k in list(out):
            arr = np.asarray(out[k])
            if arr.ndim >= 1 and len(arr) == n_old + add: out[k] = arr[order]
    else:
        out['bridged'] = np.zeros(n_old, bool)
    np.savez_compressed(a.out, **out)
    st['by_gap_len'] = {f'<={k}': v for k, v in band.items() if v}
    print(json.dumps(st, indent=2))


if __name__ == '__main__':
    main()
