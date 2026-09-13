#!/usr/bin/env python3
"""Second attempt at an own-data gap fill, after the first one lost to plain lerp.

What the first round showed (eval_self_bridge.py, 178 clips):
    gap      lerp      cur   selfnv     self
     10     0.259    0.148    0.273    0.363
Blending two damped extrapolations was WORSE than lerp, and carrying the shape across by a
Procrustes rotation was worse again. Two reasons, both structural:
  * blending two one-sided extrapolations does not reproduce the anchors' velocities -- it just
    adds each side's extrapolation error into the middle;
  * hand articulation is not a rigid rotation of the point set, so rotating anchor A's shape onto
    anchor B's moves fingers along arcs they never travelled.

A cubic Hermite fixes the first properly: it is the unique cubic that hits BOTH anchor positions
AND both anchor velocities, so it degenerates to exactly lerp when alpha=0 and can only help from
there. Run it per-keypoint, so each point follows its own smooth doubly-anchored path and shape is
preserved implicitly rather than imposed.

alpha scales the tangents; noisy velocity estimates argue for alpha<1.
"""
from __future__ import annotations
import numpy as np, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bridge2 import load, palm                                          # noqa: E402

VWIN = 7


def kp_velocity(fdict, at, win):
    """Per-keypoint velocity (21,2) in px/frame, from a line fit over the nearest `win` frames."""
    fs = np.array(sorted(fdict))
    sel = fs[np.argsort(np.abs(fs - at))[:win]]
    if len(sel) < 2: return np.zeros((21, 2))
    x = (sel - at).astype(float)
    Y = np.stack([fdict[f] for f in sel])                # (n,21,2)
    xm = x.mean(); den = float(((x - xm) ** 2).sum())
    if den < 1e-9: return np.zeros((21, 2))
    return ((x - xm)[:, None, None] * (Y - Y.mean(0))).sum(0) / den


def hermite(P0, P1, V0, V1, g, beta):
    """Cubic Hermite across `g` interior frames.

    Tangents are blended between the CHORD slope and the measured velocity:
        T = (1-beta)*chord + beta*V
    so beta=0 reproduces linear interpolation EXACTLY (a Hermite whose tangents are both the chord
    is the straight line) and beta=1 is the full measured-velocity Hermite. Zero tangents would be
    smoothstep, not lerp -- an easy and wrong thing to assume, and a self-test catches it.
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


def predict_mint(M, s, span, oa, ob):
    a0, b0 = span[0], span[-1]; g = len(span) - 2
    if a0 not in M[s] or b0 not in M[s]: return None
    out = []
    for j in range(1, g + 1):
        t = span[j]; w = j / (g + 1)
        if t not in M[s]: return None
        out.append((1-w)*(oa[a0] + (M[s][t]-M[s][a0])) + w*(ob[b0] + (M[s][t]-M[s][b0])))
    return out


def main():
    MD = '/Users/maiediab/hand_abc_out/_mintnpz'; NPZ = '/Users/maiediab/hand_abc_out/_npz'
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    clips = sorted(os.path.basename(p)[:-4] for p in glob.glob(MD + '/*.npz'))
    clips = [c for c in clips if os.path.exists(f'{NPZ}/{c}_C.npz')][:LIMIT]
    LS = [2, 5, 10, 15, 20, 30]
    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    STR = ['lerp', 'cur'] + [f'hb{a}' for a in ALPHAS]
    err = {(s, L): [] for s in STR for L in LS}
    for ci, clip in enumerate(clips):
        try: O, M, C = load(clip)
        except Exception: continue
        for s in (0, 1):
            fr = sorted(O[s])
            if len(fr) < 30: continue
            runs, cur = [], [fr[0]]
            for a, b in zip(fr, fr[1:]):
                if b == a + 1 and a in M[s] and b in M[s]: cur.append(b)
                else:
                    if len(cur) > 3: runs.append(cur)
                    cur = [b]
            if len(cur) > 3: runs.append(cur)
            for L in LS:
                need = L + 2
                for run in runs:
                    if len(run) < need: continue
                    for st in range(0, len(run) - need + 1, max(need, 25)):
                        span = run[st:st+need]; a0, b0 = span[0], span[-1]
                        oa = {f: O[s][f] for f in run if f <= a0}
                        ob = {f: O[s][f] for f in run if f >= b0}
                        P0, P1 = oa[a0], ob[b0]
                        V0 = kp_velocity(oa, a0, VWIN); V1 = kp_velocity(ob, b0, VWIN)
                        preds = {'lerp': [(1-j/(L+1))*P0 + (j/(L+1))*P1 for j in range(1, L+1)],
                                 'cur': predict_mint(M, s, span, oa, ob)}
                        for al in ALPHAS:
                            preds[f'hb{al}'] = hermite(P0, P1, V0, V1, L, al)
                        for strat, pr in preds.items():
                            if pr is None: continue
                            for j, t in enumerate(span[1:-1]):
                                p = palm(O[s][t])
                                err[(strat, L)].append(
                                    float(np.median(np.linalg.norm(pr[j]-O[s][t], axis=-1)))/p)
        if ci % 20 == 0: print(f'  ...{ci}/{len(clips)}', file=sys.stderr)

    for lbl, fn in (('MEDIAN', np.median), ('p90', lambda v: np.percentile(v, 90))):
        print(f'\nheld-out error, {lbl} palm widths   (clips={len(clips)})')
        print(f"{'gap':>5s} " + ''.join(f'{s:>9s}' for s in STR) + f"{'n':>10s}")
        for L in LS:
            print(f'{L:5d} ' + ''.join(f'{fn(err[(s,L)]):9.3f}' if err[(s,L)] else f'{"-":>9s}'
                                       for s in STR) + f'{len(err[("cur",L)]):10d}')


if __name__ == '__main__':
    main()
