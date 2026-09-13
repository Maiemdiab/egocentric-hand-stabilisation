#!/usr/bin/env python3
"""Does our OWN motion fill a gap as well as MINT's does?

Same held-out protocol as eval_bridge2.py -- hide the middle L frames of a run both pipelines hold
continuously, reconstruct, compare in palm widths -- with two strategies added:

  self     damped constant-velocity carry from each anchor (velocity from OUR track), blended by
           position, with the wrist-centred SHAPE carried across by a Procrustes rotation+scale
  selfnv   the same wrist trajectory but with the 21 points lerped, to isolate what the shape
           blend is actually worth

`cur` is the MINT bridge. `lerp` is the weak baseline the MINT bridge was originally justified
against. The question this answers is whether that justification survives a fair own-data opponent.
"""
from __future__ import annotations
import numpy as np, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_bridge import sim_transform, rot2, local_velocity                  # noqa: E402
from eval_bridge2 import load, palm                                          # noqa: E402

TAU, VWIN, ORDER = 6.0, 7, 1


def predict_self(span, oa, ob, shape_blend=True):
    a0, b0 = span[0], span[-1]; g = len(span) - 2
    Ka, Kb = oa[a0], ob[b0]
    fa = np.array(sorted(oa)); wa = np.array([oa[f][0] for f in fa])
    fb = np.array(sorted(ob)); wb = np.array([ob[f][0] for f in fb])
    _, va = local_velocity(fa, wa, a0, VWIN, ORDER)
    _, vb = local_velocity(fb, wb, b0, VWIN, ORDER)
    Sa, Sb = Ka - Ka[0], Kb - Kb[0]
    th, sc = sim_transform(Sa, Sb)
    out = []
    for j in range(1, g + 1):
        w = j / (g + 1); dtf = float(j); dtb = float(j - (g + 1))
        fw = Ka[0] + va * TAU * (1.0 - np.exp(-dtf / TAU))
        bw = Kb[0] + vb * (-TAU) * (1.0 - np.exp(dtb / TAU))
        wrist = (1 - w) * fw + w * bw
        if shape_blend:
            out.append(wrist + (Sa @ rot2(w * th).T) * (sc ** w))
        else:
            out.append((1 - w) * (Ka + (wrist - Ka[0])) + w * (Kb + (wrist - Kb[0])))
    return out


def predict_mint(M, s, span, oa, ob):
    a0, b0 = span[0], span[-1]; g = len(span) - 2
    if a0 not in M[s] or b0 not in M[s]: return None
    out = []
    for j in range(1, g + 1):
        t = span[j]; w = j / (g + 1)
        if t not in M[s]: return None
        fwd = oa[a0] + (M[s][t] - M[s][a0]); bwd = ob[b0] + (M[s][t] - M[s][b0])
        out.append((1 - w) * fwd + w * bwd)
    return out


def main():
    MD = '/Users/maiediab/hand_abc_out/_mintnpz'; NPZ = '/Users/maiediab/hand_abc_out/_npz'
    clips = sorted(os.path.basename(p)[:-4] for p in glob.glob(MD + '/*.npz'))
    clips = [c for c in clips if os.path.exists(f'{NPZ}/{c}_C.npz')]
    LS = [2, 5, 10, 15, 20, 30, 45, 60]
    STR = ['lerp', 'cur', 'selfnv', 'self']
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
                        span = run[st:st + need]
                        a0, b0 = span[0], span[-1]
                        oa = {f: O[s][f] for f in run if f <= a0}
                        ob = {f: O[s][f] for f in run if f >= b0}
                        preds = {'lerp': [(1 - j / (L + 1)) * oa[a0] + (j / (L + 1)) * ob[b0]
                                          for j in range(1, L + 1)],
                                 'cur': predict_mint(M, s, span, oa, ob),
                                 'self': predict_self(span, oa, ob, True),
                                 'selfnv': predict_self(span, oa, ob, False)}
                        for strat, pr in preds.items():
                            if pr is None: continue
                            for j, t in enumerate(span[1:-1]):
                                p = palm(O[s][t])
                                err[(strat, L)].append(
                                    float(np.median(np.linalg.norm(pr[j] - O[s][t], axis=-1))) / p)
        if ci % 30 == 0: print(f'  ...{ci}/{len(clips)}', file=sys.stderr)

    for lbl, fn in (('MEDIAN', np.median), ('p90', lambda v: np.percentile(v, 90))):
        print(f'\nheld-out reconstruction error, {lbl} palm widths   (clips={len(clips)})')
        print(f"{'gap':>5s} " + ''.join(f'{s:>9s}' for s in STR) + f"{'n':>10s}")
        for L in LS:
            row = ''.join(f'{fn(err[(s,L)]):9.3f}' if err[(s, L)] else f'{"-":>9s}' for s in STR)
            print(f'{L:5d} ' + row + f'{len(err[("cur",L)]):10d}')


if __name__ == '__main__':
    main()
