#!/usr/bin/env python3
"""Held-out measurement of bridging strategies.

Take runs where OUR pipeline and MINT both hold the same hand continuously. Hide the middle L
frames of ours, reconstruct them from the surviving anchors plus MINT, and compare against the
values we hid. Error is in palm widths, so it is scale-free with respect to hand distance.

Strategies
  lerp        straight line between the anchors, MINT unused -- the honest baseline
  cur         what mint_bridge.py does now: single-frame anchor at each end
  win{k}      anchor offset = median over k frames at each end, so one bad anchor cannot tilt it
  fwd{k}      ONE-SIDED: only the left anchor window, carried forward on MINT's motion
"""
from __future__ import annotations
import numpy as np, glob, os, sys, collections

NPZ = './out/_npz'; MD = './out/_mintnpz'


def load(clip):
    o = np.load(f'{NPZ}/{clip}_C.npz', allow_pickle=True)
    m = np.load(f'{MD}/{clip}.npz', allow_pickle=True)
    keep = o['kept'].astype(bool) & np.isin(o['hand'].astype(int), (0, 1))
    ofi, oh, ok2 = o['frame_idx'].astype(int)[keep], o['hand'].astype(int)[keep], o['kp2d'].astype(float)[keep]
    mfi, mh, mk2 = m['frame_idx'].astype(int), m['hand'].astype(int), m['kp2d'].astype(float)
    mc = m['confidence'].astype(float)
    O = {s: {} for s in (0, 1)}; M = {s: {} for s in (0, 1)}; C = {s: {} for s in (0, 1)}
    for f, h, k in zip(ofi, oh, ok2): O[int(h)][int(f)] = k
    for f, h, k, c in zip(mfi, mh, mk2, mc):
        if int(h) in (0, 1): M[int(h)][int(f)] = k; C[int(h)][int(f)] = float(c)
    return O, M, C


def palm(k): 
    p = float(np.linalg.norm(k[5] - k[17])); return p if p > 1e-6 else 60.0


def predict(strat, M, s, span, oa, ob, k):
    """span = [a0, t..., b0]; oa/ob = our kp at the anchors; k = anchor-window size."""
    a0, b0 = span[0], span[-1]; g = len(span) - 2
    out = []
    # offset between us and MINT, averaged over an anchor window -> robust to one bad frame
    def off(anchor, direction, ours_at):
        ds = []
        for j in range(k):
            f = anchor + direction * j
            if f in M[s] and f in ours_at: ds.append(ours_at[f] - M[s][f])
        return np.median(np.stack(ds), 0) if ds else None
    for j in range(1, g + 1):
        t = span[j]; w = j / (g + 1)
        if strat == 'lerp':
            out.append((1 - w) * oa[a0] + w * ob[b0]); continue
        if t not in M[s]: return None
        if strat == 'cur':
            fwd = oa[a0] + (M[s][t] - M[s][a0]); bwd = ob[b0] + (M[s][t] - M[s][b0])
            out.append((1 - w) * fwd + w * bwd); continue
        if strat.startswith('win'):
            da = off(a0, -1, oa); db = off(b0, +1, ob)
            if da is None or db is None: return None
            out.append(M[s][t] + (1 - w) * da + w * db); continue
        if strat.startswith('fwd'):
            da = off(a0, -1, oa)
            if da is None: return None
            out.append(M[s][t] + da); continue
    return out


def main():
    clips = sorted(os.path.basename(p)[:-4] for p in glob.glob(MD + '/*.npz'))
    clips = [c for c in clips if os.path.exists(f'{NPZ}/{c}_C.npz')]
    LS = [2, 5, 10, 15, 20, 30, 45, 60, 90]
    STR = ['lerp', 'cur', 'win5', 'win9', 'fwd5', 'fwd9']
    err = {(s, L): [] for s in STR for L in LS}
    for ci, clip in enumerate(clips):
        try: O, M, C = load(clip)
        except Exception: continue
        for s in (0, 1):
            fr = sorted(O[s])
            if len(fr) < 30: continue
            # maximal runs where BOTH have this hand on every frame
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
                        for strat in STR:
                            k = int(strat[3:]) if strat[-1].isdigit() and strat[:3] in ('win', 'fwd') else 1
                            pr = predict(strat, M, s, span, oa, ob, k)
                            if pr is None: continue
                            for j, t in enumerate(span[1:-1]):
                                p = palm(O[s][t])
                                err[(strat, L)].append(float(np.median(np.linalg.norm(pr[j] - O[s][t], axis=-1))) / p)
        if ci % 30 == 0: print(f'  ...{ci}/{len(clips)}', file=sys.stderr)

    print(f'\nheld-out reconstruction error, MEDIAN palm widths   (clips={len(clips)})')
    print(f"{'gap':>5s} " + ''.join(f'{s:>9s}' for s in STR) + f"{'n':>9s}")
    for L in LS:
        n = len(err[('cur', L)])
        row = ''.join(f'{np.median(err[(s,L)]):9.3f}' if err[(s, L)] else f'{"-":>9s}' for s in STR)
        print(f'{L:5d} ' + row + f'{n:9d}')
    print(f'\nsame, 90th percentile (the bad cases that get seen)')
    print(f"{'gap':>5s} " + ''.join(f'{s:>9s}' for s in STR))
    for L in LS:
        row = ''.join(f'{np.percentile(err[(s,L)],90):9.3f}' if err[(s, L)] else f'{"-":>9s}' for s in STR)
        print(f'{L:5d} ' + row)


if __name__ == '__main__':
    main()
