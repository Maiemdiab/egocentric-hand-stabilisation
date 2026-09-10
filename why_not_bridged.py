#!/usr/bin/env python3
"""Explain, frame by frame, why a given frame was or was not bridged.

Every rejection in mint_bridge2 is a specific gate. This replays them for one frame so the answer
is 'gate X rejected it, by this margin' rather than a guess.
"""
from __future__ import annotations
import argparse
import numpy as np

L, R = 0, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True); ap.add_argument('--mint', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--window', type=int, default=0, help='also report +/- this many frames')
    ap.add_argument('--max-gap', type=int, default=120)
    ap.add_argument('--min-conf', type=float, default=0.0)
    ap.add_argument('--min-median-conf', type=float, default=0.0)
    ap.add_argument('--max-mint-hole', type=int, default=20)
    ap.add_argument('--max-hole-frac', type=float, default=0.5)
    ap.add_argument('--max-anchor-disagree', type=float, default=1.5)
    a = ap.parse_args()

    z = np.load(a.ours, allow_pickle=True); f = z.files
    fi = z['frame_idx'].astype(int); k2 = z['kp2d'].astype(float)
    hand = z['hand'].astype(int)
    kept = z['kept'].astype(bool) if 'kept' in f else np.ones(len(fi), bool)
    drawn = kept & np.isin(hand, (L, R))
    m = np.load(a.mint, allow_pickle=True)
    mfi = m['frame_idx'].astype(int); mh = m['hand'].astype(int); mk2 = m['kp2d'].astype(float)
    mc = m['confidence'].astype(float) if 'confidence' in m.files else np.ones(len(mfi))

    O = {L: {}, R: {}}; M = {L: {}, R: {}}; CF = {L: {}, R: {}}
    for i in np.where(drawn)[0]: O[int(hand[i])][int(fi[i])] = k2[i]
    for i in range(len(mfi)):
        h = int(mh[i])
        if h in (L, R): M[h][int(mfi[i])] = mk2[i]; CF[h][int(mfi[i])] = float(mc[i])
    # rows we hold but dropped, so we can say "we HAD it and threw it away"
    DROP = {L: {}, R: {}}
    for i in np.where(~drawn)[0]:
        h = int(hand[i]) if int(hand[i]) in (L, R) else None
        DROP.setdefault(h, {}) if h is None else DROP[h].setdefault(int(fi[i]), []).append(
            (str(z['source'][i]) if 'source' in f else '?', bool(kept[i])))

    for t in range(a.frame - a.window, a.frame + a.window + 1):
        print(f'\n=== frame {t} ===')
        for s, nm in ((L, 'LEFT '), (R, 'RIGHT')):
            if t in O[s]:
                print(f'  {nm}: WE HAVE IT (drawn)'); continue
            inm = t in M[s]
            fr = sorted(O[s])
            if not fr:
                print(f'  {nm}: we have no track at all for this hand; MINT={inm}'); continue
            before = [x for x in fr if x < t]; after = [x for x in fr if x > t]
            if not before or not after:
                side = 'BEFORE the first' if not before else 'AFTER the last'
                print(f'  {nm}: NOT BRIDGEABLE - frame is {side} detection we have '
                      f'(our range {fr[0]}..{fr[-1]}). One-sided extension is off by default '
                      f'(measured worse than two-sided at any length). MINT has it: {inm}')
                continue
            a0, b0 = before[-1], after[0]; g = b0 - a0 - 1
            span = list(range(a0 + 1, b0))
            cov = [x for x in span if x in M[s]]
            print(f'  {nm}: inside gap f{a0+1}-{b0-1}  ({g} frames, anchors f{a0} and f{b0})')
            if g > a.max_gap:
                print(f'         REJECTED: gap {g} > max-gap {a.max_gap}'); continue
            if a0 not in M[s] or b0 not in M[s]:
                print('         REJECTED: MINT has no hand at one of the two anchors, so its '
                      'placement bias cannot be measured or cancelled there'); continue
            miss = [x for x in span if x not in M[s]]
            if miss:
                runs_, cur_ = [], [miss[0]]
                for x in miss[1:]:
                    if x == cur_[-1] + 1: cur_.append(x)
                    else: runs_.append(cur_); cur_ = [x]
                runs_.append(cur_)
                mx = max(len(r) for r in runs_)
                if mx > a.max_mint_hole or mx > a.max_hole_frac * g:
                    print(f'         REJECTED: MINT drops out too; longest dropout {mx} frames '
                          f'(cap {a.max_mint_hole}, and {a.max_hole_frac:.0%} of the {g}-frame gap '
                          f'= {a.max_hole_frac*g:.0f}). Dropout runs: {[len(r) for r in runs_]}'); continue
                print(f'         note: MINT drops out for {len(miss)} frames in runs '
                      f'{[len(r) for r in runs_]} -- interpolated, within cap')
            cs = np.array([CF[s][x] for x in span if x in CF[s]])
            if cs.min() < a.min_conf:
                print(f'         REJECTED: MINT confidence floor {cs.min():.3f} < {a.min_conf}'); continue
            if np.median(cs) < a.min_median_conf:
                print(f'         REJECTED: median MINT confidence {np.median(cs):.3f} < {a.min_median_conf}'); continue
            oa, ob = O[s][a0], O[s][b0]
            pm = float(np.linalg.norm(oa[5] - oa[17])) or 60.0
            dis = float(np.linalg.norm(np.median(oa - M[s][a0], 0) - np.median(ob - M[s][b0], 0))) / pm
            if dis > a.max_anchor_disagree:
                print(f'         REJECTED: anchor disagreement {dis:.2f} > {a.max_anchor_disagree} palm '
                      f'widths -- MINT is not holding the same hand at both ends'); continue
            print(f'         SHOULD BRIDGE (conf med {np.median(cs):.3f}, anchor disagree {dis:.2f})')


if __name__ == '__main__':
    main()
