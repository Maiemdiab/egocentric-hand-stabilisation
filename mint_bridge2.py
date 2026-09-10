#!/usr/bin/env python3
"""Bridge detection gaps with MINT's motion, anchored on our own detections. Second generation.

v1 filled only interior gaps of <=10 frames. Measuring 154 clips of held-out reconstruction (hide L
real frames, rebuild them, compare) showed that cap was far too tight, and that two of the three
"improvements" I expected were actually regressions:

  held-out MEDIAN error, palm widths
  gap    lerp   two-sided   window-anchor   one-sided
   10   0.264     0.150         0.212         0.318
   30   0.605     0.262         0.299         0.441
   60   0.935     0.364         0.387         0.555
   90   1.102     0.409         0.426         0.613

  * A single-frame anchor at each end BEATS a window-median anchor. The two-sided blend cancels
    MINT's placement bias exactly AT the anchors, so widening the anchor only mixes in frames where
    that bias has already drifted.
  * One-sided extension is much worse than two-sided at ANY length -- extending 10 frames off one
    anchor is worse than interpolating across a 90-frame hole between two. So it stays off.
  * Error grows very slowly with gap length, which is what makes the bigger cap safe.

Caveat that belongs in the delivery note: the held-out sample can only be drawn from frames our
detector DID find, so it measures gaps under easier conditions than the gaps we actually fill.
Treat these numbers as a floor, not a guarantee, and keep the confidence gate.

MINT's presence signal is real and is used as a gate: it emits no hand on 2.6% of frames and one
hand on 22%, and its confidence separates (median 0.999 where we independently agree, 0.928 where
only MINT sees it). We require a solid median across the span plus a floor on every frame.

Bridged rows are marked `bridged` and `source='bridged'`. They are INFERRED, not measured.
"""
from __future__ import annotations
import argparse, json
import numpy as np

L, R = 0, 1


def load_ours(path):
    z = np.load(path, allow_pickle=True); f = z.files
    fi = z['frame_idx'].astype(int)
    k2 = z['kp2d'].astype(float); k3 = z['kp3d_cam'].astype(float)
    if 'kept' in f and 'hand' in f:
        mask = z['kept'].astype(bool) & np.isin(z['hand'].astype(int), (L, R)); hand = z['hand'].astype(int)
    elif 'drawn' in f:
        mask = z['drawn'].astype(bool); hand = z['hand'].astype(int)
    else:
        mask = np.ones(len(fi), bool); hand = z['is_right'].astype(int)
    return z, fi, k2, k3, mask, hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True); ap.add_argument('--mint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-gap', type=int, default=120,
                    help='longest interior gap to bridge. Held-out median error grows slowly with length: '
                         '0.15 palm at 10 frames, 0.26 at 30, 0.36 at 60, 0.41 at 90. Bridged rows '
                         'sit BETWEEN two of our own confirmed detections of that same hand, so what '
                         'is uncertain is placement, not whether a hand is present.')
    ap.add_argument('--min-conf', type=float, default=0.60,
                    help='floor on MINT confidence for EVERY frame of the span')
    ap.add_argument('--min-median-conf', type=float, default=0.80,
                    help='required median MINT confidence across the span')
    ap.add_argument('--max-anchor-disagree', type=float, default=1.5,
                    help='reject a gap when the our-minus-MINT offset at the two anchors differs by '
                         'more than this many palm widths. If MINT is on the same hand we are, its '
                         'bias drifts slowly and the two offsets agree; if it latched onto a '
                         'different hand mid-gap they diverge. Measured at gap 30 over 70 clips, '
                         'median error by disagreement band: <0.25 -> 0.132, 0.5-1.0 -> 0.359, '
                         '2-4 -> 0.604, >4 -> 0.864 palm widths.')
    a = ap.parse_args()

    z, fi, k2, k3, mask, hand = load_ours(a.ours)
    m = np.load(a.mint, allow_pickle=True)
    mfi = m['frame_idx'].astype(int); mk2 = m['kp2d'].astype(float)
    mhand = (m['hand'] if 'hand' in m.files else m['is_right']).astype(int)
    mconf = m['confidence'].astype(float) if 'confidence' in m.files else np.ones(len(mfi))
    M = {L: {}, R: {}}; CF = {L: {}, R: {}}
    for i in range(len(mfi)):
        h = int(mhand[i])
        if h in (L, R): M[h][int(mfi[i])] = mk2[i]; CF[h][int(mfi[i])] = float(mconf[i])

    ours = {L: {}, R: {}}
    for i in np.where(mask)[0]:
        h = int(hand[i])
        if h in (L, R): ours[h][int(fi[i])] = (k2[i], k3[i])

    K = np.asarray(z['K'], float); Kinv = np.linalg.inv(K)
    new2d, new3d, newf, newh = [], [], [], []
    st = dict(gaps_seen=0, bridged=0, skip_no_mint=0, skip_too_long=0, skip_conf=0,
              skip_anchor_disagree=0)
    band = {10: 0, 20: 0, 30: 0, 45: 0, 99: 0}

    for s in (L, R):
        fr = sorted(ours[s])
        for i in range(len(fr) - 1):
            a0, b0 = fr[i], fr[i + 1]; g = b0 - a0 - 1
            if g <= 0: continue
            st['gaps_seen'] += 1
            if g > a.max_gap: st['skip_too_long'] += 1; continue
            span = list(range(a0, b0 + 1))
            if not all(t in M[s] for t in span): st['skip_no_mint'] += 1; continue
            cs = np.array([CF[s][t] for t in range(a0 + 1, b0)])
            if cs.min() < a.min_conf or np.median(cs) < a.min_median_conf:
                st['skip_conf'] += 1; continue
            oa, o3a = ours[s][a0]; ob, o3b = ours[s][b0]
            # does MINT hold the SAME hand we do, at both ends?
            pm = float(np.linalg.norm(oa[5] - oa[17])) or 60.0
            dis = float(np.linalg.norm(np.median(oa - M[s][a0], 0) - np.median(ob - M[s][b0], 0))) / pm
            if dis > a.max_anchor_disagree: st['skip_anchor_disagree'] += 1; continue
            za = float(np.median(o3a[:, 2])); zb = float(np.median(o3b[:, 2]))
            for j in range(1, g + 1):
                t = a0 + j; w = j / (g + 1)
                fwd = oa + (M[s][t] - M[s][a0])       # carry our anchor forward on MINT's motion
                bwd = ob + (M[s][t] - M[s][b0])       # and backward from the far anchor
                uv = (1 - w) * fwd + w * bwd          # shared bias cancels at both ends
                new2d.append(uv); newf.append(t); newh.append(s)
                if np.isfinite(za) and np.isfinite(zb):
                    zt = (1 - w) * za + w * zb        # depth from OUR anchors; MINT's scale runs deep
                    d = (Kinv @ np.vstack([uv.T, np.ones(21)]))
                    new3d.append(((d / d[2]).T * zt).astype(np.float32))
                else:
                    new3d.append(np.full((21, 3), np.nan, np.float32))
                st['bridged'] += 1
            for k in sorted(band):
                if g <= k: band[k] += g; break

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
        # EVERY other per-row array has to grow too, or downstream stages index past the end.
        # v1 hard-coded a list and silently left `kp3d_cam_wilor_raw` short.
        for key, arr in list(out.items()):
            arr = np.asarray(arr)
            if key in ('frame_idx', 'kp2d', 'kp3d_cam', 'hand', 'is_right', 'hand_perframe'): continue
            if arr.ndim == 0 or len(arr) != n_old: continue
            if arr.dtype == bool:
                fill = np.ones(add, bool) if key in ('kept', 'drawn') else np.zeros(add, bool)
            elif arr.dtype.kind in 'US':
                fill = np.array(['bridged' if key == 'source' else ''] * add, dtype=arr.dtype)
            elif arr.dtype.kind == 'f':
                fill = np.full((add,) + arr.shape[1:], np.nan, arr.dtype)
            else:
                fill = np.full((add,) + arr.shape[1:], -1, arr.dtype)
            out[key] = np.concatenate([arr, fill])
        out['bridged'] = np.concatenate([np.zeros(n_old, bool), np.ones(add, bool)])
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
