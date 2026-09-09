#!/usr/bin/env python3
"""Fill detection gaps in our pipeline using MINT's MOTION, anchored on our own detections.

Measured basis for this:
  * MINT finds a hand on 16.5% of frames where our pipeline finds nothing, and misses almost
    nothing we find (1196 frames vs 8, over four clips).
  * MINT's ABSOLUTE placement is unusable -- 42-64 px off, wandering +-40 px over ~1 s.
  * MINT's FRAME-TO-FRAME motion is excellent -- 6.9 px between adjacent frames, better than our
    own detector's per-frame step (12-17 px).

So the bias is slow and shared between neighbouring frames. Anchor on OUR detection at each end of a
gap, propagate MINT's deltas inward from both sides, and blend by position: the shared bias cancels
at both ends and is linearly interpolated between. Held-out error on a 5-frame gap is ~12.8 px,
inside our own detector's noise floor, and 24-30% better than linear interpolation.

Only interior gaps are bridged. A gap with an anchor on one side only cannot cancel the bias, and is
left alone. Depth is interpolated from OUR anchors rather than taken from MINT, whose metric scale
runs ~47% deep.

Bridged rows are marked `bridged=True`. They are INFERRED, not measured, and must be disclosed.
"""
from __future__ import annotations
import argparse, json
import numpy as np

L, R = 0, 1


def per_hand(fi, k2, k3, mask, hand):
    """-> {label: {frame: (kp2d, kp3d)}}"""
    out = {L: {}, R: {}}
    for i in np.where(mask)[0]:
        h = int(hand[i])
        if h in (L, R):
            out[h][int(fi[i])] = (k2[i], k3[i] if k3 is not None else None)
    return out


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
    ap.add_argument('--ours', required=True, help='our C-model npz')
    ap.add_argument('--mint', required=True, help='MINT keypoint npz (mint_kp.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-gap', type=int, default=10,
                    help='longest gap to bridge; beyond this the anchors stop constraining the bias')
    a = ap.parse_args()

    z, fi, k2, k3, mask, hand = load_ours(a.ours)
    m = np.load(a.mint, allow_pickle=True)
    mfi = m['frame_idx'].astype(int); mk2 = m['kp2d'].astype(float)
    mhand = (m['hand'] if 'hand' in m.files else m['is_right']).astype(int)
    M = {L: {}, R: {}}
    for i in range(len(mfi)):
        h = int(mhand[i])
        if h in (L, R): M[h][int(mfi[i])] = mk2[i]

    ours = per_hand(fi, k2, k3, mask, hand)
    K = np.asarray(z['K'], float)
    new2d, new3d, newf, newh = [], [], [], []
    stats = dict(bridged=0, skipped_no_mint=0, skipped_too_long=0, gaps_seen=0)

    for s in (L, R):
        fr = sorted(ours[s])
        for i in range(len(fr) - 1):
            a0, b0 = fr[i], fr[i + 1]
            g = b0 - a0 - 1
            if g <= 0: continue
            stats['gaps_seen'] += 1
            if g > a.max_gap: stats['skipped_too_long'] += 1; continue
            span = list(range(a0, b0 + 1))
            if not all(t in M[s] for t in span): stats['skipped_no_mint'] += 1; continue
            oa, o3a = ours[s][a0]; ob, o3b = ours[s][b0]
            za = float(np.median(o3a[:, 2])) if o3a is not None else None
            zb = float(np.median(o3b[:, 2])) if o3b is not None else None
            for j in range(1, g + 1):
                t = a0 + j; w = j / (g + 1)
                fwd = oa + (M[s][t] - M[s][a0])          # carry our anchor forward on MINT's motion
                bwd = ob + (M[s][t] - M[s][b0])          # and backward from the far anchor
                uv = (1 - w) * fwd + w * bwd             # blend: shared bias cancels at both ends
                new2d.append(uv); newf.append(t); newh.append(s)
                if za is not None and zb is not None:
                    # depth from OUR anchors, not MINT's (its metric scale runs ~47% deep)
                    zt = (1 - w) * za + w * zb
                    d = np.linalg.inv(K) @ np.vstack([uv.T, np.ones(21)])
                    d = (d / d[2]).T
                    new3d.append((d * zt).astype(np.float32))
                else:
                    new3d.append(np.full((21, 3), np.nan, np.float32))
                stats['bridged'] += 1

    n_old = len(fi)
    if new2d:
        out = {k: z[k] for k in z.files}
        add = len(new2d)
        def cat(key, extra):
            return np.concatenate([np.asarray(out[key]), np.asarray(extra)]) if key in out else np.asarray(extra)
        out['frame_idx'] = cat('frame_idx', np.array(newf, np.int32)).astype(np.int32)
        out['kp2d'] = cat('kp2d', np.array(new2d, np.float32)).astype(np.float32)
        out['kp3d_cam'] = cat('kp3d_cam', np.array(new3d, np.float32)).astype(np.float32)
        for key, val in (('hand', np.array(newh, np.int8)), ('is_right', np.array(newh, np.int8))):
            if key in out: out[key] = cat(key, val)
        if 'kept' in out: out['kept'] = np.concatenate([np.asarray(out['kept']), np.ones(add, bool)])
        if 'source' in out: out['source'] = np.concatenate([np.asarray(out['source']), np.array(['bridged'] * add)])
        for key in ('kp3d_cam_raw', 'kp2d_observed', 'kp2d_raw'):
            if key in out:
                out[key] = np.concatenate([np.asarray(out[key]), np.full((add, 21, out[key].shape[-1]), np.nan, np.float32)])
        for key in ('confidence', 'quality', 'track_id', 'smoothed', 'drawn', 'hand_perframe',
                    'hand_src_rescue', 'low_confidence', 'is_wearer', 'depth_measured', 'fuse_residual_px'):
            if key in out:
                arr = np.asarray(out[key])
                if arr.ndim == 1 and len(arr) == n_old:
                    fill = (np.ones(add, arr.dtype) if arr.dtype == bool else np.zeros(add, arr.dtype))
                    if key in ('drawn', 'kept'): fill = np.ones(add, bool)
                    out[key] = np.concatenate([arr, fill])
        out['bridged'] = np.concatenate([np.zeros(n_old, bool), np.ones(add, bool)])
        order = np.argsort(out['frame_idx'], kind='stable')
        for k in out:
            arr = np.asarray(out[k])
            if arr.ndim >= 1 and len(arr) == n_old + add: out[k] = arr[order]
        np.savez_compressed(a.out, **out)
    else:
        np.savez_compressed(a.out, **{k: z[k] for k in z.files},
                            bridged=np.zeros(n_old, bool))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
