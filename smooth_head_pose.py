#!/usr/bin/env python3
"""Zero-phase smoothing of a 6-DoF head trajectory.

Measured motivation. The delivery visualisation draws the wrist in the WORLD frame, so what a
reviewer sees is `world_p = R_head @ cam_p + t_head` -- head-pose noise lands in the trajectory even
when the hand keypoints are perfect. Freezing the hand in the camera frame and moving only the head
reproduces nearly all of the observed shake:

    clip           world shake   head-only   hand-only   head's share
    episode_045       4.55 mm      2.91         2.72          40%
    episode_0011      5.51 mm      4.57         1.76          68%
    episode_008       8.00 mm      6.36         2.52          68%

So on most clips the majority of the visible vibration is the head trajectory, not the hands. Every
hand-side fix in this repo was working on the smaller half of the problem.

Rotation is the dominant channel, not translation: DROID's frame-to-frame rotation noise is ~0.6-1.1
deg, and at a ~0.6 m arm that is ~6-11 mm of wrist displacement -- which is the size of the effect.

Two things are deliberately not done naively:
  * rotations are smoothed as MATRICES and re-projected to SO(3) by SVD; smoothing Euler angles or
    quaternion components directly tears at the wrap points and can flip the head backwards;
  * the filter is symmetric and evaluated at the sample, so it adds no lag -- the head must not
    trail the video.
"""
from __future__ import annotations
import argparse, json
import numpy as np


def smooth_se3(T, win, order=2):
    """Local weighted polynomial fit, evaluated at each sample. Zero phase by construction."""
    n = len(T)
    h = max(win, 1)
    t = np.arange(n, dtype=float)
    out = np.repeat(np.eye(4)[None], n, 0)
    trans = T[:, :3, 3]
    R = T[:, :3, :3].reshape(n, 9)
    sm_t = np.empty_like(trans); sm_R = np.empty_like(R)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        tt = t[lo:hi] - t[i]
        w = (1.0 - np.abs(tt / (h + 1e-9)) ** 3) ** 3          # tricube
        V = np.vander(tt, order + 1)
        W = np.sqrt(np.maximum(w, 1e-12))[:, None]
        for arr, dst in ((trans, sm_t), (R, sm_R)):
            coef, *_ = np.linalg.lstsq(V * W, arr[lo:hi] * W, rcond=None)
            dst[i] = coef[-1]
    for i in range(n):
        U, _, Vt = np.linalg.svd(sm_R[i].reshape(3, 3))
        d = np.sign(np.linalg.det(U @ Vt))
        out[i, :3, :3] = U @ np.diag([1, 1, d]) @ Vt          # nearest true rotation
    out[:, :3, 3] = sm_t
    return out


def metrics(T):
    t = T[:, :3, 3]
    lin_acc = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=-1) * 1000.0
    R = T[:, :3, :3]
    dR = np.einsum('nij,nkj->nik', R[1:], R[:-1])
    # first difference = angular SPEED (real head turning). Jitter is the change in that speed,
    # i.e. the second difference -- confusing the two makes a filter that eats real motion look
    # like a filter that removes noise.
    ang_speed = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    ang_jitter = np.abs(np.diff(ang_speed))
    step = np.linalg.norm(np.diff(t, axis=0), axis=-1) * 1000.0
    return lin_acc, ang_jitter, step, ang_speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--win', type=int, default=6, help='half-window in frames')
    ap.add_argument('--order', type=int, default=2)
    a = ap.parse_args()

    z = np.load(a.head, allow_pickle=True)
    T = z['T'].astype(np.float64)
    T2 = smooth_se3(T, a.win, a.order)
    l0, a0, s0, v0 = metrics(T); l1, a1, s1, v1 = metrics(T2)

    out = {k: z[k] for k in z.files}
    out['T'] = T2.astype(np.float32)
    out['translation'] = T2[:, :3, 3].astype(np.float32)
    out['T_unsmoothed'] = T.astype(np.float32)
    out['head_smooth_win'] = np.int32(a.win)
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(
        frames=len(T), win=a.win,
        lin_jitter_mm=[round(float(np.median(l0)), 2), round(float(np.median(l1)), 2)],
        rot_jitter_deg=[round(float(np.median(a0)), 4), round(float(np.median(a1)), 4)],
        rot_speed_deg_per_frame=[round(float(np.median(v0)), 4), round(float(np.median(v1)), 4)],
        rot_speed_p90_kept_pct=round(100 * float(np.percentile(v1, 90) / max(np.percentile(v0, 90), 1e-9)), 1),
        # motion retention: a filter that flattens real head turns is not a fix
        step_p50_kept_pct=round(100 * float(np.median(s1) / max(np.median(s0), 1e-9)), 1),
        step_p90_kept_pct=round(100 * float(np.percentile(s1, 90) / max(np.percentile(s0, 90), 1e-9)), 1),
        step_p99_kept_pct=round(100 * float(np.percentile(s1, 99) / max(np.percentile(s0, 99), 1e-9)), 1),
        ), indent=2))


if __name__ == '__main__':
    main()
