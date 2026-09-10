#!/usr/bin/env python3
"""Decode a MINT prediction.npz into camera-frame 21-keypoints + 2D projection.

Uses MINT's own decode path so the result is what MINT means, not a reimplementation:
  hand[T,218] -> per-side {transl_cam, orient6d, pose6d, betas}
              -> decode_hand_6d -> run_mano -> joints[:, :21]   (OpenPose-21 order)
The keypoints stay in the CAMERA frame (no world transform), which is what the delivery schema wants.
2D comes from MINT's OWN predicted intrinsics, evaluated at the target image size.
"""
import argparse, sys, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--prediction', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--width', type=int, default=1920)
ap.add_argument('--height', type=int, default=1080)
ap.add_argument('--conf', type=float, default=0.5)
a = ap.parse_args()

sys.path.insert(0, '/opt/mint/repo')
import warnings; warnings.filterwarnings('ignore')
from mint.visualization import mano, geometry
from mint.visualization.render import prediction_to_hands

z = np.load(a.prediction)
pred = {k: z[k] for k in z.files}
T = len(pred['pose_enc'])
hands = prediction_to_hands(pred['hand'])

# MINT's own camera: extrinsics + intrinsics at the requested image size
extr, K = geometry.decode_camera_pose_enc(pred['pose_enc'], a.height, a.width)
K = np.asarray(K, np.float64).reshape(3, 3)

conf = pred.get('hand_confidence')
rows = []
for side, is_right in (('left', False), ('right', True)):
    v = hands[side]
    betas = np.repeat(v['betas'].mean(0, keepdims=True), T, axis=0)   # per-clip shape, as MINT does
    dec = mano.decode_hand_6d(v['transl_cam'], v['orient6d'], v['pose6d'], betas, is_right)
    _, joints = mano.run_mano(dec['trans'], dec['rot'], dec['hand_pose'], dec['betas'], is_right)
    J = np.asarray(joints)[:, :21]                                   # (T,21,3) camera frame, metres
    Zc = np.clip(J[..., 2], 1e-6, None)
    u = K[0, 0] * J[..., 0] / Zc + K[0, 2]
    v2 = K[1, 1] * J[..., 1] / Zc + K[1, 2]
    uv = np.stack([u, v2], -1)
    c = conf[:, 1 if is_right else 0] if conf is not None else np.ones(T)
    for t in range(T):
        if c[t] >= a.conf:
            rows.append((t, int(is_right), J[t], uv[t], float(c[t])))

rows.sort(key=lambda r: (r[0], r[1]))
out = dict(
    frame_idx=np.array([r[0] for r in rows], np.int32),
    is_right=np.array([r[1] for r in rows], np.int8),
    hand=np.array([r[1] for r in rows], np.int8),
    kp3d_cam=np.array([r[2] for r in rows], np.float32),
    kp3d_cam_raw=np.array([r[2] for r in rows], np.float32),
    kp2d=np.array([r[3] for r in rows], np.float32),
    confidence=np.array([r[4] for r in rows], np.float32),
    source=np.array(['fused'] * len(rows)),
    kept=np.ones(len(rows), bool),
    K=K,
)
np.savez_compressed(a.out, **out)
print(json.dumps(dict(frames=T, detections=len(rows),
                      K=[round(float(K[0,0]),2), round(float(K[1,1]),2),
                         round(float(K[0,2]),2), round(float(K[1,2]),2)],
                      depth_median_m=round(float(np.median(out['kp3d_cam'][:,0,2])),3)), indent=2))
