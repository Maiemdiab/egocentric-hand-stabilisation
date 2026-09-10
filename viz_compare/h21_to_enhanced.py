#!/usr/bin/env python3
"""Map a hand21kp `*_hand21_keypoints.npz` onto the delivered `_enhanced_keypoints` schema so the
SAME viz_delivery panel renderer can draw it. Nothing is recomputed -- this is a field rename plus
the two extra fields the patched renderer understands (`hand`, `drawn`), so the visual A/B/C
differs only in the labels, never in the drawing code.

  kp3d_cam, kp2d, frame_idx, K, source  -> carried straight across
  hand                                  -> hand21kp's track-constant handedness (2 = 'other'/bystander)
  drawn                                 -> hand21kp's own `kept` filter decision
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-other', action='store_true',
                    help="also draw tracks hand21kp labelled `other` (a bystander's hands)")
    a = ap.parse_args()
    z = np.load(a.src, allow_pickle=True)
    f = z.files
    n = len(z['frame_idx'])

    drawn = z['kept'].astype(bool) if 'kept' in f else np.ones(n, bool)
    hand = z['hand'].astype(np.int8) if 'hand' in f else np.zeros(n, np.int8)
    if not a.keep_other:
        drawn &= (hand == 0) | (hand == 1)          # renderer's colour map only knows left/right

    # renderer needs a left/right value for every row it draws
    hr = np.where(np.isin(hand, (0, 1)), hand, 0).astype(np.int8)

    raw = z['kp3d_cam_wilor_raw'] if 'kp3d_cam_wilor_raw' in f else z['kp3d_cam']
    out = dict(
        kp3d_cam=z['kp3d_cam'].astype(np.float32),
        kp3d_cam_raw=raw.astype(np.float32),
        kp2d=z['kp2d'].astype(np.float32),
        frame_idx=z['frame_idx'].astype(np.int32),
        is_right=hr,
        source=z['source'] if 'source' in f else np.array(['fused'] * n),
        K=np.asarray(z['K'], float),
        hand=hr,
        drawn=drawn,
        n_dropped_dupes=z['n_dropped_dupes'] if 'n_dropped_dupes' in f else np.int32(0),
    )
    np.savez_compressed(a.out, **out)
    u, c = np.unique(hand, return_counts=True)
    print(f"{a.src} -> {a.out}")
    print(f"  rows {n}  drawn {int(drawn.sum())} ({100*drawn.mean():.1f}%)  hand mix {dict(zip(u.tolist(), c.tolist()))}")


if __name__ == '__main__':
    main()
