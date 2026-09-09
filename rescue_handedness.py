#!/usr/bin/env python3
"""Non-causal handedness rescue for tracked 21-keypoint hand output.

The upstream pipeline assigns handedness PER TRACK. A detection that lands in no surviving track therefore never
gets a label (`hand = -1`) and is dropped (`kept = False`) -- even though it was detected by both
models and fused successfully. On one measured clip that is 167 rows, and 100% of the hand-frames
the pipeline "loses" are of this kind: nothing is missing, it is discarded.

This pass looks BOTH WAYS in time and rescues those rows, but only where the label is forced rather
than guessed. Two rules, in order:

  A  complement    The frame already shows a confidently labelled hand, the candidate sits on the
                   correct side of it, and the opposite label is free. A person has two hands, so
                   the label is determined, not inferred. Highest confidence.

  B  temporal      No anchor on this frame. Take the nearest confident detection before AND after
                   that is spatially compatible with the candidate. If both exist they must AGREE.
                   A single-sided anchor is only accepted inside a much tighter window.

Anything neither rule settles is left dropped. The point is to lose fewer frames, not to guess.
Every rescued row is marked in `hand_src_rescue` so it can be audited or reverted.
"""
from __future__ import annotations
import argparse, json
import numpy as np

WRIST = 0
L, R, OTHER, NONE = 0, 1, 2, -1


def palm_of(kp):
    p = float(np.linalg.norm(kp[5] - kp[17]))
    return p if p > 1e-6 else 60.0


def compatible(cand_kp, anchor_kp, dframes, per_frame_px=14.0, palm_mult=1.8):
    """Could these two detections be the same physical hand `dframes` apart?

    Budget grows with the time gap: a hand moves, but not arbitrarily far, and the allowance is
    expressed in palm widths so it is scale-free with respect to how near the hand is.
    """
    d = float(np.linalg.norm(cand_kp[WRIST] - anchor_kp[WRIST]))
    budget = palm_mult * palm_of(cand_kp) + per_frame_px * abs(dframes)
    return d <= budget, d, budget


def gravity_drop(npz, head_npz, imu_csv):
    """Metres the wrist sits BELOW the head, measured along true gravity.

    An earlier internal finding — that position cannot separate feet from hands — was measured on
    image-y and camera-frame Z. Both are confounded by head tilt: look down and a hand's Z grows
    exactly like a foot's. Rotating into the world and projecting onto the IMU gravity vector
    removes that confound, and it is a different feature from the ones that failed.
    Returns NaN where no head pose covers the frame.
    """
    import csv as _csv
    H = np.load(head_npz, allow_pickle=True)
    T = H['T'].astype(np.float64)
    hfi = H['frame_idx'].astype(int) if 'frame_idx' in H.files else np.arange(len(T))
    Tm = {int(f): T[i] for i, f in enumerate(hfi)}
    rows = list(_csv.reader(open(imu_csv)))
    hdr = [h.strip().lower() for h in rows[0]]
    ix, iy, iz = hdr.index('x'), hdr.index('y'), hdr.index('z')
    acc = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows[1:] if len(r) > iz])
    up_cam = acc.mean(0); up_cam /= (np.linalg.norm(up_cam) + 1e-9)
    n = min(len(T), 400)
    up = np.mean([T[f][:3, :3] @ up_cam for f in range(n)], 0)
    up /= (np.linalg.norm(up) + 1e-9)

    fi = npz['frame_idx'].astype(int); k3 = npz['kp3d_cam'].astype(float)
    out = np.full(len(fi), np.nan)
    for i in range(len(fi)):
        M = Tm.get(int(fi[i]))
        if M is None: continue
        world = M[:3, :3] @ k3[i][WRIST] + M[:3, 3]
        out[i] = float((M[:3, 3] - world) @ up)
    return out


def rescue(npz, win_both=20, win_single=8, include_demoted=False, verbose=True,
           require_mp=True, drop=None, max_drop=0.95):
    fi = npz['frame_idx'].astype(int)
    kp2 = npz['kp2d'].astype(float)
    hand = npz['hand'].astype(int).copy()
    kept = npz['kept'].astype(bool).copy()
    n = len(fi)
    src_rescue = np.array([''] * n, dtype='<U12')

    anchors = np.where(kept & np.isin(hand, (L, R)))[0]
    by_frame_anchor = {}
    for i in anchors:
        by_frame_anchor.setdefault(int(fi[i]), []).append(i)
    anchor_frames = np.array(sorted(by_frame_anchor)) if by_frame_anchor else np.array([], int)

    src = npz['source'] if 'source' in npz.files else np.array(['?'] * n)
    cand = list(np.where((hand == NONE))[0])
    if include_demoted:
        # tracks the wearer owns that lost a label to a conflict, not to the bystander gate
        demoted = np.where((hand == OTHER) & kept)[0]
        cand += list(demoted)
    cand.sort(key=lambda i: int(fi[i]))

    stats = dict(candidates=len(cand), rule_a=0, rule_b_both=0, rule_b_single=0,
                 rejected_no_anchor=0, rejected_disagree=0, rejected_far=0, rejected_taken=0,
                 rejected_no_mediapipe=0, rejected_below_head=0)

    def admissible(i):
        """Quality gates a rescue candidate must pass BEFORE any label reasoning.

        A row can be unlabelled for two very different reasons: nobody worked out which hand it is
        (recoverable), or it is not a hand at all (must stay out). The label rules below cannot tell
        those apart, so the filtering has to happen here.
        """
        if require_mp and str(src[i]) != 'fused':
            # Only `fused` has BOTH models agreeing AND a measured depth. A sampled blind
            # review put the not-on-a-hand rate at ~7% for fused, ~32% for wilor-only and ~60% for
            # lifted_2d -- lifted_2d is the WORST class, not a safe one. It is also the reason the
            # gravity gate cannot protect us there: a lifted_2d row has no WiLoR match, so its depth
            # is a borrowed clip median and every depth-derived test on it is meaningless.
            # (This is exactly how a verified foot false-positive survived an earlier, looser guard.)
            stats['rejected_no_mediapipe'] += 1
            return False
        if drop is not None and np.isfinite(drop[i]) and drop[i] > max_drop:
            stats['rejected_below_head'] += 1
            return False
        return True

    def labels_on(frame):
        return {hand[j] for j in by_frame_anchor.get(frame, [])}

    def nearest_anchor(frame, direction, cand_kp, window):
        """Closest confident detection within `window` frames that could be this same hand."""
        best = None
        for df in range(1, window + 1):
            f = frame + direction * df
            for j in by_frame_anchor.get(f, []):
                ok, _, _ = compatible(cand_kp, kp2[j], df)
                if ok:
                    return j, df
            if best is not None:
                break
        return None, None

    for i in cand:
        if not admissible(i):
            continue
        f = int(fi[i]); ck = kp2[i]
        taken = labels_on(f)

        # ---- rule A: the frame already shows one hand; a person has two
        same_frame = by_frame_anchor.get(f, [])
        if len(same_frame) == 1:
            j = same_frame[0]
            other = R if hand[j] == L else L
            if other not in taken:
                # the candidate must lie on the correct side of the labelled hand
                cand_is_right = ck[WRIST, 0] > kp2[j][WRIST, 0]
                if (other == R) == cand_is_right:
                    hand[i] = other; kept[i] = True; src_rescue[i] = 'A_complement'
                    by_frame_anchor.setdefault(f, []).append(i)
                    stats['rule_a'] += 1
                    continue
            else:
                stats['rejected_taken'] += 1

        # ---- rule B: look both ways
        jb, dfb = nearest_anchor(f, -1, ck, win_both)
        ja, dfa = nearest_anchor(f, +1, ck, win_both)
        lab = None; how = None
        if jb is not None and ja is not None:
            if hand[jb] == hand[ja]:
                lab, how = int(hand[jb]), 'B_both'
            else:
                stats['rejected_disagree'] += 1
        elif jb is not None and dfb <= win_single:
            lab, how = int(hand[jb]), 'B_single'
        elif ja is not None and dfa <= win_single:
            lab, how = int(hand[ja]), 'B_single'
        else:
            stats['rejected_no_anchor'] += 1

        if lab is None:
            continue
        if lab in labels_on(f):                     # never duplicate a label on one frame
            stats['rejected_taken'] += 1
            continue
        hand[i] = lab; kept[i] = True; src_rescue[i] = how
        by_frame_anchor.setdefault(f, []).append(i)
        stats['rule_b_both' if how == 'B_both' else 'rule_b_single'] += 1

    return hand, kept, src_rescue, stats


def frame_coverage(fi, hand, kept, T):
    d = kept & np.isin(hand, (L, R))
    Lp = np.zeros(T, bool); Rp = np.zeros(T, bool)
    for i in np.where(d)[0]:
        f = int(fi[i])
        if 0 <= f < T: (Lp if hand[i] == L else Rp)[f] = True
    return Lp, Rp, int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--win-both', type=int, default=20)
    ap.add_argument('--win-single', type=int, default=8)
    ap.add_argument('--head', default=None, help='head_pose_6dof.npz — enables the gravity gate')
    ap.add_argument('--imu', default=None, help='imu_accel.csv — enables the gravity gate')
    ap.add_argument('--max-drop', type=float, default=0.95,
                    help='reject a rescue whose wrist is more than this many metres below the head')
    ap.add_argument('--allow-uncorroborated', action='store_true',
                    help='also rescue non-fused rows (wilor-only ~32%% bad, lifted_2d ~60%% bad; '
                         'not recommended -- lifted_2d also defeats the gravity gate)')
    ap.add_argument('--include-demoted', action='store_true',
                    help='also reconsider wearer tracks demoted to `other` on a label conflict')
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    fi = z['frame_idx'].astype(int)
    T = int(fi.max()) + 1
    h0, k0 = z['hand'].astype(int), z['kept'].astype(bool)
    L0, R0, n0 = frame_coverage(fi, h0, k0, T)

    drop = None
    if a.head and a.imu:
        try:
            drop = gravity_drop(z, a.head, a.imu)
        except Exception as e:
            print(f'[warn] gravity gate unavailable ({type(e).__name__}: {e}); continuing without it')
    hand, kept, src, stats = rescue(z, a.win_both, a.win_single, a.include_demoted,
                                    require_mp=not a.allow_uncorroborated,
                                    drop=drop, max_drop=a.max_drop)
    L1, R1, n1 = frame_coverage(fi, hand, kept, T)

    out = {k: z[k] for k in z.files}
    out['hand'] = hand.astype(np.int8)
    out['kept'] = kept
    out['hand_src_rescue'] = src
    np.savez_compressed(a.out, **out)

    both0 = int((L0 & R0).sum()); both1 = int((L1 & R1).sum())
    none0 = int((~L0 & ~R0).sum()); none1 = int((~L1 & ~R1).sum())
    print(json.dumps(stats, indent=2))
    print(f"\n{'':22s} {'before':>8} {'after':>8} {'delta':>8}")
    print(f"{'drawn detections':22s} {n0:8d} {n1:8d} {n1-n0:+8d}")
    print(f"{'frames with LEFT':22s} {int(L0.sum()):8d} {int(L1.sum()):8d} {int(L1.sum()-L0.sum()):+8d}")
    print(f"{'frames with RIGHT':22s} {int(R0.sum()):8d} {int(R1.sum()):8d} {int(R1.sum()-R0.sum()):+8d}")
    print(f"{'frames with BOTH':22s} {both0:8d} {both1:8d} {both1-both0:+8d}")
    print(f"{'frames with NEITHER':22s} {none0:8d} {none1:8d} {none1-none0:+8d}")
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    main()
