#!/usr/bin/env python3
"""Light side-by-side review strip: baseline vs candidate, time-locked.

Built for triage across hundreds of clips, so it deliberately drops any 3D panels --
those cost minutes per clip and the visible problem is the skeleton on the ego view.
One decode pass, cv2 drawing only, downscaled, ~25 MB per 60 s clip.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
import numpy as np
import cv2

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
BLUE = (255, 120, 40)      # left  (BGR)
RED  = (40, 40, 235)       # right
GREY = (150, 150, 150)


def build_tracks(fi, k2, gate=150.0, maxgap=10, min_len=10):
    order = np.argsort(fi, kind='stable'); tk = []
    for gi in order:
        f = int(fi[gi]); xy = k2[gi, 0]; best, bd = None, gate
        for t in tk:
            if 0 < f - t['lf'] <= maxgap:
                d = float(np.hypot(*(t['lxy'] - xy)))
                if d < bd: best, bd = t, d
        if best is None: tk.append(dict(lxy=xy, lf=f, idx=[gi]))
        else: best['lxy'] = xy; best['lf'] = f; best['idx'].append(gi)
    return [np.array(sorted(t['idx'], key=lambda i: int(fi[i]))) for t in tk if len(t['idx']) >= min_len]


def variant_index(path, kind, W=1920):
    """-> {frame: [(kp2d, colour), ...]}, plus a short caption."""
    if path is None or not os.path.exists(path):
        return {}, 'MISSING'
    z = np.load(path, allow_pickle=True); f = z.files
    fi = z['frame_idx']; k2 = z['kp2d'].astype(np.float32)
    if kind == 'delivered':
        drawn = np.ones(len(fi), bool); hand = None
    elif kind == 'fix':
        drawn = z['drawn'].astype(bool) if 'drawn' in f else np.ones(len(fi), bool)
        hand = z['hand'] if 'hand' in f else None
    else:
        drawn = z['kept'].astype(bool) if 'kept' in f else np.ones(len(fi), bool)
        hand = z['hand'] if 'hand' in f else None
        if hand is not None: drawn = drawn & np.isin(hand, (0, 1))
    idx = np.where(drawn)[0]
    if len(idx) == 0: return {}, 'no detections'
    fi2, k22 = fi[idx], k2[idx]
    tracks = build_tracks(fi2, k22)
    # one colour per track: persisted handedness when the file has it, else wrist-x majority
    out = {}
    for tr in tracks:
        if hand is not None:
            v = [int(hand[idx[i]]) for i in tr if 0 <= int(hand[idx[i]]) <= 1]
            lab = int(round(np.mean(v))) if v else int(np.median(k22[tr, 0, 0]) >= W / 2)
        else:
            lab = int(np.median(k22[tr, 0, 0]) >= W / 2)
        col = RED if lab else BLUE
        for i in tr:
            out.setdefault(int(fi2[i]), []).append((k22[i], col))
    return out, f'{len(tracks)} tracks, {len(idx)} drawn'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--delivered', required=True)
    ap.add_argument('--fix', default=None)
    ap.add_argument('--hand21kp', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--c-title', default='C  candidate: rescue + smoothing')
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.max_frames: N = min(N, a.max_frames)
    ow, oh = int(W * a.scale) // 2 * 2, int(H * a.scale) // 2 * 2

    # only render panels whose labels were actually supplied
    panels = [('A  BASELINE (unchanged)', *variant_index(a.delivered, 'delivered', W))]
    if a.fix:
        panels.append(('B  ALTERNATE', *variant_index(a.fix, 'fix', W)))
    if a.hand21kp:
        panels.append((a.c_title, *variant_index(a.hand21kp, 'h21', W)))

    tmp = a.out + '.tmp.mp4'
    NP = len(panels)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow * NP, oh))
    s = a.scale
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(N):
        ok, frame = cap.read()
        if not ok: break
        small = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        cells = []
        for title, index, caption in panels:
            img = small.copy()
            for kp, col in index.get(t, []):
                p = kp * s
                for x, y in ((int(u), int(v)) for u, v in p):
                    cv2.circle(img, (x, y), 2, col, -1, cv2.LINE_AA)
                for i, j in HAND_EDGES:
                    cv2.line(img, (int(p[i, 0]), int(p[i, 1])), (int(p[j, 0]), int(p[j, 1])),
                             col, 1, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (ow, 26), (0, 0, 0), -1)
            cv2.putText(img, f'{title}  [{caption}]', (7, 18), fnt, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
        strip = np.hstack(cells)
        cv2.putText(strip, f'f{t}', (ow * NP - 78, oh - 10), fnt, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        vw.write(strip)
    cap.release(); vw.release()

    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264',
                        '-crf', str(a.crf), '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', a.out], capture_output=True)
    if r.returncode == 0: os.remove(tmp)
    else: os.replace(tmp, a.out)
    print(f'{a.out}  {os.path.getsize(a.out)/1e6:.1f} MB  {ow*NP}x{oh}')


if __name__ == '__main__':
    main()
