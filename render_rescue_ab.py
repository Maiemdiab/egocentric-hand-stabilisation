#!/usr/bin/env python3
"""Side-by-side validation render for the non-causal handedness rescue.

LEFT  : input as it stands
RIGHT : input + rescue

Rescued detections are drawn in GREEN and tagged `R*` / `L*` at the wrist, so the label itself can be
checked by eye — the question is not just "did a skeleton appear" but "is it on the correct hand and
called the correct side". Everything already present keeps its normal blue/red.
"""
from __future__ import annotations
import argparse, os, subprocess
import numpy as np
import cv2

E = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
     (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
BLUE, RED, GREEN = (255, 120, 40), (40, 40, 235), (60, 220, 60)
WRIST = 0


def index(path, mark_rescued):
    z = np.load(path, allow_pickle=True)
    fi = z['frame_idx'].astype(int); hand = z['hand'].astype(int)
    kept = z['kept'].astype(bool); k2 = z['kp2d'].astype(float)
    rs = z['hand_src_rescue'] if 'hand_src_rescue' in z.files else np.array([''] * len(fi))
    out, n_res = {}, 0
    for i in np.where(kept & np.isin(hand, (0, 1)))[0]:
        resc = mark_rescued and str(rs[i]) != ''
        n_res += resc
        col = GREEN if resc else (BLUE if hand[i] == 0 else RED)
        tag = (('L*' if hand[i] == 0 else 'R*') if resc else '')
        out.setdefault(int(fi[i]), []).append((k2[i], col, tag))
    return out, n_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--original', required=True)
    ap.add_argument('--rescued', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--crf', type=int, default=26)
    a = ap.parse_args()

    A, _ = index(a.original, False)
    B, n_res = index(a.rescued, True)

    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow, oh = int(W * a.scale) // 2 * 2, int(H * a.scale) // 2 * 2
    tmp = a.out + '.tmp.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow * 2, oh))
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    s = a.scale
    seen = 0
    for t in range(N):
        ok, frame = cap.read()
        if not ok: break
        small = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        cells = []
        for title, ix in (('ORIGINAL', A),
                          ('+ HANDEDNESS RESCUE   (green = rescued)', B)):
            img = small.copy()
            for kp, col, tag in ix.get(t, []):
                p = kp * s
                for i, j in E:
                    cv2.line(img, (int(p[i,0]), int(p[i,1])), (int(p[j,0]), int(p[j,1])),
                             col, 1, cv2.LINE_AA)
                for x, y in p:
                    cv2.circle(img, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
                if tag:
                    cv2.putText(img, tag, (int(p[WRIST,0]) + 6, int(p[WRIST,1]) - 6),
                                fnt, 0.5, GREEN, 1, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (ow, 26), (0, 0, 0), -1)
            cv2.putText(img, title, (7, 18), fnt, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
        seen += sum(1 for _, c, _ in B.get(t, []) if c == GREEN)
        strip = np.hstack(cells)
        cv2.putText(strip, f'f{t}   rescued so far: {seen}', (ow * 2 - 240, oh - 10),
                    fnt, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        vw.write(strip)
    cap.release(); vw.release()
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264',
                        '-crf', str(a.crf), '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', a.out], capture_output=True)
    if r.returncode == 0: os.remove(tmp)
    else: os.replace(tmp, a.out)
    print(f'{a.out}  {os.path.getsize(a.out)/1e6:.1f} MB  {ow*2}x{oh}  '
          f'{n_res} rescued detections marked')


if __name__ == '__main__':
    main()
