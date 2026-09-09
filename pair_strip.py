#!/usr/bin/env python3
"""Two labelled keypoint sets, side by side over the same video, time-locked."""
import argparse, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abc_strip import variant_index, HAND_EDGES

ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True)
ap.add_argument('--left', required=True);  ap.add_argument('--left-title', default='LEFT')
ap.add_argument('--right', required=True); ap.add_argument('--right-title', default='RIGHT')
ap.add_argument('--out', required=True)
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--crf', type=int, default=28)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ow, oh = int(W*a.scale)//2*2, int(H*a.scale)//2*2
panels = [(a.left_title, *variant_index(a.left, 'h21', W)),
          (a.right_title, *variant_index(a.right, 'h21', W))]
tmp = a.out + '.tmp.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow*2, oh))
fnt = cv2.FONT_HERSHEY_SIMPLEX
for t in range(N):
    ok, fr = cap.read()
    if not ok: break
    small = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
    cells = []
    for title, idx, cap_ in panels:
        img = small.copy()
        for kp, col in idx.get(t, []):
            p = kp * a.scale
            for i, j in HAND_EDGES:
                cv2.line(img, (int(p[i,0]),int(p[i,1])), (int(p[j,0]),int(p[j,1])), col, 1, cv2.LINE_AA)
            for x, y in p: cv2.circle(img, (int(x),int(y)), 2, col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (0,0), (ow,26), (0,0,0), -1)
        cv2.putText(img, f'{title}  [{cap_}]', (7,18), fnt, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cells.append(img)
    strip = np.hstack(cells)
    cv2.putText(strip, f'f{t}', (ow*2-70, oh-10), fnt, 0.5, (0,255,255), 1, cv2.LINE_AA)
    vw.write(strip)
cap.release(); vw.release()
r = subprocess.run(['ffmpeg','-y','-loglevel','error','-i',tmp,'-c:v','libx264','-crf',str(a.crf),
                    '-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',a.out],
                   capture_output=True, stdin=subprocess.DEVNULL)
os.remove(tmp) if r.returncode == 0 else os.replace(tmp, a.out)
print(f'{a.out}  {os.path.getsize(a.out)/1e6:.1f} MB')
