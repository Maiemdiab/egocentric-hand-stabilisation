#!/usr/bin/env python3
"""Pair two already-rendered viz panels side by side, time-locked, with a title bar on each.

Both inputs come from the SAME renderer with the same head pose and the same flags -- only the hand
label file differs -- so any visible difference between the two halves is the hand pipeline and
nothing else. Frame counts can differ by a frame or two if the two runs saw different tails; the
shorter one is held on its last frame rather than truncating the comparison.
"""
from __future__ import annotations
import argparse, os, subprocess
import cv2
import numpy as np


def letterbox_rows(cap, probe=5):
    """Rows of pure black at the top and bottom, sampled over a few frames.

    The delivered panels were padded to 16:9 by the web conversion; the pad is not content and
    doubling it when stacking would push the real pixels off screen."""
    top = bot = None
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for _ in range(probe):
        ok, f = cap.read()
        if not ok: break
        g = f.max(axis=(1, 2))                      # brightest pixel per row
        nz = np.where(g > 8)[0]
        if len(nz) == 0: continue
        t, b = int(nz[0]), int(len(g) - 1 - nz[-1])
        top = t if top is None else min(top, t)
        bot = b if bot is None else min(bot, b)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return (top or 0), (bot or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left', required=True); ap.add_argument('--right', required=True)
    ap.add_argument('--left-title', default='DELIVERED'); ap.add_argument('--right-title', default='C+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--stack', choices=('h', 'v'), default='v',
                    help="v stacks the two panels vertically. Side by side, a 2892-wide panel "
                         "shown twice is 3840+ px across and each half is unreadable on a normal "
                         "screen; stacked, each keeps full width and the eye compares the SAME "
                         "region of the two renders by looking straight up and down.")
    ap.add_argument('--keep-letterbox', action='store_true',
                    help='keep the black bars the delivered web conversion padded in. By default '
                         'they are detected and cropped, which is ~33%% of the height of each '
                         'panel and pure waste once the panels are stacked.')
    ap.add_argument('--crf', type=int, default=24)
    a = ap.parse_args()

    ca, cb = cv2.VideoCapture(a.left), cv2.VideoCapture(a.right)
    fps = ca.get(cv2.CAP_PROP_FPS) or 30.0
    wa, ha = int(ca.get(3)), int(ca.get(4))
    wb, hb = int(cb.get(3)), int(cb.get(4))
    ta = ba = tb = bb = 0
    if not a.keep_letterbox:
        ta, ba = letterbox_rows(ca); tb, bb = letterbox_rows(cb)
        ha -= ta + ba; hb -= tb + bb
    if a.stack == 'v':
        W = int(max(wa, wb) * a.scale) // 2 * 2
        HA = int(ha * W / wa) // 2 * 2; HB = int(hb * W / wb) // 2 * 2
    else:
        H = int(max(ha, hb) * a.scale) // 2 * 2
        WA = int(wa * H / ha) // 2 * 2; WB = int(wb * H / hb) // 2 * 2
    na = int(ca.get(cv2.CAP_PROP_FRAME_COUNT)); nb = int(cb.get(cv2.CAP_PROP_FRAME_COUNT))
    N = max(na, nb)
    BAR = 30
    tmp = a.out + '.tmp.mp4'
    OW = W if a.stack == 'v' else WA + WB
    OH = (HA + HB + 2 * BAR) if a.stack == 'v' else (H + BAR)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OW, OH))
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    la = lb = None
    for t in range(N):
        oka, fa = ca.read()
        okb, fb = cb.read()
        if oka:
            fa = fa[ta:fa.shape[0] - ba] if (ta or ba) else fa
            la = cv2.resize(fa, (W, HA) if a.stack == 'v' else (WA, H), interpolation=cv2.INTER_AREA)
        if okb:
            fb = fb[tb:fb.shape[0] - bb] if (tb or bb) else fb
            lb = cv2.resize(fb, (W, HB) if a.stack == 'v' else (WB, H), interpolation=cv2.INTER_AREA)
        if la is None or lb is None: break
        strip = np.zeros((OH, OW, 3), np.uint8)
        if a.stack == 'v':
            strip[BAR:BAR + HA] = la
            strip[2 * BAR + HA:] = lb
            cv2.putText(strip, a.left_title, (10, 21), fnt, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(strip, a.right_title, (10, BAR + HA + 21), fnt, 0.62, (120, 255, 120), 2, cv2.LINE_AA)
            cv2.line(strip, (0, BAR + HA), (OW, BAR + HA), (70, 70, 70), 2)
            cv2.putText(strip, f'f{t}', (OW - 80, OH - 9), fnt, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            strip[BAR:, :WA] = la; strip[BAR:, WA:] = lb
            cv2.putText(strip, a.left_title, (10, 21), fnt, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(strip, a.right_title, (WA + 10, 21), fnt, 0.62, (120, 255, 120), 2, cv2.LINE_AA)
            cv2.line(strip, (WA, 0), (WA, OH), (70, 70, 70), 2)
            cv2.putText(strip, f'f{t}', (OW - 80, OH - 9), fnt, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        vw.write(strip)
    ca.release(); cb.release(); vw.release()
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264',
                        '-crf', str(a.crf), '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', a.out], capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode == 0: os.remove(tmp)
    else: os.replace(tmp, a.out)
    print(f'{a.out}  {os.path.getsize(a.out)/1e6:.1f} MB  {OW}x{OH}  stack={a.stack}  '
          f'cropped letterbox L={ta}+{ba} R={tb}+{bb}  frames L={na} R={nb}')


if __name__ == '__main__':
    main()
