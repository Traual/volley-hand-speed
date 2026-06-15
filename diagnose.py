#!/usr/bin/env python
"""Diagnostic qualité sur un clip : pourquoi un poignet semble anormalement rapide.

Usage : python diagnose.py <clip>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vhs import kinematics, pose, video

path = sys.argv[1]
model = pose.MoveNetThunder(str(Path(__file__).parent / "models" / "movenet_thunder.onnx"))
meta = video.read_meta(path)

# Passe 1 : pose
kps, times = [], []
for i, t, rgb in video.read_frames(path):
    kps.append(model.infer(rgb))
    times.append(t)
K = np.asarray(kps)
N = len(K)

# --- Cadence / VFR ---
ta = np.asarray([x for x in times if x is not None])
dt = np.diff(ta)
dt = dt[dt > 0]
print(f"\nFrames={N}  durée={ta[-1]-ta[0]:.3f}s")
print(f"fps: avg_meta={meta['fps_average']:.1f}  via_count={N/(ta[-1]-ta[0]):.1f}  "
      f"median_dt={1/np.median(dt):.1f}")
print(f"dt: min={dt.min()*1000:.2f}ms max={dt.max()*1000:.2f}ms "
      f"-> jitter cadence x{dt.max()/dt.min():.1f}  (VFR si >>1)")

# --- Qualité par poignet ---
fps = N / (ta[-1] - ta[0])
for name, idx in (("DROIT", pose.RIGHT_WRIST), ("GAUCHE", pose.LEFT_WRIST)):
    conf = K[:, idx, 2]
    x, y = K[:, idx, 0], K[:, idx, 1]
    jumps = np.hypot(np.diff(x), np.diff(y))
    big = (jumps > 80).sum()
    print(f"\nPoignet {name} (idx {idx}):")
    print(f"  conf: médiane={np.median(conf):.2f}  %≥0.3={100*(conf>=0.3).mean():.0f}%  "
          f"%≥0.5={100*(conf>=0.5).mean():.0f}%")
    print(f"  sauts >80px entre frames: {big} ({100*big/N:.0f}% des frames) "
          "<- téléportations = détection instable / confusion G-D")
    res = kinematics.compute_speed(x, y, conf, ta, 1.0, fps, cutoff_hz=15.0)
    pi = res.peak_index
    interp_at_peak = conf[pi] < 0.3
    print(f"  pic à frame {pi}: conf brute={conf[pi]:.2f} "
          f"{'(INTERPOLÉE, peu fiable)' if interp_at_peak else '(détectée)'}")

# Passe 2 : rendre les frames de pic des deux poignets
peaks = {}
for name, idx in (("right", pose.RIGHT_WRIST), ("left", pose.LEFT_WRIST)):
    conf = K[:, idx, 2]
    res = kinematics.compute_speed(K[:, idx, 0], K[:, idx, 1], conf, ta, 1.0, fps, cutoff_hz=15.0)
    peaks[name] = res.peak_index
want = set(peaks.values())
grabbed = {}
for i, _, rgb in video.read_frames(path):
    if i in want:
        grabbed[i] = rgb.copy()
    if len(grabbed) == len(want):
        break

Path("out").mkdir(exist_ok=True)
for name, pi in peaks.items():
    rgb = grabbed.get(pi)
    if rgb is None:
        continue
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kp = K[pi]
    for j, col, r in ((pose.RIGHT_WRIST, (0, 0, 255), 12), (pose.LEFT_WRIST, (255, 0, 0), 12),
                      (pose.RIGHT_ELBOW, (0, 255, 0), 6), (pose.LEFT_ELBOW, (0, 255, 0), 6),
                      (pose.RIGHT_SHOULDER, (0, 255, 255), 6), (pose.LEFT_SHOULDER, (0, 255, 255), 6)):
        cv2.circle(bgr, (int(kp[j, 0]), int(kp[j, 1])), r, col, -1)
    cv2.putText(bgr, f"pic {name} (rouge=poignet D, bleu=G) conf D={kp[pose.RIGHT_WRIST,2]:.2f} G={kp[pose.LEFT_WRIST,2]:.2f}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    out = f"out/diag_pic_{name}_frame{pi}.png"
    cv2.imwrite(out, bgr)
    print(f"-> {out}")
