#!/usr/bin/env python
"""Vérification visuelle de la détection : pose MoveNet sur une image fixe.

Dessine le squelette + keypoints (poignets en surbrillance) et enregistre un PNG.
Usage : python validate_on_image.py [chemin_ou_url] [sortie.png]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from vhs import pose

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]


def load_image(src: str) -> np.ndarray:
    if src.startswith("http"):
        data = urllib.request.urlopen(src).read()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(src)
    if bgr is None:
        raise SystemExit(f"Image illisible : {src}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/zidane.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "out/validate.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    rgb = load_image(src)
    model = pose.MoveNetThunder(str(Path(__file__).parent / "models" / "movenet_thunder.onnx"))
    print(f"provider={model.active_provider}")
    kp = model.infer(rgb)  # [17,3] = (x,y,conf)

    print("Keypoints (x, y, conf) :")
    for i, name in enumerate(pose.KEYPOINT_NAMES):
        print(f"  {i:2d} {name:14s} ({kp[i,0]:7.1f}, {kp[i,1]:7.1f})  conf={kp[i,2]:.2f}")
    for w, lbl in ((pose.RIGHT_WRIST, "droit"), (pose.LEFT_WRIST, "gauche")):
        print(f"  -> poignet {lbl}: conf={kp[w,2]:.2f}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for a, b in EDGES:
        if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
            cv2.line(bgr, (int(kp[a, 0]), int(kp[a, 1])),
                     (int(kp[b, 0]), int(kp[b, 1])), (0, 255, 0), 2)
    for i in range(17):
        if kp[i, 2] > 0.2:
            color = (0, 0, 255) if i in (pose.LEFT_WRIST, pose.RIGHT_WRIST) else (0, 200, 255)
            r = 9 if i in (pose.LEFT_WRIST, pose.RIGHT_WRIST) else 5
            cv2.circle(bgr, (int(kp[i, 0]), int(kp[i, 1])), r, color, -1)
    cv2.imwrite(out, bgr)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
