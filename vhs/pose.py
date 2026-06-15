"""Pose estimation MoveNet SinglePose Thunder (ONNX) via onnxruntime-DirectML.

MoveNet Thunder = régression mono-personne (pas de détection, pas d'anchors,
pas de NMS) -> coordonnées sous-pixel directes, peu de jitter. C'est exactement
ce qu'il faut quand on va DÉRIVER la position pour obtenir une vitesse
(la dérivation amplifie le bruit haute fréquence).

Modèle : Xenova/movenet-singlepose-thunder (Apache-2.0, opset 11).
  Entrée  : "input"   int32  [1, 256, 256, 3] NHWC, RGB, valeurs 0-255 (NON normalisé)
  Sortie  : "output_0" float32 [1, 1, 17, 3]  -> (y_norm, x_norm, score) par keypoint,
            y/x normalisés 0-1 par rapport à l'image 256x256 *paddée*.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

# Ordre COCO-17 (identique au fallback YOLOv8-pose)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

WRIST_INDEX = {"left": LEFT_WRIST, "right": RIGHT_WRIST}


class MoveNetThunder:
    SIZE = 256

    def __init__(self, model_path: str, providers: list[str] | None = None):
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_type = inp.type  # ex. 'tensor(int32)'
        if "uint8" in self.input_type:
            self._dtype = np.uint8
        elif "int32" in self.input_type:
            self._dtype = np.int32
        elif "float" in self.input_type:
            self._dtype = np.float32
        else:
            self._dtype = np.int32

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    def _preprocess(self, rgb: np.ndarray):
        """resize-with-pad vers 256x256 en conservant le ratio (façon MoveNet).

        Retourne (tensor [1,256,256,3], scale, pad_x, pad_y) pour pouvoir
        re-projeter ensuite les keypoints vers les pixels d'origine.
        """
        h, w = rgb.shape[:2]
        scale = self.SIZE / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=self._dtype)
        pad_x = (self.SIZE - nw) // 2
        pad_y = (self.SIZE - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas[None, ...], scale, pad_x, pad_y

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """rgb [H,W,3] uint8 -> keypoints [17, 3] = (x_px, y_px, score) en pixels d'origine."""
        inp, scale, pad_x, pad_y = self._preprocess(rgb)
        out = self.session.run(None, {self.input_name: inp})[0]  # [1,1,17,3]
        kp = out[0, 0]                       # [17,3] = (y_norm, x_norm, score)
        y_pad = kp[:, 0] * self.SIZE
        x_pad = kp[:, 1] * self.SIZE
        conf = kp[:, 2]
        x_orig = (x_pad - pad_x) / scale
        y_orig = (y_pad - pad_y) / scale
        return np.stack([x_orig, y_orig, conf], axis=1)


def track(model: MoveNetThunder, frame_iter, progress=None):
    """Lance la pose sur toutes les frames.

    frame_iter : générateur de (index, t_seconds_or_None, frame_rgb).
    Retourne (keypoints [N,17,3], times [N]) ; times en secondes (uniforme si PTS absent).
    """
    kps: list[np.ndarray] = []
    times: list[float | None] = []
    for i, t, rgb in frame_iter:
        kps.append(model.infer(rgb))
        times.append(t)
        if progress is not None:
            progress(i)
    keypoints = np.asarray(kps, dtype=float)          # [N,17,3]
    if any(t is None for t in times):
        times_arr = np.arange(len(times), dtype=float)  # placeholder; corrigé par le caller
    else:
        times_arr = np.asarray(times, dtype=float)
    return keypoints, times_arr
