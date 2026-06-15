"""Calibration pixels -> mètres.

Deux méthodes (cf. recommandations) :
  1. OBJET DE LONGUEUR CONNUE dans le plan du geste (le plus précis).
     scale = longueur_réelle_m / longueur_pixels.
     L'objet DOIT être à la même profondeur que la trajectoire de la main.
  2. TAILLE DU CORPS (auto, sans accessoire ; moins précis, ~±5-10%).
     On estime la stature en pixels à partir des keypoints (œil/oreille -> cheville)
     puis on remonte à la stature complète par un ratio anthropométrique.

Rappel : une seule caméra ne mesure que la vitesse DANS LE PLAN IMAGE -> la valeur
est une BORNE INFÉRIEURE de la vraie vitesse 3D. Le delta entre tes essais (même
réglage) reste fiable même si l'absolu est légèrement sous-estimé.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pose import (LEFT_ANKLE, LEFT_EAR, LEFT_EYE, NOSE, RIGHT_ANKLE,
                   RIGHT_EAR, RIGHT_EYE)

# Fraction de la stature comprise entre le niveau des yeux et l'articulation de la
# cheville (yeux ~0.93*stature, cheville ~0.04*stature -> ~0.89). Approximatif.
EYE_TO_ANKLE_FRACTION = 0.89


@dataclass
class Calibration:
    m_per_px: float
    source: str          # description lisible de la méthode
    detail: dict


def from_reference(real_length_m: float, p1, p2) -> Calibration:
    """Objet de longueur connue : p1, p2 = (x,y) en pixels des deux extrémités."""
    px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if px <= 0:
        raise ValueError("Les deux points de référence sont confondus.")
    return Calibration(
        m_per_px=real_length_m / px,
        source=f"objet de référence ({real_length_m:.3f} m = {px:.1f} px)",
        detail={"real_length_m": real_length_m, "pixel_length": px},
    )


def from_reference_pixels(real_length_m: float, pixel_length: float) -> Calibration:
    if pixel_length <= 0:
        raise ValueError("pixel_length doit être > 0.")
    return Calibration(
        m_per_px=real_length_m / pixel_length,
        source=f"objet de référence ({real_length_m:.3f} m = {pixel_length:.1f} px)",
        detail={"real_length_m": real_length_m, "pixel_length": pixel_length},
    )


def from_height(keypoints: np.ndarray, real_height_m: float,
                conf_thresh: float = 0.4) -> Calibration:
    """Auto-calibration par la taille debout.

    keypoints : [N,17,3] = (x, y, conf). On prend la stature visible (haut de tête
    approx via œil/oreille -> cheville) sur les frames les plus "debout/nettes",
    puis on remonte à la stature complète. y croît vers le bas.
    """
    head_idx = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    ankle_idx = [LEFT_ANKLE, RIGHT_ANKLE]

    spans = []
    for kp in keypoints:
        head_pts = [kp[i] for i in head_idx if kp[i, 2] >= conf_thresh]
        ankle_pts = [kp[i] for i in ankle_idx if kp[i, 2] >= conf_thresh]
        if not head_pts or not ankle_pts:
            continue
        head_y = min(p[1] for p in head_pts)      # plus haut (y min)
        ankle_y = max(p[1] for p in ankle_pts)    # plus bas (y max)
        span = ankle_y - head_y
        if span > 0:
            spans.append(span)

    if len(spans) < 3:
        raise ValueError("Calibration par la taille impossible : corps entier (tête+chevilles) "
                         "rarement visible. Filme le corps en entier sur quelques frames, "
                         "ou utilise un objet de référence (--ref-length-m).")

    spans = np.asarray(spans)
    # On vise une pose bien droite -> la plus grande extension verticale (90e centile,
    # robuste aux frames de fin de geste).
    eye_to_ankle_px = float(np.percentile(spans, 90))
    stature_px = eye_to_ankle_px / EYE_TO_ANKLE_FRACTION
    return Calibration(
        m_per_px=real_height_m / stature_px,
        source=f"taille du corps ({real_height_m:.2f} m, stature≈{stature_px:.0f} px)",
        detail={"real_height_m": real_height_m, "stature_px": stature_px,
                "eye_to_ankle_px": eye_to_ankle_px, "n_frames_used": len(spans),
                "note": "approx ±5-10% ; préférer --ref-length-m pour l'absolu"},
    )


def pick_reference_interactive(frame_rgb: np.ndarray, real_length_m: float) -> Calibration:
    """Affiche une frame et laisse cliquer les deux extrémités de l'objet de référence.

    Nécessite un affichage (cv2 GUI). Clic gauche x2, puis une touche pour valider.
    """
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    pts: list[tuple[int, int]] = []
    win = f"Calibration : clique les 2 extremites ({real_length_m:.3f} m) puis une touche"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = bgr.copy()
        for p in pts:
            cv2.circle(disp, p, 6, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key != 255 and len(pts) == 2:
            break
        if key == 27:  # Echap
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("Calibration annulée.")
    cv2.destroyWindow(win)
    return from_reference(real_length_m, pts[0], pts[1])
