"""Décodage vidéo robuste via PyAV.

Pourquoi PyAV plutôt qu'OpenCV pour la lecture :
  1. VRAI fps. Sur un ralenti iPhone (240 fps stocké dans un conteneur 30 fps),
     cv2.CAP_PROP_FPS renvoie souvent 30 -> la vitesse serait 8x trop basse.
     PyAV expose average_rate / base_rate / guessed_rate du flux.
  2. Décodage HEVC (H.265) fiable. Le HEVC iPhone fait régulièrement planter le
     FFmpeg embarqué d'OpenCV sous Windows (frames noires, EOF prématuré).
  3. Timestamps réels par frame (PTS) -> robuste aux frames variables/perdues.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np


def _display_rotation(frame) -> int:
    """Angle de rotation d'affichage (0/90/180/270) lu dans la DISPLAYMATRIX.

    Les vidéos iPhone portrait sont stockées en paysage (1920x1080) + une matrice
    de rotation que PyAV n'applique PAS. Sans ça, MoveNet voit une personne couchée.
    """
    try:
        for sd in frame.side_data:
            if "DISPLAYMATRIX" in str(sd.type):
                m = np.frombuffer(bytes(sd), dtype=np.int32)
                ang = math.degrees(math.atan2(m[1] / 65536.0, m[0] / 65536.0))
                return int(round(ang / 90.0)) * 90 % 360
    except Exception:
        pass
    return 0


# disp = angle CCW de av_display_rotation_get -> on applique une rotation HORAIRE de
# ce même angle pour rétablir l'affichage (validé empiriquement sur clip iPhone).
_ROT_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class VideoInfo:
    width: int
    height: int
    n_frames: int            # nombre de frames réellement décodées
    duration_s: float        # durée d'après les timestamps décodés
    fps_average: float | None   # avg_frame_rate (frames/durée annoncée)
    fps_base: float | None      # r_frame_rate (cadence de base / tick)
    fps_guessed: float | None   # estimation FFmpeg
    fps_effective: float        # 1 / médiane(dt) sur les frames décodées (le plus fiable)
    codec: str
    likely_baked: bool       # True si la cadence semble "aplatie" à ~30 fps

    @property
    def fps(self) -> float:
        """Cadence à utiliser pour le traitement du signal."""
        return self.fps_effective


def _to_float(rate) -> float | None:
    if rate is None:
        return None
    try:
        return float(Fraction(rate))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_frames(path: str):
    """Générateur : produit (index, t_seconds, frame_rgb_uint8 [H,W,3]).

    t_seconds vient du PTS réel quand il est disponible, sinon None
    (le caller retombera alors sur un échantillonnage uniforme).
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        start_pts = None
        rot_code = None
        rot_known = False
        for i, frame in enumerate(container.decode(stream)):
            if not rot_known:                      # rotation constante : lue une fois
                rot_code = _ROT_CODE.get(_display_rotation(frame))
                rot_known = True
            t = None
            if frame.pts is not None and time_base is not None:
                if start_pts is None:
                    start_pts = frame.pts
                t = float((frame.pts - start_pts) * time_base)
            img = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8, RGB
            if rot_code is not None:
                img = cv2.rotate(img, rot_code)     # rétablit l'orientation portrait
            yield i, t, img


def read_meta(path: str) -> dict:
    """Métadonnées du flux SANS tout décoder (rapide)."""
    with av.open(path) as container:
        s = container.streams.video[0]
        w, h = s.codec_context.width, s.codec_context.height
        disp = 0
        try:
            for f in container.decode(s):          # 1 frame pour lire la rotation
                disp = _display_rotation(f)
                break
        except Exception:
            pass
        if disp in (90, 270):
            w, h = h, w                            # dimensions d'affichage (portrait)
        return {
            "width": w,
            "height": h,
            "codec": s.codec_context.name,
            "fps_average": _to_float(s.average_rate),
            "fps_base": _to_float(s.base_rate),
            "fps_guessed": _to_float(s.guessed_rate),
        }


def make_info(meta: dict, times, n_frames: int, width: int, height: int) -> VideoInfo:
    """Construit VideoInfo à partir des métadonnées + timestamps mesurés à la passe pose."""
    times_arr = np.asarray([t for t in times if t is not None], dtype=float)
    if times_arr.size >= 2:
        dt = np.diff(times_arr)
        dt = dt[dt > 0]
        fps_eff = float(1.0 / np.median(dt)) if dt.size else (meta["fps_average"] or 30.0)
        duration = float(times_arr[-1] - times_arr[0])
    else:
        fps_eff = meta["fps_average"] or meta["fps_guessed"] or meta["fps_base"] or 30.0
        duration = n_frames / fps_eff if fps_eff else 0.0
    cands = [c for c in (fps_eff, meta["fps_average"], meta["fps_base"]) if c]
    baked = bool(cands) and max(cands) <= 65.0
    return VideoInfo(
        width=meta["width"] or width, height=meta["height"] or height,
        n_frames=n_frames, duration_s=duration,
        fps_average=meta["fps_average"], fps_base=meta["fps_base"],
        fps_guessed=meta["fps_guessed"], fps_effective=fps_eff,
        codec=meta["codec"], likely_baked=baked,
    )


def probe(path: str) -> VideoInfo:
    """Décode l'intégralité du clip une fois pour mesurer le VRAI fps.

    On ne se fie pas aux métadonnées seules : on calcule fps_effective à partir
    des timestamps des frames effectivement décodées (gère VFR / frames perdues).
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        codec = stream.codec_context.name
        fps_average = _to_float(stream.average_rate)
        fps_base = _to_float(stream.base_rate)
        fps_guessed = _to_float(stream.guessed_rate)

    times: list[float] = []
    n = 0
    last_w = last_h = 0
    for _, t, img in read_frames(path):
        n += 1
        last_h, last_w = img.shape[:2]
        if t is not None:
            times.append(t)

    times_arr = np.asarray(times, dtype=float)
    if times_arr.size >= 2:
        dt = np.diff(times_arr)
        dt = dt[dt > 0]
        fps_effective = float(1.0 / np.median(dt)) if dt.size else (fps_average or 30.0)
        duration_s = float(times_arr[-1] - times_arr[0])
    else:
        # Pas de PTS exploitable : on retombe sur les métadonnées.
        fps_effective = fps_average or fps_guessed or fps_base or 30.0
        duration_s = n / fps_effective if fps_effective else 0.0

    # Heuristique "ralenti aplati" : un clip que l'utilisateur croit en ralenti
    # mais dont toutes les cadences retombent autour de 30 fps a probablement été
    # ré-encodé (AirDrop/montage) en perdant la résolution temporelle 240 fps.
    candidates = [c for c in (fps_effective, fps_average, fps_base) if c]
    likely_baked = bool(candidates) and max(candidates) <= 65.0

    return VideoInfo(
        width=width or last_w,
        height=height or last_h,
        n_frames=n,
        duration_s=duration_s,
        fps_average=fps_average,
        fps_base=fps_base,
        fps_guessed=fps_guessed,
        fps_effective=fps_effective,
        codec=codec,
        likely_baked=likely_baked,
    )
