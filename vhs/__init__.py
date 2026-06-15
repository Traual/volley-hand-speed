"""vhs — Volley Hand Speed.

Mesure la vitesse du poignet (geste d'attaque au volley) à partir d'une vidéo
iPhone au ralenti, via pose estimation MoveNet (ONNX) + traitement du signal.

Pipeline :
    vidéo --(PyAV)--> frames + vrai fps --(MoveNet ONNX)--> poignet (px)
          --> gating/interpolation --> filtrage zéro-phase --> dérivée --> vitesse (m/s)
"""

__version__ = "0.1.0"
