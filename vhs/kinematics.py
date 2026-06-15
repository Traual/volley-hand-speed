"""Traitement du signal : trajectoire bruitée du poignet -> courbe de vitesse propre.

Chaîne (ordre important) :
    gating de confiance -> rejet d'outliers (Hampel) -> interpolation des petits trous
    -> filtre passe-bas zéro-phase -> dérivée centrée -> magnitude de vitesse.

Choix clés (cf. littérature biomécanique, Winter ; lancer/pitching) :
  * Filtre Butterworth zéro-phase (sosfiltfilt). Passe-bande plat (pas de distorsion
    d'amplitude dans la bande utile) + phase nulle (le pic de vitesse n'est pas décalé
    dans le temps). C'est LE standard en biomécanique.
  * Coupure ~12 Hz par défaut. Un geste balistique distal (poignet d'un spike/lancer)
    porte du signal jusqu'à ~10-15 Hz : les 6 Hz de la marche écraseraient le pic.
    L'analyse résiduelle de Winter permet de confirmer la coupure par clip.
  * On filtre la POSITION puis on dérive (la dérivation est un passe-haut : filtrer
    d'abord évite d'amplifier le bruit).
  * Dérivée centrée via np.gradient : O(dt^2), aucun décalage temporel, accepte un
    vecteur temps non uniforme (frames perdues).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


@dataclass
class SpeedResult:
    t: np.ndarray              # temps (s)
    speed_mps: np.ndarray      # vitesse (m/s), NaN dans les longs trous
    x_px: np.ndarray           # trajectoire filtrée (px) — pour overlay/debug
    y_px: np.ndarray
    valid: np.ndarray          # masque bool des frames exploitables
    peak_speed_mps: float
    peak_time_s: float
    peak_index: int
    peak_conf: float
    cutoff_hz: float
    method: str
    peak_interpolated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def peak_speed_kmh(self) -> float:
        return self.peak_speed_mps * 3.6


@dataclass
class AngularResult:
    t: np.ndarray
    angle_deg: np.ndarray         # angle filtré (deg), NaN hors détection
    omega_deg_s: np.ndarray       # vitesse angulaire (deg/s), NaN hors détection
    omega_rad_s: np.ndarray
    valid: np.ndarray
    peak_index: int
    peak_time_s: float
    peak_omega_deg_s: float
    peak_omega_rad_s: float
    peak_conf: float
    cutoff_hz: float
    kind: str                     # "upperarm" | "elbow" | "trunk"
    side: str
    meta: dict = field(default_factory=dict)


def hampel(x: np.ndarray, k: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Rejet d'outliers par fenêtre glissante (médiane + MAD). Les outliers -> NaN.

    Tue les sauts d'un seul frame (keypoint qui part en vrille) qui se traduiraient
    par un faux pic de vitesse. Ignore les NaN déjà présents.
    """
    x = x.astype(float).copy()
    n = len(x)
    half = k // 2
    out = x.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = x[lo:hi]
        win = win[np.isfinite(win)]
        if win.size < 3 or not np.isfinite(x[i]):
            continue
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        sigma = 1.4826 * mad
        if sigma > 0 and abs(x[i] - med) > n_sigma * sigma:
            out[i] = np.nan
    return out


def _fill_short_gaps(sig: np.ndarray, t: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpole (spline cubique) les trous <= max_gap frames ; laisse les longs en NaN."""
    out = sig.copy()
    valid = np.isfinite(sig)
    if valid.sum() < 4:
        return out
    cs = CubicSpline(t[valid], sig[valid])
    n = len(sig)
    i = 0
    while i < n:
        if not np.isfinite(sig[i]):
            j = i
            while j < n and not np.isfinite(sig[j]):
                j += 1
            if (j - i) <= max_gap and i > 0 and j < n:   # trou interne court
                out[i:j] = cs(t[i:j])
            i = j
        else:
            i += 1
    return out


def residual_cutoff(sig: np.ndarray, fps: float,
                    f_min: float = 2.0, f_max: float | None = None,
                    step: float = 0.5) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Analyse résiduelle de Winter -> coupure optimale (Hz).

    Pour chaque fc : R(fc) = RMS(raw - filtré). La courbe chute puis s'aplatit en
    une zone ~linéaire (bruit blanc). On ajuste une droite sur cette queue, on
    l'extrapole à fc=0 (intercept 'a' = RMS du bruit), et fc_opt = plus petite fc
    où R(fc) <= a. (Yu et al. : l'analyse résiduelle tend à sous-estimer fc à haut
    fps -> on la borne ensuite avec un a priori métier.)
    """
    sig = np.asarray(sig, float)
    finite = np.isfinite(sig)
    sig = np.interp(np.arange(len(sig)), np.flatnonzero(finite), sig[finite])
    nyq = fps / 2.0
    if f_max is None:
        f_max = 0.45 * nyq
    f_max = min(f_max, 0.9 * nyq)
    fcs = np.arange(f_min, f_max, step)
    R = np.empty_like(fcs)
    for idx, fc in enumerate(fcs):
        sos = signal.butter(2, fc, btype="low", fs=fps, output="sos")
        filt = signal.sosfiltfilt(sos, sig)
        R[idx] = np.sqrt(np.mean((sig - filt) ** 2))
    tail = fcs > fcs[len(fcs) // 2]
    coeffs = np.polyfit(fcs[tail], R[tail], 1)
    a = float(np.polyval(coeffs, 0.0))
    below = np.where(R <= a)[0]
    fc_opt = float(fcs[below[0]]) if below.size else float(fcs[-1])
    return fc_opt, fcs, R, a


def choose_cutoff(x: np.ndarray, y: np.ndarray, fps: float,
                  prior=(8.0, 18.0)) -> float:
    """Coupure auto : analyse résiduelle, bornée par l'a priori métier (8-18 Hz)
    et par Nyquist (<= 0.45 * fps/2)."""
    lo, hi = prior
    fc_x, *_ = residual_cutoff(x, fps)
    fc_y, *_ = residual_cutoff(y, fps)
    fc = float(np.mean([fc_x, fc_y]))
    fc = float(np.clip(fc, lo, hi))
    fc = min(fc, 0.45 * fps / 2.0)
    return fc


def assess_quality(keypoints: np.ndarray, result: "SpeedResult",
                   times: np.ndarray) -> tuple[bool, list[str]]:
    """Juge si le résultat est exploitable. Retourne (fiable, avertissements).

    keypoints : [N,17,3] (x,y,conf) ; result : SpeedResult du poignet retenu.
    """
    warns: list[str] = []
    reliable = True

    # --- Fiabilité = qualité de détection AUTOUR DU PIC (la mesure elle-même) ---
    if result.peak_interpolated:
        warns.append(f"le PIC tombe sur une frame non détectée (conf {result.peak_conf:.2f}) "
                     "— probablement un artefact, pas une vraie vitesse.")
        reliable = False

    loc_det = result.meta.get("peak_local_frac_detected", 1.0)
    if loc_det < 0.6:
        warns.append(f"autour du pic, le poignet n'est détecté que sur {loc_det*100:.0f}% "
                     "des frames — mesure peu fiable (flou / hors cadre au moment clé).")
        reliable = False

    loc_med = result.meta.get("peak_local_median_conf", 1.0)
    if loc_med < 0.40:
        warns.append(f"confiance faible autour du pic ({loc_med:.2f}) — mesure peu fiable.")
        reliable = False

    # --- Cadence variable (VFR) : casse les vitesses, signale un ré-encodage ---
    t = np.asarray(times, float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size and dt.max() / dt.min() > 1.3:
        warns.append(f"cadence variable (VFR, dt {dt.min()*1000:.1f}–{dt.max()*1000:.1f} ms) : "
                     "clip ré-encodé (WhatsApp/montage). Envoie l'ORIGINAL.")
        reliable = False

    # --- Infos (n'invalident pas la mesure si le pic est propre) ---
    fi = result.meta.get("frac_interpolated", 0.0)
    if fi > 0.3:
        warns.append(f"(info) {fi*100:.0f}% du clip interpolé, mais surtout HORS du geste "
                     "(phases où tu poses/reprends la caméra) — sans impact sur le pic.")

    return reliable, warns


def compute_speed(x_px: np.ndarray, y_px: np.ndarray, conf: np.ndarray,
                  t: np.ndarray, m_per_px: float, fps: float,
                  conf_thresh: float = 0.3, max_gap_s: float = 0.05,
                  cutoff_hz: float | None = None, method: str = "butter",
                  butter_order: int = 2, savgol_time_s: float = 0.13,
                  eligible: np.ndarray | None = None,
                  peak_conf_min: float = 0.5) -> SpeedResult:
    """Trajectoire poignet (px) -> SpeedResult (vitesse en m/s).

    method = "butter" : Butterworth zéro-phase sur la position puis dérivée centrée.
    method = "savgol" : Savitzky-Golay (lisse + dérive en une passe), préserve bien le pic.
    """
    x = x_px.astype(float).copy()
    y = y_px.astype(float).copy()
    n = len(x)

    # 1. Gating de confiance -> frames invalides en NaN
    bad = (conf < conf_thresh) | ~np.isfinite(x) | ~np.isfinite(y)
    x[bad] = np.nan
    y[bad] = np.nan

    # 2. Rejet d'outliers (Hampel) sur chaque axe
    x = hampel(x)
    y = hampel(y)

    # 3. Interpolation des petits trous (<= max_gap frames)
    max_gap = max(1, int(round(max_gap_s * fps)))
    x = _fill_short_gaps(x, t, max_gap)
    y = _fill_short_gaps(y, t, max_gap)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        raise ValueError("Trop peu de frames exploitables (poignet rarement détecté). "
                         "Vérifie le cadrage / la confiance / le bon poignet.")

    # Pour filtrer il faut un signal sans NaN : on interpole les longs trous
    # uniquement pour le calcul, puis on re-masque pour le pic.
    xs_full = np.interp(t, t[valid], x[valid])
    ys_full = np.interp(t, t[valid], y[valid])

    # 4. Coupure
    if cutoff_hz is None:
        cutoff_hz = choose_cutoff(xs_full, ys_full, fps)

    # 5. Filtrage + dérivée -> vitesse (px/s)
    if method == "savgol":
        win = max(5, int(round(savgol_time_s * fps)))
        if win % 2 == 0:
            win += 1
        poly = 3 if win > 5 else 2
        xs = signal.savgol_filter(xs_full, win, poly)
        ys = signal.savgol_filter(ys_full, win, poly)
        vx = signal.savgol_filter(xs_full, win, poly, deriv=1, delta=1.0 / fps)
        vy = signal.savgol_filter(ys_full, win, poly, deriv=1, delta=1.0 / fps)
    else:  # butter
        if cutoff_hz >= fps / 2.0:
            cutoff_hz = 0.45 * fps / 2.0
        sos = signal.butter(butter_order, cutoff_hz, btype="low", fs=fps, output="sos")
        xs = signal.sosfiltfilt(sos, xs_full)
        ys = signal.sosfiltfilt(sos, ys_full)
        vx = np.gradient(xs, t)
        vy = np.gradient(ys, t)

    speed_px = np.hypot(vx, vy)
    speed_mps = speed_px * m_per_px

    # Courbe de vitesse de sortie : masquée aux frames valides
    speed_masked = speed_mps.copy()
    speed_masked[~valid] = np.nan

    # 6. Pic — cherché UNIQUEMENT sur des frames réellement détectées (non
    #    interpolées) ET, si fourni, où le corps est visible (eligible).
    #    -> ignore automatiquement l'approche/le retour caméra (gros plans).
    conf_arr = np.asarray(conf, dtype=float)
    # Le pic doit être sur une frame où le poignet est VRAIMENT détecté (conf élevée),
    # pas interpolée : exclut les gros plans approche/retour (poignet conf basse).
    candidate = valid & (conf_arr >= peak_conf_min)
    if eligible is not None:
        candidate = candidate & np.asarray(eligible, dtype=bool)
    if not candidate.any():
        candidate = valid & (conf_arr >= conf_thresh)
    if not candidate.any():
        candidate = valid                      # dernier repli
    speed_for_peak = np.where(candidate, speed_mps, np.nan)
    peak_index = int(np.nanargmax(speed_for_peak))
    # Pic INSTANTANÉ = sommet du signal filtré (pas une moyenne) : c'est le momentum
    # au contact qui compte, pas la vitesse moyenne du geste. Le filtrage zéro-phase
    # + Hampel ont déjà retiré le bruit/jitter, donc ce sommet est robuste.
    peak_speed = float(speed_masked[peak_index])

    # Qualité LOCALE autour du pic (±0.3 s) = fiabilité de la mesure elle-même,
    # indépendamment des phases approche/retour qui polluent les stats globales.
    half = max(1, int(round(0.3 * fps)))
    wlo, whi = max(0, peak_index - half), min(n, peak_index + half + 1)
    loc_conf = np.asarray(conf, dtype=float)[wlo:whi]
    peak_local_frac_detected = float((loc_conf >= conf_thresh).mean())
    peak_local_median_conf = float(np.median(loc_conf))
    return SpeedResult(
        t=t, speed_mps=speed_masked, x_px=xs, y_px=ys, valid=valid,
        peak_speed_mps=peak_speed, peak_time_s=float(t[peak_index]),
        peak_index=peak_index, peak_conf=float(conf_arr[peak_index]),
        cutoff_hz=float(cutoff_hz), method=method,
        peak_interpolated=bool(conf_arr[peak_index] < conf_thresh),
        meta={"n_valid": int(valid.sum()), "n_total": n,
              "frac_interpolated": float(1.0 - valid.sum() / n),
              "median_conf": float(np.median(conf_arr)),
              "frac_conf_ok": float((conf_arr >= conf_thresh).mean()),
              "n_candidate": int(candidate.sum()),
              "peak_local_frac_detected": peak_local_frac_detected,
              "peak_local_median_conf": peak_local_median_conf},
    )


def compute_joint_angle(keypoints: np.ndarray, t: np.ndarray, fps: float,
                        side: str = "right", kind: str = "upperarm",
                        conf_thresh: float = 0.3, max_gap_s: float = 0.05,
                        cutoff_hz: float | None = None, butter_order: int = 2,
                        prior=(6.0, 14.0), eligible: np.ndarray | None = None,
                        peak_conf_min: float = 0.5) -> AngularResult:
    """Vitesse angulaire d'un segment (deg/s), même chaîne que compute_speed.

    kind="upperarm" : orientation du bras (épaule->coude) vs horizontale image
                      = "cadence de swing d'épaule" (PAS la rotation interne 3D).
    kind="elbow"    : flexion du coude (angle avant-bras vs bras), signé.
    kind="trunk"    : bras relatif au tronc (axe hanche->épaule).

    Spécificités angulaires : y image inversé (sinon signe faux), et np.unwrap AVANT
    filtrage/dérivée (sinon le saut de ±2π de atan2 crée un faux pic).
    """
    from .pose import (LEFT_ELBOW, LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST,
                       RIGHT_ELBOW, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST)
    L = side == "left"
    SH = LEFT_SHOULDER if L else RIGHT_SHOULDER
    EL = LEFT_ELBOW if L else RIGHT_ELBOW
    WR = LEFT_WRIST if L else RIGHT_WRIST
    HIP = LEFT_HIP if L else RIGHT_HIP

    K = np.asarray(keypoints, dtype=float)
    n = len(K)

    def pt(i):  # x, y_up (y image inversé -> repère math), conf
        return K[:, i, 0], -K[:, i, 1], K[:, i, 2]

    sx, sy, sc = pt(SH); ex, ey, ec = pt(EL); wx, wy, wc = pt(WR); hx, hy, hc = pt(HIP)

    if kind == "upperarm":
        theta = np.arctan2(ey - sy, ex - sx)
        good = (sc >= conf_thresh) & (ec >= conf_thresh)
        pconf = np.minimum(sc, ec)
    elif kind == "elbow":                       # atan2(cross, dot), pas acos (instable)
        ax, ay = sx - ex, sy - ey
        bx, by = wx - ex, wy - ey
        theta = np.arctan2(ax * by - ay * bx, ax * bx + ay * by)
        good = (sc >= conf_thresh) & (ec >= conf_thresh) & (wc >= conf_thresh)
        pconf = np.minimum.reduce([sc, ec, wc])
    elif kind == "trunk":
        tx, ty = hx - sx, hy - sy
        ux, uy = ex - sx, ey - sy
        theta = np.arctan2(tx * uy - ty * ux, tx * ux + ty * uy)
        good = (sc >= conf_thresh) & (ec >= conf_thresh) & (hc >= conf_thresh)
        pconf = np.minimum.reduce([sc, ec, hc])
    else:
        raise ValueError(f"kind inconnu: {kind}")

    theta = theta.astype(float)
    theta[~good | ~np.isfinite(theta)] = np.nan

    valid0 = np.isfinite(theta)
    if valid0.sum() < 5:
        raise ValueError(f"Trop peu de frames exploitables pour l'angle '{kind}' (côté {side}).")

    # unwrap AVANT tout (sur le signal radian rempli), puis re-masque les trous
    th_lin = np.interp(t, t[valid0], theta[valid0])
    th_unw = np.unwrap(th_lin)
    th_unw[~valid0] = np.nan

    th_unw = hampel(th_unw)
    max_gap = max(1, int(round(max_gap_s * fps)))
    th_unw = _fill_short_gaps(th_unw, t, max_gap)
    valid = np.isfinite(th_unw)
    th_full = np.interp(t, t[valid], th_unw[valid])

    if cutoff_hz is None:
        fc, *_ = residual_cutoff(th_full, fps)
        cutoff_hz = float(np.clip(fc, prior[0], prior[1]))
    cutoff_hz = min(cutoff_hz, 0.45 * fps / 2.0)

    sos = signal.butter(butter_order, cutoff_hz, btype="low", fs=fps, output="sos")
    th_filt = signal.sosfiltfilt(sos, th_full)
    omega_rad = np.gradient(th_filt, t)
    omega_deg = np.degrees(omega_rad)

    angle_deg = np.degrees(th_filt)
    angle_deg[~valid] = np.nan
    omega_masked = omega_deg.copy()
    omega_masked[~valid] = np.nan

    candidate = valid & (pconf >= peak_conf_min)
    if eligible is not None:
        candidate = candidate & np.asarray(eligible, dtype=bool)
    if not candidate.any():
        candidate = valid
    omega_for_peak = np.where(candidate, np.abs(omega_deg), np.nan)
    pk = int(np.nanargmax(omega_for_peak))
    w = max(1, int(round(0.01 * fps)))
    lo, hi = max(0, pk - w), min(n, pk + w + 1)
    peak_deg = float(np.nanmean(np.abs(omega_deg)[lo:hi]))

    return AngularResult(
        t=t, angle_deg=angle_deg, omega_deg_s=omega_masked,
        omega_rad_s=np.radians(omega_masked), valid=valid,
        peak_index=pk, peak_time_s=float(t[pk]),
        peak_omega_deg_s=peak_deg, peak_omega_rad_s=float(np.radians(peak_deg)),
        peak_conf=float(pconf[pk]), cutoff_hz=float(cutoff_hz), kind=kind, side=side,
        meta={"peak_local_median_conf": float(np.median(pconf[lo:hi]))},
    )
