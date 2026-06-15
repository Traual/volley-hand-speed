"""Sorties : résumé console, CSV, graphe de vitesse (PNG), vidéo annotée."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # pas de GUI, on enregistre des PNG
import matplotlib.pyplot as plt  # noqa: E402

from .kinematics import SpeedResult
from .video import VideoInfo, read_frames


def _seq_verdict(lag_ms: float) -> tuple[str, str]:
    """Verdict de séquençage proximal->distal d'après le décalage épaule->poignet."""
    sense = "épaule avant poignet" if lag_ms > 0 else "poignet avant épaule"
    if 20 <= lag_ms <= 60:
        return sense, "BON séquençage (fouet)"
    if 0 < lag_ms < 20:
        return sense, "un peu simultané (sequence plus tôt l'épaule)"
    if lag_ms <= 0:
        return sense, "INVERSÉ — séquençage à corriger"
    return sense, "décalage large"


def print_summary(info: VideoInfo, result: SpeedResult, calib_source: str,
                  hand: str, warnings: list[str], reliable: bool = True,
                  angular=None, lag_ms: float | None = None,
                  whip: float | None = None, arm_len_m: float | None = None) -> None:
    print()
    print("=" * 60)
    print("  VITESSE DE MAIN — RÉSULTAT")
    print("=" * 60)
    if not reliable:
        print("  ⛔ RÉSULTAT NON FIABLE — chiffre à ne PAS utiliser (voir avertissements)")
        print("-" * 60)
    print(f"  Main mesurée        : poignet {hand}")
    print(f"  VITESSE DE POINTE   : {result.peak_speed_mps:5.1f} m/s   "
          f"({result.peak_speed_kmh:5.1f} km/h)"
          f"{'   (?)' if not reliable else ''}")
    print(f"  Instant du pic      : t = {result.peak_time_s*1000:.0f} ms "
          f"(frame {result.peak_index})")
    print(f"  Confiance au pic    : {result.peak_conf:.2f}")
    if angular is not None:
        print("-" * 60)
        print(f"  ROTATION ÉPAULE     : {angular.peak_omega_deg_s:5.0f} deg/s "
              f"({angular.peak_omega_rad_s:4.1f} rad/s)  @ t = {angular.peak_time_s*1000:.0f} ms")
        if lag_ms is not None:
            sense, verdict = _seq_verdict(lag_ms)
            print(f"  DÉCALAGE prox→dist  : {lag_ms:+.0f} ms ({sense})  [{verdict}]")
        if whip is not None:
            print(f"  RATIO FOUET         : {whip:.2f}   "
                  f"(poignet / épaule×bras {arm_len_m:.2f} m ; bon > 1)")
        print("  (« rotation épaule » = swing du bras dans le plan image,")
        print("   PAS la rotation interne 3D — non mesurable de profil.)")
    print("-" * 60)
    print(f"  Vrai fps (effectif) : {info.fps_effective:.1f} "
          f"(avg={info.fps_average}, base={info.fps_base})")
    print(f"  Frames décodées     : {info.n_frames}  | durée {info.duration_s*1000:.0f} ms")
    print(f"  Codec               : {info.codec}  | {info.width}x{info.height}")
    print(f"  Calibration         : {calib_source}")
    print(f"  Filtre              : {result.method}, coupure {result.cutoff_hz:.1f} Hz")
    print(f"  Frames interpolées  : {result.meta['frac_interpolated']*100:.1f} %")
    print("-" * 60)
    print("  Note : vitesse DANS LE PLAN IMAGE = borne inférieure de la vraie 3D.")
    print("         Le delta entre essais (même réglage) est la mesure fiable.")
    if warnings:
        print("-" * 60)
        for w in warnings:
            print(f"  ⚠  {w}")
    print("=" * 60)
    print()


def save_csv(path: str | Path, result: SpeedResult, x_raw: np.ndarray,
             y_raw: np.ndarray, conf: np.ndarray, angular=None) -> None:
    path = Path(path)
    has_ang = angular is not None
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["frame", "t_s", "x_raw_px", "y_raw_px", "conf",
                  "x_filt_px", "y_filt_px", "speed_mps", "valid"]
        if has_ang:
            header += ["shoulder_angle_deg", "shoulder_omega_deg_s"]
        w.writerow(header)
        for i in range(len(result.t)):
            row = [i, f"{result.t[i]:.6f}",
                   f"{x_raw[i]:.3f}" if np.isfinite(x_raw[i]) else "",
                   f"{y_raw[i]:.3f}" if np.isfinite(y_raw[i]) else "",
                   f"{conf[i]:.4f}",
                   f"{result.x_px[i]:.3f}", f"{result.y_px[i]:.3f}",
                   f"{result.speed_mps[i]:.4f}" if np.isfinite(result.speed_mps[i]) else "",
                   int(bool(result.valid[i]))]
            if has_ang:
                row += [f"{angular.angle_deg[i]:.3f}" if np.isfinite(angular.angle_deg[i]) else "",
                        f"{angular.omega_deg_s[i]:.2f}" if np.isfinite(angular.omega_deg_s[i]) else ""]
            w.writerow(row)


def save_plot(path: str | Path, result: SpeedResult, hand: str, angular=None) -> None:
    t_ms = result.t * 1000.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ms, result.speed_mps, lw=1.8, color="#1f77b4", label="vitesse poignet (m/s)")
    ax.axvline(result.peak_time_s * 1000.0, color="#1f77b4", ls="--", lw=1)
    ax.scatter([result.peak_time_s * 1000.0], [result.peak_speed_mps],
               color="#1f77b4", zorder=5)
    ax.annotate(f"{result.peak_speed_mps:.1f} m/s\n({result.peak_speed_kmh:.0f} km/h)",
                xy=(result.peak_time_s * 1000.0, result.peak_speed_mps),
                xytext=(8, -28), textcoords="offset points",
                fontsize=11, fontweight="bold", color="#1f77b4")
    ax.set_xlabel("temps (ms)")
    ax.set_ylabel("vitesse poignet (m/s)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.3)

    title = f"Poignet {hand} — filtre {result.method} {result.cutoff_hz:.1f} Hz"
    if angular is not None:
        ax2 = ax.twinx()
        omega = np.abs(angular.omega_deg_s)
        ax2.plot(t_ms, omega, lw=1.5, color="#ff7f0e", label="rotation épaule (deg/s)")
        ax2.axvline(angular.peak_time_s * 1000.0, color="#ff7f0e", ls="--", lw=1)
        ax2.scatter([angular.peak_time_s * 1000.0], [angular.peak_omega_deg_s],
                    color="#ff7f0e", zorder=5)
        ax2.annotate(f"{angular.peak_omega_deg_s:.0f} deg/s",
                     xy=(angular.peak_time_s * 1000.0, angular.peak_omega_deg_s),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=10, fontweight="bold", color="#ff7f0e")
        ax2.set_ylabel("rotation épaule (deg/s)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        lag = (result.peak_time_s - angular.peak_time_s) * 1000.0
        title += f"  |  épaule→poignet : {lag:+.0f} ms"
        lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    else:
        ax.legend(loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_annotated_video(path: str | Path, video_path: str,
                         x_px: np.ndarray, y_px: np.ndarray, valid: np.ndarray,
                         speed_mps: np.ndarray, peak_index: int,
                         peak_speed: float, out_fps: float = 30.0,
                         trail: int = 15) -> None:
    """Re-décode le clip et superpose : trajectoire, marqueur poignet, vitesse live.

    Écrit en out_fps (par défaut 30) pour un rendu regardable au ralenti.
    """
    path = Path(path)
    writer = None
    try:
        for i, _, rgb in read_frames(video_path):
            if i >= len(x_px):
                break
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, out_fps, (w, h))

            # Traînée
            for j in range(max(1, i - trail), i + 1):
                if valid[j] and valid[j - 1]:
                    p0 = (int(x_px[j - 1]), int(y_px[j - 1]))
                    p1 = (int(x_px[j]), int(y_px[j]))
                    cv2.line(bgr, p0, p1, (0, 200, 255), 3)
            # Marqueur poignet
            if valid[i]:
                cv2.circle(bgr, (int(x_px[i]), int(y_px[i])), 8, (0, 0, 255), -1)
            # Texte vitesse
            v = speed_mps[i]
            txt = f"v = {v:5.1f} m/s ({v*3.6:5.1f} km/h)" if np.isfinite(v) else "v = --"
            cv2.putText(bgr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(bgr, f"PIC: {peak_speed:.1f} m/s", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            if i == peak_index:
                cv2.putText(bgr, "<<< PIC", (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3, cv2.LINE_AA)
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()
