#!/usr/bin/env python
"""Analyse de clip — vitesse du poignet pour le geste d'attaque au volley.

Exemples :
  # Calibration la plus simple (par ta taille), poignet droit :
  python analyze.py geste.mov --height-cm 178

  # Calibration précise par objet de référence (clic sur 2 points) :
  python analyze.py geste.mov --ref-length-m 1.0 --calibrate-click

  # Objet de référence dont tu connais déjà la longueur pixel :
  python analyze.py geste.mov --ref-length-m 1.0 --ref-px 480

  # Forcer le fps (si l'auto-détection se trompe) et choisir la main :
  python analyze.py geste.mov --height-cm 178 --fps 240 --hand left
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):           # console Windows -> UTF-8 (accents, ⚠, ≈)
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from vhs import calibrate, kinematics, pose, report, video

DEFAULT_MODEL = Path(__file__).parent / "models" / "movenet_thunder.onnx"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mesure la vitesse de pointe du poignet (geste d'attaque volley) "
                    "depuis une vidéo iPhone au ralenti.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Clip vidéo (idéalement ralenti 240 fps iPhone)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Chemin du modèle ONNX MoveNet")
    p.add_argument("--hand", choices=["right", "left", "auto"], default="auto",
                   help="Poignet à mesurer ('auto' = le plus rapide des deux)")

    cal = p.add_argument_group("Calibration pixels -> mètres (choisir une méthode)")
    cal.add_argument("--height-cm", type=float, help="Ta taille en cm (auto-calibration)")
    cal.add_argument("--ref-length-m", type=float, help="Longueur réelle de l'objet de référence (m)")
    cal.add_argument("--ref-px", type=float, help="Longueur de cet objet en pixels (si déjà mesurée)")
    cal.add_argument("--calibrate-click", action="store_true",
                     help="Cliquer les 2 extrémités de l'objet de référence sur la 1re frame")
    cal.add_argument("--scale-m-per-px", type=float, help="Échelle directe (m/px), si déjà connue")

    sig = p.add_argument_group("Traitement du signal")
    sig.add_argument("--fps", type=float, help="Forcer le vrai fps (sinon auto-détecté)")
    sig.add_argument("--method", choices=["butter", "savgol"], default="butter")
    sig.add_argument("--cutoff", type=float, help="Coupure passe-bas (Hz) ; auto si omis (8-18 Hz)")
    sig.add_argument("--shoulder-cutoff", type=float, help="Coupure pour l'angle d'épaule (Hz) ; auto 6-14")
    sig.add_argument("--conf-thresh", type=float, default=0.3, help="Seuil de confiance du keypoint")
    sig.add_argument("--max-gap-ms", type=float, default=50.0, help="Trou max interpolé (ms)")
    sig.add_argument("--start", type=float, help="Ne chercher le pic qu'après cet instant (s)")
    sig.add_argument("--end", type=float, help="Ne chercher le pic qu'avant cet instant (s)")

    out = p.add_argument_group("Sorties")
    out.add_argument("--outdir", help="Dossier de sortie (défaut: out/<nom_du_clip>)")
    out.add_argument("--no-video", action="store_true", help="Ne pas générer la vidéo annotée")
    out.add_argument("--video-fps", type=float, default=30.0, help="fps de la vidéo annotée")
    return p.parse_args(argv)


def resolve_calibration(args, keypoints, first_frame):
    if args.scale_m_per_px:
        return calibrate.Calibration(args.scale_m_per_px, "échelle directe (m/px)", {})
    if args.ref_length_m and args.ref_px:
        return calibrate.from_reference_pixels(args.ref_length_m, args.ref_px)
    if args.ref_length_m and args.calibrate_click:
        return calibrate.pick_reference_interactive(first_frame, args.ref_length_m)
    if args.height_cm:
        return calibrate.from_height(keypoints, args.height_cm / 100.0, args.conf_thresh)
    raise SystemExit(
        "Calibration manquante. Donne UNE des options :\n"
        "  --height-cm 178                      (simple, auto)\n"
        "  --ref-length-m 1.0 --calibrate-click (précis, clic)\n"
        "  --ref-length-m 1.0 --ref-px 480      (précis, déjà mesuré)\n"
        "  --scale-m-per-px 0.00208             (échelle connue)")


def main(argv=None):
    args = parse_args(argv)
    vpath = args.video
    if not Path(vpath).exists():
        raise SystemExit(f"Fichier introuvable : {vpath}")
    if not Path(args.model).exists():
        raise SystemExit(f"Modèle introuvable : {args.model}\n"
                         "Lance d'abord : python download_model.py")

    # 1. Pose — UNE seule passe de décodage (keypoints + timestamps)
    print("Chargement du modèle MoveNet…")
    model = pose.MoveNetThunder(args.model)
    print(f"  provider actif : {model.active_provider}")
    meta = video.read_meta(vpath)

    print("Analyse des frames (pose)…")
    kps: list[np.ndarray] = []
    times: list[float | None] = []
    first_frame = None
    for i, t, rgb in video.read_frames(vpath):
        if first_frame is None:
            first_frame = rgb
        kps.append(model.infer(rgb))
        times.append(t)
        if i % 50 == 0:
            print(f"\r  frame {i}…", end="", flush=True)
    print()
    if not kps:
        raise SystemExit("Aucune frame décodée (codec ? fichier corrompu ?).")
    keypoints = np.asarray(kps, dtype=float)           # [N,17,3]
    n = len(keypoints)

    info = video.make_info(meta, times, n, *first_frame.shape[1::-1])

    # Vecteur temps : PTS réels si dispo, sinon uniforme au fps choisi
    fps = args.fps or info.fps_effective
    if all(t is not None for t in times) and not args.fps:
        t_arr = np.asarray(times, dtype=float)
    else:
        t_arr = np.arange(n, dtype=float) / fps

    warnings: list[str] = []
    if info.likely_baked and not args.fps:
        warnings.append(f"fps détecté ≈ {fps:.0f} : le ralenti a peut-être été 'aplati' "
                        "à la vitesse normale (ré-encodage/AirDrop). Renvoie le clip ORIGINAL "
                        "non monté, ou force --fps 240. Le pic sera sous-résolu sinon.")
    if model._dtype == np.float32:
        warnings.append("Entrée du modèle en float32 (export inattendu) : vérifie les valeurs.")

    # 2. Calibration
    calib = resolve_calibration(args, keypoints, first_frame)

    # Masque "torse visible" = épaules ET hanches détectées -> personne filmée à distance
    # (pas un gros plan approche/retour). On NE PAS exiger les pieds : ils décrochent
    # souvent pile au moment explosif du swing, ce qui excluait le vrai pic. Le pic est
    # de toute façon verrouillé sur une frame où le POIGNET est bien détecté (conf>=0.5).
    sh = (keypoints[:, pose.LEFT_SHOULDER, 2] >= 0.3) | (keypoints[:, pose.RIGHT_SHOULDER, 2] >= 0.3)
    hip = (keypoints[:, pose.LEFT_HIP, 2] >= 0.3) | (keypoints[:, pose.RIGHT_HIP, 2] >= 0.3)
    eligible = sh & hip
    if args.start is not None:
        eligible &= t_arr >= args.start
    if args.end is not None:
        eligible &= t_arr <= args.end
    if eligible.sum() >= 5:
        tw = t_arr[eligible]
        print(f"Fenêtre torse-visible : {int(eligible.sum())} frames "
              f"entre t={tw.min():.1f}s et t={tw.max():.1f}s")
    else:
        warnings.append("Aucune fenêtre 'torse visible' : pic cherché sur tout le clip.")
        eligible = None

    # 3. Vitesse — calcule les deux poignets pour pouvoir choisir/afficher
    def speed_for(hand):
        idx = pose.WRIST_INDEX[hand]
        return kinematics.compute_speed(
            keypoints[:, idx, 0], keypoints[:, idx, 1], keypoints[:, idx, 2],
            t_arr, calib.m_per_px, fps,
            conf_thresh=args.conf_thresh, max_gap_s=args.max_gap_ms / 1000.0,
            cutoff_hz=args.cutoff, method=args.method, eligible=eligible)

    results = {}
    for hand in ("right", "left"):
        try:
            results[hand] = speed_for(hand)
        except ValueError as e:
            warnings.append(f"poignet {hand} : {e}")

    if not results:
        raise SystemExit("Aucun poignet exploitable. Vérifie le cadrage (corps + bras visibles).")

    if args.hand == "auto":
        hand = max(results, key=lambda h: results[h].peak_speed_mps)
    elif args.hand in results:
        hand = args.hand
    else:
        raise SystemExit(f"Poignet {args.hand} non exploitable dans ce clip.")
    result = results[hand]

    # 3b. Garde-fou qualité
    reliable, quality_warns = kinematics.assess_quality(keypoints, result, t_arr)
    warnings.extend(quality_warns)

    # 3c. Rotation d'épaule (cadence de swing du bras frappeur) + séquençage chaîne
    angular = None
    lag_ms = whip = arm_len_m = None
    try:
        angular = kinematics.compute_joint_angle(
            keypoints, t_arr, fps, side=hand, kind="upperarm",
            conf_thresh=args.conf_thresh, max_gap_s=args.max_gap_ms / 1000.0,
            cutoff_hz=args.shoulder_cutoff, eligible=eligible)
    except ValueError as e:
        warnings.append(f"rotation épaule: {e}")
    if angular is not None:
        lag_ms = (result.peak_time_s - angular.peak_time_s) * 1000.0  # >0 = épaule avant poignet
        SHk = pose.LEFT_SHOULDER if hand == "left" else pose.RIGHT_SHOULDER
        WRk = pose.LEFT_WRIST if hand == "left" else pose.RIGHT_WRIST
        g = (keypoints[:, SHk, 2] >= 0.5) & (keypoints[:, WRk, 2] >= 0.5)
        if g.sum() >= 5:
            arm_len_m = float(np.median(np.hypot(
                keypoints[g, SHk, 0] - keypoints[g, WRk, 0],
                keypoints[g, SHk, 1] - keypoints[g, WRk, 1]))) * calib.m_per_px
        else:
            arm_len_m = 0.55
        if angular.peak_omega_rad_s > 0:
            whip = result.peak_speed_mps / (angular.peak_omega_rad_s * arm_len_m)
        deg = angular.peak_omega_deg_s
        if deg < 300 or deg > 2500:
            warnings.append(f"rotation épaule {deg:.0f} deg/s hors plage plausible (300-2500) "
                            "— suspecte du jitter ou une détection instable.")

    # 4. Sorties
    outdir = Path(args.outdir) if args.outdir else Path("out") / Path(vpath).stem
    outdir.mkdir(parents=True, exist_ok=True)

    report.print_summary(info, result, calib.source, hand, warnings, reliable,
                         angular=angular, lag_ms=lag_ms, whip=whip, arm_len_m=arm_len_m)
    if "right" in results and "left" in results:
        other = "left" if hand == "right" else "right"
        print(f"  (poignet {other} : pic {results[other].peak_speed_mps:.1f} m/s)\n")

    idx = pose.WRIST_INDEX[hand]
    report.save_csv(outdir / "trajectoire_vitesse.csv", result,
                    keypoints[:, idx, 0], keypoints[:, idx, 1], keypoints[:, idx, 2],
                    angular=angular)
    report.save_plot(outdir / "courbe_vitesse.png", result, hand, angular=angular)
    print(f"  CSV   -> {outdir / 'trajectoire_vitesse.csv'}")
    print(f"  Plot  -> {outdir / 'courbe_vitesse.png'}")

    if not args.no_video:
        print("  Génération de la vidéo annotée…")
        report.save_annotated_video(
            outdir / "annotee.mp4", vpath, result.x_px, result.y_px, result.valid,
            result.speed_mps, result.peak_index, result.peak_speed_mps,
            out_fps=args.video_fps)
        print(f"  Vidéo -> {outdir / 'annotee.mp4'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
