# Volley Hand Speed 🏐⚡

Mesure la **vitesse de pointe de ton poignet** pendant un geste d'attaque au volley,
à partir d'une vidéo iPhone au ralenti. Objectif : comparer tes essais et voir si une
correction technique te fait gagner des m/s (frapper plus vite / plus fort).

Pose estimation **MoveNet Thunder (ONNX)** → suivi du poignet → traitement du signal
(filtrage zéro-phase + dérivée) → courbe de vitesse + pic.

> Tourne sur **Python 3.14 / Windows / GPU AMD** (RX 9070 XT) via ONNX Runtime DirectML.
> Pas de PyTorch, pas de MediaPipe (incompatibles 3.14 sur cette machine).

---

## Installation

```powershell
# depuis le dossier volley_hand_speed
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py   # télécharge MoveNet (~25 Mo)
.venv\Scripts\python.exe selftest.py         # vérifie que tout marche
```

## Utilisation

```powershell
# Le plus simple : calibration par ta taille, choix auto du poignet le plus rapide
.venv\Scripts\python.exe analyze.py geste.mov --height-cm 178

# Précis : objet de référence, clic des 2 extrémités sur la 1re frame
.venv\Scripts\python.exe analyze.py geste.mov --ref-length-m 1.0 --calibrate-click

# Forcer le vrai fps si l'auto-détection se trompe, et fixer la main
.venv\Scripts\python.exe analyze.py geste.mov --height-cm 178 --fps 240 --hand right
```

Sorties dans `out/<nom_du_clip>/` :
- `courbe_vitesse.png` — double axe : vitesse poignet (m/s) + **rotation d'épaule** (deg/s), les deux pics marqués
- `annotee.mp4` — ton clip avec poignet suivi + vitesse en direct (au ralenti 30 fps)
- `trajectoire_vitesse.csv` — données par frame (poignet + angle/vitesse angulaire d'épaule)

### Rotation d'épaule & séquençage (chaîne cinétique)
Le tool sort aussi la **vitesse angulaire du bras frappeur** (segment épaule→coude),
appelée « rotation d'épaule » — c'est le **swing du bras dans le plan image**, PAS la
rotation interne 3D (invisible de profil). À quoi ça sert :
- **Corroboration** : `rotation(rad/s) × longueur_bras ≈ vitesse_poignet`. Si les deux
  concordent, la mesure est fiable (l'épaule/coude sont mieux trackés que le poignet flou).
- **Effet fouet** : décalage **épaule→poignet** (l'épaule doit culminer ~20-60 ms AVANT le
  poignet) + **ratio fouet** = `vitesse_poignet / (rotation × bras)`. Ratio > 1 = le coude/
  poignet ajoutent de la vitesse par-dessus l'épaule (vrai fouet) ; ≈1 = swing « bras raide ».
- Attendu pour un spike : rotation ~1400-2100 deg/s.

---

## 🎥 Protocole de filmage (à respecter pour des chiffres exploitables)

La qualité de la mesure dépend **surtout du filmage**, pas du code.

1. **Ralenti 240 fps** (Réglages iPhone → Appareil photo → Enreg. au ralenti → 1080p 240).
   En 30 fps la main bouge trop entre 2 images → le pic est raté et **sous-estimé**.
2. **Caméra de profil**, perpendiculaire au plan du geste : la main doit se déplacer
   **haut/bas et gauche/droite** dans l'image, jamais vers/loin de l'objectif.
   (Une caméra ne mesure que le mouvement dans le plan image.)
3. **Objectif principal 1×** (26 mm). **Pas le 0.5× ultra-grand-angle** (distorsion).
4. **Trépied**, niveau activé (grille), à ~hauteur d'épaule, à **3–5 m** de recul.
   On doit te voir **en entier (pieds compris)** pendant tout le geste, et il te faut
   **de l'espace pour armer un vrai swing** (pas sous une suspension / au-dessus d'un lit).
   Le tool ignore tout seul les moments où tu poses/reprends le téléphone (gros plans).
5. **Lumière franche** (moins de flou de mouvement sur la main).
6. **Calibration** : filme une fois, dans le **même plan que ta main**, un objet de
   longueur connue (mètre, ou 2 repères à 1 m exactement). Sinon, donne ta taille
   (`--height-cm`, moins précis : ~±5–10 %).
7. **Transfère le clip ORIGINAL** (AirDrop du fichier non monté). Un export/montage
   peut « aplatir » le ralenti à 30 fps et **perdre** la résolution temporelle.

## Comment lire le résultat

- Le chiffre est la **vitesse dans le plan image** = **borne inférieure** de la vraie
  vitesse 3D (tout mouvement vers/loin de la caméra est invisible → jamais surestimé).
- **La mesure fiable, c'est le DELTA entre tes essais** à réglage identique. Un +3 m/s
  après un ajustement technique est crédible même si l'absolu est légèrement biaisé.
- Ne compare pas ton chiffre à des vitesses de *ballon* (radar) ou aux vidéos d'autres :
  réglages différents = comparaison absolue non fiable. Ordre de grandeur poignet d'un
  bon spike : ~15–25 m/s (~55–90 km/h).

---

## Détails techniques

| Brique | Choix | Pourquoi |
|---|---|---|
| Décodage vidéo | **PyAV** | vrai fps (≠ OpenCV qui renvoie souvent 30 sur ralenti iPhone) + HEVC fiable + timestamps réels |
| Rotation | **auto via DISPLAYMATRIX** | les vidéos iPhone portrait sont stockées en paysage + matrice de rotation que PyAV n'applique pas ; redressées sinon MoveNet voit une personne couchée |
| Fenêtrage du geste | **corps entier (tête→pieds) requis** | le pic n'est cherché que là où tu es filmé en pied → ignore approche/retour caméra |
| Garde-fou qualité | **jugé autour du pic** | fiabilité évaluée localement (détection au moment du pic), pas sur tout le clip |
| Pose | **MoveNet Thunder ONNX** (Apache-2.0) | régression mono-personne : pas de NMS/anchors → peu de jitter, idéal à dériver |
| Inférence | **onnxruntime-directml** | GPU AMD sous Windows (fallback CPU auto) |
| Filtrage | **Butterworth zéro-phase** (`sosfiltfilt`), ~12 Hz | standard biomécanique ; coupure calée sur la littérature lancer/pitching (pas les 6 Hz de la marche qui écraseraient le pic) ; analyse résiduelle de Winter pour confirmer |
| Dérivée | `np.gradient` (différence centrée) | O(dt²), aucun décalage temporel du pic |

Repli (fallback) : `--method savgol` (Savitzky-Golay, lisse+dérive en une passe).
Modèle de secours possible : YOLOv8-pose ONNX (plus robuste si la scène est encombrée,
mais licence AGPL et décodage plus lourd).

### Structure

```
analyze.py            # CLI principal
download_model.py     # télécharge le modèle
selftest.py           # tests (kinematique, pose I/O, vidéo)
validate_on_image.py  # vérif visuelle de la pose sur une image fixe
vhs/
  video.py        # décodage PyAV + détection du vrai fps
  pose.py         # MoveNet ONNX (préproc, inférence, décodage keypoints)
  kinematics.py   # gating, interpolation, filtrage, dérivée, pic
  calibrate.py    # échelle px→m (objet de référence / taille)
  report.py       # résumé console, CSV, plot, vidéo annotée
models/movenet_thunder.onnx
```

## Limites connues

- Une seule caméra → vitesse 2D dans le plan (voir « Comment lire »).
- Flou de mouvement au pic → la confiance du keypoint baisse pile au moment clé ;
  atténué par le 240 fps + le lissage (le pic est lu sur la courbe filtrée).
- Calibration par la taille = approximative ; privilégier l'objet de référence pour
  l'absolu.

## Idée suite (optionnel)

Une app iOS (Swift + Vision) pour un retour **temps réel** du dernier pic sur le
téléphone — mais ironiquement **moins précise** que ce clip-analyseur (caméra live
~30–60 fps vs 240 fps en ralenti).
