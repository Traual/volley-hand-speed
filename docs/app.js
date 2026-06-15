/* Volley Hand Speed — web app temps réel.
   Caméra -> MoveNet (TF.js, on-device) -> suivi du poignet -> vitesse + pic.
   Tout reste sur l'appareil ; aucune donnée envoyée. */

const $ = (id) => document.getElementById(id);
const KMH = 3.6;

const state = {
  hand: "right",
  heightM: 1.88,
  detector: null,
  stream: null,
  running: false,
  mPerPx: null,          // échelle calibrée (m/px), maintenue
  buf: [],               // échantillons récents du poignet {t, x, y}
  peak: 0,               // pic de vitesse (m/s) depuis le dernier reset
  emaSpeed: 0,
};

const VEL_WINDOW_MS = 45;   // fenêtre de calcul de vitesse (lisse le jitter)
const MAX_PLAUSIBLE = 40;   // m/s : au-delà = jitter, ignoré
const CONF = 0.3;           // seuil de confiance keypoint
const EYE_TO_ANKLE = 0.88;  // fraction stature (œil->cheville), pour la calibration

/* ---------- UI accueil ---------- */
document.querySelectorAll(".seg-btn").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.hand = b.dataset.hand;
  });
});
$("startBtn").addEventListener("click", start);
$("resetBtn").addEventListener("click", resetPeak);
$("stopBtn").addEventListener("click", stop);

function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }

/* ---------- Démarrage ---------- */
async function start() {
  $("homeErr").textContent = "";
  state.heightM = Math.min(2.2, Math.max(1.2, (parseFloat($("heightCm").value) || 188) / 100));
  try {
    $("home").classList.add("hidden");
    $("live").classList.remove("hidden");
    setStatus("Chargement du modèle…");

    if (!state.detector) {
      await tf.setBackend("webgl");
      await tf.ready();
      state.detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING });
    }

    setStatus("Accès caméra…");
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 },
               frameRate: { ideal: 60 } },
    });
    const video = $("video");
    video.srcObject = state.stream;
    await video.play();

    state.running = true;
    resetPeak();
    setStatus("Place-toi en entier dans le cadre…", "warn");
    requestAnimationFrame(loop);
  } catch (e) {
    $("live").classList.add("hidden");
    $("home").classList.remove("hidden");
    $("homeErr").textContent = "Erreur : " + (e && e.message ? e.message : e) +
      "\n(La caméra exige HTTPS et une autorisation.)";
  }
}

function stop() {
  state.running = false;
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("live").classList.add("hidden");
  $("home").classList.remove("hidden");
}

function resetPeak() {
  state.peak = 0; state.buf = []; state.emaSpeed = 0;
  $("peakVal").textContent = "0.0"; $("peakKmh").textContent = "0";
}

/* ---------- Helpers ---------- */
function kpMap(keypoints) {
  const m = {};
  for (const k of keypoints) m[k.name] = k;
  return m;
}

function calibrate(m) {
  // stature en px depuis œil/oreille (haut) -> cheville (bas), sur frames bien vues
  const heads = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    .map((n) => m[n]).filter((k) => k && k.score >= CONF);
  const ankles = ["left_ankle", "right_ankle"].map((n) => m[n]).filter((k) => k && k.score >= CONF);
  if (!heads.length || !ankles.length) return false;
  const topY = Math.min(...heads.map((k) => k.y));
  const botY = Math.max(...ankles.map((k) => k.y));
  const span = botY - topY;
  if (span < 60) return false;
  const staturePx = span / EYE_TO_ANKLE;
  const mpp = state.heightM / staturePx;
  // EMA pour stabiliser
  state.mPerPx = state.mPerPx ? 0.85 * state.mPerPx + 0.15 * mpp : mpp;
  return true;
}

function wristSpeed(now, wrist) {
  // vitesse sur une fenêtre ~45 ms (lisse le bruit image-à-image)
  state.buf.push({ t: now, x: wrist.x, y: wrist.y });
  while (state.buf.length > 2 && now - state.buf[0].t > 250) state.buf.shift();
  let ref = null;
  for (let i = state.buf.length - 2; i >= 0; i--) {
    if (now - state.buf[i].t >= VEL_WINDOW_MS) { ref = state.buf[i]; break; }
  }
  if (!ref) return null;
  const dt = (now - ref.t) / 1000;
  if (dt <= 0) return null;
  const dpx = Math.hypot(wrist.x - ref.x, wrist.y - ref.y);
  const vmps = (dpx / dt) * state.mPerPx;
  if (!isFinite(vmps) || vmps > MAX_PLAUSIBLE) return null;  // rejette le jitter
  return vmps;
}

/* ---------- Boucle ---------- */
let adj = null;
async function loop() {
  if (!state.running) return;
  const video = $("video"), cv = $("overlay"), ctx = cv.getContext("2d");
  if (video.readyState >= 2) {
    if (cv.width !== video.videoWidth) { cv.width = video.videoWidth; cv.height = video.videoHeight; }
    let poses = [];
    try { poses = await state.detector.estimatePoses(video, { flipHorizontal: false }); }
    catch (e) { /* ignore une frame ratée */ }

    ctx.clearRect(0, 0, cv.width, cv.height);
    if (poses && poses[0]) {
      const kps = poses[0].keypoints;
      const m = kpMap(kps);
      drawSkeleton(ctx, kps);

      const okCal = calibrate(m);
      const wrist = m[state.hand + "_wrist"];
      const now = performance.now();

      if (!state.mPerPx) {
        setStatus("Recule / cadre-toi en entier (tête → pieds) pour calibrer.", "warn");
      } else if (wrist && wrist.score >= CONF) {
        drawDot(ctx, wrist.x, wrist.y, "#ff3b30", 14);
        const v = wristSpeed(now, wrist);
        if (v != null) {
          state.emaSpeed = 0.6 * state.emaSpeed + 0.4 * v;
          $("liveVal").textContent = state.emaSpeed.toFixed(1);
          if (v > state.peak) {
            state.peak = v;
            $("peakVal").textContent = v.toFixed(1);
            $("peakKmh").textContent = Math.round(v * KMH);
          }
        }
        setStatus(okCal ? "Prêt — fais ton geste 🏐" : "Mesure en cours…", "ok");
      } else {
        state.buf = [];
        setStatus("Poignet " + (state.hand === "right" ? "droit" : "gauche") + " non détecté…", "warn");
      }
    } else {
      setStatus("Aucune personne détectée.", "warn");
    }
  }
  requestAnimationFrame(loop);
}

/* ---------- Dessin ---------- */
function drawDot(ctx, x, y, color, r) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = color; ctx.fill();
}
function drawSkeleton(ctx, kps) {
  if (!adj) adj = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);
  ctx.lineWidth = 4; ctx.strokeStyle = "rgba(54,211,153,.9)";
  for (const [a, b] of adj) {
    const p = kps[a], q = kps[b];
    if (p.score >= CONF && q.score >= CONF) {
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    }
  }
  for (const k of kps) if (k.score >= CONF) drawDot(ctx, k.x, k.y, "rgba(47,155,255,.95)", 5);
}
