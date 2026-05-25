"""
SecurAI Store — app.py version GPU direct
A lancer sur le PC du pote (RTX 8Go).
Fait TOUT en local : détection, reconnaissance, FGSM.
Aucun appel réseau externe. Latence < 50ms.

Lance : python app_gpu_direct.py
"""
import os, cv2, time, numpy as np, threading, base64, logging, queue
from dataclasses import dataclass, field
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image as PILImage

from modules.face_detector    import FaceDetector
from modules.defender         import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager
from paths import BASE_DIR, MODELS_DIR, ENROLLED_DIR, AUDIT_LOG

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE — détection automatique GPU
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FACE_CROP_SIZE = 160
FGSM_EPSILON   = 0.03
FRAME_SKIP     = 1     # traite TOUTES les frames (GPU assez rapide)

app = Flask(__name__)
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — fichier + terminal
# ─────────────────────────────────────────────────────────────────────────────
_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
_fh  = logging.FileHandler(AUDIT_LOG)
_fh.setFormatter(_fmt); _fh.setLevel(logging.INFO)
_ch  = logging.StreamHandler()
_ch.setFormatter(_fmt); _ch.setLevel(logging.DEBUG)

log = logging.getLogger('securai')
log.setLevel(logging.DEBUG)
log.addHandler(_fh)
log.addHandler(_ch)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

log.info("=" * 55)
log.info("  SecurAI Store — GPU Direct")
log.info(f"  Device : {device}")
if device.type == 'cuda':
    log.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
    log.info(f"  VRAM   : {vram} Mo")
else:
    log.warning("  Aucun GPU détecté — mode CPU (lent)")
log.info("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES GPU
# ─────────────────────────────────────────────────────────────────────────────
log.info("Chargement MTCNN + FaceNet sur GPU...")
mtcnn   = MTCNN(
    image_size=FACE_CROP_SIZE,
    device=device,
    keep_all=False,
    post_process=False,
    margin=20
)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
log.info("Modèles GPU chargés.")

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE EMBEDDINGS (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
enrolled: dict[str, torch.Tensor] = {}
enrolled_lock = threading.Lock()

def _get_embedding(img_bgr: np.ndarray) -> torch.Tensor | None:
    """BGR numpy → embedding FaceNet (1, 512) sur GPU. Retourne None si pas de visage."""
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        face_t  = mtcnn(pil_img)               # (3, 160, 160) ou None
        if face_t is None:
            return None
        face_t  = face_t.unsqueeze(0).to(device)
        face_t  = (face_t / 255.0 - 0.5) / 0.5   # normalisation [-1, 1]
        with torch.no_grad():
            emb = facenet(face_t)              # (1, 512)
        return emb
    except Exception as e:
        log.error(f"_get_embedding : {e}")
        return None

def _identify(emb: torch.Tensor, threshold: float = 0.9) -> tuple[str, float]:
    """Compare un embedding avec le registre. Retourne (name, confidence)."""
    with enrolled_lock:
        if not enrolled:
            return 'Inconnu', 0.0
        best_name = 'Inconnu'
        best_dist = float('inf')
        for name, ref in enrolled.items():
            d = torch.dist(emb, ref).item()
            if d < best_dist:
                best_dist, best_name = d, name
    confidence = max(0.0, round(1.0 - best_dist / threshold, 4))
    if best_dist > threshold:
        return 'Inconnu', confidence
    return best_name, confidence

# ─────────────────────────────────────────────────────────────────────────────
# FGSM SUR GPU
# ─────────────────────────────────────────────────────────────────────────────
def _fgsm(img_bgr: np.ndarray, target_name: str,
          epsilon: float = FGSM_EPSILON) -> tuple[np.ndarray | None, str, float]:
    """
    Perturbation FGSM sur GPU.
    Retourne (attacked_bgr, recognized_name, confidence).
    """
    with enrolled_lock:
        target_emb = enrolled.get(target_name)
    if target_emb is None:
        log.warning(f"[FGSM] Cible '{target_name}' non enrôlée")
        return None, 'Inconnu', 0.0
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        face_t  = mtcnn(pil_img)
        if face_t is None:
            return None, 'Inconnu', 0.0

        face_t = face_t.unsqueeze(0).to(device)
        face_t = (face_t / 255.0 - 0.5) / 0.5
        face_t.requires_grad_(True)

        emb  = facenet(face_t)
        loss = torch.dist(emb, target_emb)   # minimise distance → ressemble à la cible
        loss.backward()

        with torch.no_grad():
            perturbed = face_t - epsilon * face_t.grad.sign()
            perturbed = torch.clamp(perturbed, -1, 1)
            new_emb   = facenet(perturbed)

        rec_name, rec_conf = _identify(new_emb)
        log.info(f"[FGSM] eps={epsilon} | {rec_name} conf={rec_conf:.4f}")

        # Convertir en numpy BGR
        p = perturbed.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        p = ((p * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(p, cv2.COLOR_RGB2BGR), rec_name, rec_conf

    except Exception as e:
        log.error(f"[FGSM] {e}")
        return None, 'Inconnu', 0.0

# ─────────────────────────────────────────────────────────────────────────────
# MODULES LOCAUX (pas GPU, très légers)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Chargement modules locaux...")
face_detector    = FaceDetector()
rights_manager   = RightsManager()
defender         = Defender()
anomaly_detector = AnomalyDetector()
log.info("Modules locaux prêts.")

# ─────────────────────────────────────────────────────────────────────────────
# ENRÔLEMENT AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Enrôlement depuis {ENROLLED_DIR}...")
for fname in os.listdir(ENROLLED_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    name = os.path.splitext(fname)[0].split('-')[0]
    img  = cv2.imread(os.path.join(ENROLLED_DIR, fname))
    if img is None:
        continue
    emb = _get_embedding(img)
    if emb is not None:
        with enrolled_lock:
            enrolled[name] = emb
        log.info(f"  Enrôlé : {name}")

log.info(f"{len(enrolled)} identité(s) enrôlée(s).")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SystemState:
    identity:         str   = "Aucun"
    access_level:     str   = "DENIED"
    permissions:      dict  = field(default_factory=lambda: {
                                'entrance': False, 'stock': False,
                                'cashier':  False, 'server': False})
    anomaly_detected: bool  = False
    anomaly_score:    float = 0.0
    attack_active:    bool  = False
    model_mode:       str   = "standard"
    fps:              int   = 0
    confidence:       float = 0.0
    latest_frame:     bytes = None

state      = SystemState()
state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _color_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

def _encode_b64(img: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def _decode_b64(b64_str: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64_str.split(',')[-1])
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# THREAD VIDÉO — inférence GPU directe dans le thread
# ─────────────────────────────────────────────────────────────────────────────
def _video_thread():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        log.error("Webcam inaccessible.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("Webcam ouverte (640x480).")

    frame_count = 0
    fps_count   = 0
    fps_start   = time.time()
    infer_times = []   # pour log latence GPU

    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_count += 1
        fps_count   += 1
        if time.time() - fps_start >= 1.0:
            fps_val = int(fps_count / (time.time() - fps_start))
            with state_lock:
                state.fps = fps_val
            if infer_times:
                avg_ms = int(sum(infer_times) / len(infer_times))
                log.debug(f"[VIDEO] FPS={fps_val} | latence GPU moy={avg_ms}ms")
                infer_times.clear()
            fps_count = 0
            fps_start = time.time()

        with state_lock:
            attack_active = state.attack_active
            current_mode  = state.model_mode

        bboxes      = face_detector.detect(frame)
        is_attacked = False
        anom_score  = 0.0

        if bboxes:
            bbox      = bboxes[0]
            face_crop = face_detector.crop_face(frame, bbox, size=FACE_CROP_SIZE)

            if face_crop is not None:
                # FGSM direct sur GPU
                if attack_active and frame_count % 3 == 0:
                    attacked, _, _ = _fgsm(face_crop, target_name="Manager_Demo")
                    if attacked is not None:
                        face_crop = attacked

                # Défense locale
                if current_mode == 'hardened':
                    face_crop = defender.apply_defense(face_crop, defense_type='gaussian')

                # Anomalie FFT (~2ms)
                is_attacked, anom_score = anomaly_detector.analyze(face_crop)
                if is_attacked:
                    log.warning(f"[ANOMALIE] score={anom_score:.2f}")

                # Inférence GPU directe
                t0  = time.time()
                emb = _get_embedding(face_crop)
                if emb is not None:
                    name, conf = _identify(emb)
                    access     = 'GRANTED' if name != 'Inconnu' else 'DENIED'
                    ms = int((time.time() - t0) * 1000)
                    infer_times.append(ms)

                    symbol = "✅" if access == 'GRANTED' else "🚫"
                    log.info(f"[GPU] {symbol} {name} | conf={conf:.2f} | {ms}ms")

                    with state_lock:
                        state.identity     = name
                        state.confidence   = conf
                        state.access_level = access
                        state.permissions  = rights_manager.get_permissions(name)

            # Overlay
            with state_lock:
                identity = state.identity
                conf     = state.confidence
                access   = state.access_level

            x1, y1, x2, y2 = bbox
            ui_cfg    = rights_manager.get_ui_config(identity)
            color_bgr = _color_bgr(ui_cfg['color'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{identity} ({conf:.2f}) [{access}]",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

        with state_lock:
            state.anomaly_detected = is_attacked
            state.anomaly_score    = anom_score
            ok2, buf = cv2.imencode('.jpg', frame)
            if ok2:
                state.latest_frame = buf.tobytes()

threading.Thread(target=_video_thread, daemon=True, name="video-thread").start()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES FLASK
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('EntranceControl.html')

@app.route('/static_analysis')
def static_analysis():
    return render_template('StaticAnalysis.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = state.latest_frame
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    with state_lock:
        return jsonify({
            'identity':         state.identity,
            'access_level':     state.access_level,
            'permissions':      state.permissions,
            'anomaly_detected': state.anomaly_detected,
            'anomaly_score':    round(state.anomaly_score, 4),
            'attack_active':    state.attack_active,
            'model_mode':       state.model_mode,
            'fps':              state.fps,
            'confidence':       round(state.confidence, 4),
            'device':           str(device),
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json or {}
    if 'active' not in data:
        return jsonify({"success": False, "error": "Paramètre 'active' manquant."}), 400
    with state_lock:
        state.attack_active = bool(data['active'])
    status = "activée" if data['active'] else "désactivée"
    log.info(f"[API] Attaque {status}")
    return jsonify({"success": True, "message": f"Attaque {status}."})

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json or {}
    mode = data.get('mode', '')
    if mode not in ('standard', 'hardened'):
        return jsonify({"success": False, "error": "Mode invalide."}), 400
    with state_lock:
        state.model_mode = mode
    log.info(f"[API] Mode -> {mode}")
    return jsonify({"success": True, "message": f"Mode -> {mode}"})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes."}), 400
    file  = request.files['image']
    name  = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400

    path = os.path.join(ENROLLED_DIR, secure_filename(f"{name}_{file.filename}"))
    file.save(path)

    img = cv2.imread(path)
    if img is not None:
        emb = _get_embedding(img)
        if emb is not None:
            with enrolled_lock:
                enrolled[name] = emb
            log.info(f"[ENROLL] {name} ajouté au registre GPU")

    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}."})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée."}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide."}), 400

        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({"success": True, "results": {
                "identity": "Aucun visage", "access_level": "DENIED",
                "confidence": 0.0, "anomaly_detected": False, "anomaly_score": 0.0
            }})

        face_crop = face_detector.crop_face(img, bboxes[0], size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        is_attacked, anom_score = anomaly_detector.analyze(face_crop)
        emb      = _get_embedding(face_crop)
        identity = 'Inconnu'
        conf     = 0.0
        if emb is not None:
            identity, conf = _identify(emb)

        x1, y1, x2, y2 = bboxes[0]
        color_bgr = _color_bgr(rights_manager.get_ui_config(identity)['color'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        _, buf = cv2.imencode('.jpg', img)
        log.info(f"[STATIC] {identity} conf={conf:.2f} anomalie={is_attacked}")
        return jsonify({"success": True, "results": {
            "identity":         identity,
            "confidence":       round(conf, 4),
            "access_level":     rights_manager.get_access_level(identity),
            "permissions":      rights_manager.get_permissions(identity),
            "anomaly_detected": is_attacked,
            "anomaly_score":    round(anom_score, 4),
            "image_base64":     base64.b64encode(buf).decode(),
        }})
    except Exception as e:
        log.error(f"[STATIC] {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate_glasses_attack', methods=['POST'])
def generate_glasses_attack():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Image manquante."}), 400
    target_name = request.form.get('target', 'Manager_Demo')
    with enrolled_lock:
        if target_name not in enrolled:
            return jsonify({
                "success": False,
                "error":   f"'{target_name}' non enrôlé.",
                "enrolled": list(enrolled.keys())
            }), 404

    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Format invalide."}), 400

        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({"success": False, "error": "Aucun visage."}), 422

        x1, y1, x2, y2 = bboxes[0]
        face_crop = face_detector.crop_face(img, bboxes[0], size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        # Avant attaque
        emb_before               = _get_embedding(face_crop)
        id_before, conf_before   = _identify(emb_before) if emb_before is not None else ('Inconnu', 0.0)

        # FGSM GPU
        attacked, id_after, conf_after = _fgsm(face_crop, target_name)
        if attacked is None:
            return jsonify({"success": False, "error": "FGSM échoué."}), 500

        access_after        = 'GRANTED' if id_after != 'Inconnu' else 'DENIED'
        is_anom, anom_score = anomaly_detector.analyze(attacked)

        result_img   = img.copy()
        img_h, img_w = img.shape[:2]
        rx1 = max(0, x1); ry1 = max(0, y1)
        rx2 = min(img_w, x2); ry2 = min(img_h, y2)
        result_img[ry1:ry2, rx1:rx2] = cv2.resize(attacked, (rx2-rx1, ry2-ry1))

        color_bgr = _color_bgr(rights_manager.get_ui_config(id_after)['color'])
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(result_img, f"{id_after} ({conf_after:.2f}) [PATCH]",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

        _, buf  = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode()

        log.info(f"[PATCH] {id_before}({conf_before:.2f}) -> {id_after}({conf_after:.2f})")
        return jsonify({
            "success": True, "target": target_name,
            "before": {"identity": id_before,  "confidence": round(conf_before, 4)},
            "after":  {
                "identity":         id_after,
                "confidence":       round(conf_after, 4),
                "access_level":     access_after,
                "permissions":      rights_manager.get_permissions(id_after),
                "anomaly_detected": is_anom,
                "anomaly_score":    round(anom_score, 4),
                "image_base64":     img_b64,
            }
        })
    except Exception as e:
        log.error(f"[PATCH] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    log.info(f"Flask sur http://0.0.0.0:5000 | device={device}")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
