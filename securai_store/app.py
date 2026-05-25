"""
Backend Flask — SecurAI Store
Inférence déportée sur Hugging Face Space (GPU).
Webcam locale, tunnel Cloudflare, multithreading.
"""
import os, cv2, time, numpy as np, threading, base64, requests, logging, queue
from dataclasses import dataclass, field
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

from modules.face_detector   import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.fgsm_attacker   import FGSMAttacker
from modules.patch_attacker  import PatchAttacker
from modules.defender        import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager
from paths import BASE_DIR, MODELS_DIR, ENROLLED_DIR, AUDIT_LOG

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
HF_BASE          = 'https://demimolchabite-securai-api.hf.space'
HF_INFER_URL     = f'{HF_BASE}/infer'
HF_FGSM_URL      = f'{HF_BASE}/fgsm'
HF_ENROLL_URL    = f'{HF_BASE}/enroll'

FRAME_SKIP       = 3       # envoie 1 frame sur 3 vers HF
FACE_CROP_SIZE   = 160
DEBUG            = False   # passe à True pour logs verbeux

app = Flask(__name__)
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=AUDIT_LOG,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("SecurAI demarré.")

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL thread-safe
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SystemState:
    identity:         str   = "Aucun"
    access_level:     str   = "DENIED"
    permissions:      dict  = field(default_factory=lambda: {
                                'entrance': False, 'stock': False,
                                'cashier': False, 'server': False})
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
# SESSION HTTP réutilisable (évite de re-négocier TLS à chaque requête)
# ─────────────────────────────────────────────────────────────────────────────
hf_session = requests.Session()
hf_session.headers.update({'Content-Type': 'application/json'})

# ─────────────────────────────────────────────────────────────────────────────
# MODULES IA
# ─────────────────────────────────────────────────────────────────────────────
print("[INIT] Chargement des modules IA...")
face_detector    = FaceDetector()
rights_manager   = RightsManager()
face_recognizer  = FaceRecognizer(mode='standard')
fgsm_attacker    = FGSMAttacker(face_recognizer.model, epsilon=0.03)
defender         = Defender()
anomaly_detector = AnomalyDetector()
patch_attacker   = PatchAttacker(face_recognizer.model, epsilon=0.35, steps=40, alpha=0.02)

# ─────────────────────────────────────────────────────────────────────────────
# ENRÔLEMENT AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────
print("[INIT] Enrôlement des visages connus...")
for filename in os.listdir(ENROLLED_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    identity_name = os.path.splitext(filename)[0].split('-')[0]
    img = cv2.imread(os.path.join(ENROLLED_DIR, filename))
    if img is None:
        continue
    bboxes = face_detector.detect(img)
    if not bboxes:
        continue
    face_crop = face_detector.crop_face(img, bboxes[0], size=FACE_CROP_SIZE)
    if face_crop is None:
        continue
    face_recognizer.enroll_face(identity_name, face_crop)
    # Sync vers HF Space
    try:
        _, buf = cv2.imencode('.jpg', face_crop)
        b64    = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        hf_session.post(HF_ENROLL_URL, json={"name": identity_name, "image": b64}, timeout=5)
        print(f"  [SYNC HF] {identity_name}")
    except Exception as e:
        print(f"  [SYNC HF ERREUR] {identity_name} : {e}")

print(f"[INIT] {len(face_recognizer.enrolled_embeddings)} identité(s) enrôlée(s).")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS : encodage image
# ─────────────────────────────────────────────────────────────────────────────
def _encode_b64(img: np.ndarray, quality: int = 85) -> str:
    """BGR ndarray → data URI base64 JPEG."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def _decode_b64(b64_str: str) -> np.ndarray | None:
    """data URI base64 → BGR ndarray. Retourne None si erreur."""
    try:
        raw    = base64.b64decode(b64_str.split(',')[-1])
        np_arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE DISTANTE — toutes les fonctions utilisent HF (plus de Colab)
# ─────────────────────────────────────────────────────────────────────────────
def send_to_hf_infer(face_crop: np.ndarray) -> dict | None:
    """
    Envoie un crop 160x160 vers HF /infer.
    Retourne dict {name, confidence, access} ou None si erreur.
    """
    try:
        t0   = time.time()
        resp = hf_session.post(
            HF_INFER_URL,
            json={"image": _encode_b64(face_crop)},
            timeout=6
        )
        ms = int((time.time() - t0) * 1000)
        resp.raise_for_status()
        result = resp.json()

        # Normalisation du nom et du champ access
        name           = result.get('name') or result.get('identity') or 'Inconnu'
        confidence     = float(result.get('confidence', 0.0))
        result['name'] = name
        result['access'] = 'DENIED' if name in ('Unknown', 'Inconnu', None) else 'GRANTED'

        if DEBUG:
            print(f"[HF INFER] {name} | conf={confidence:.2f} | {ms}ms")
        return result

    except requests.exceptions.Timeout:
        print("[HF INFER] Timeout")
        logging.warning("HF /infer timeout")
        return None
    except Exception as e:
        print(f"[HF INFER] Erreur : {e}")
        logging.error(f"HF /infer : {e}")
        return None


def send_to_hf_fgsm(face_crop: np.ndarray, target: str = "Manager_Demo") -> np.ndarray | None:
    """
    Envoie un crop vers HF /fgsm pour calcul FGSM sur GPU.
    Retourne le crop attaqué (ndarray BGR) ou None si erreur.

    FIX : le champ retourné peut s'appeler 'attacked_image' ou 'image'.
          On cherche les deux. Le champ 'name' du résultat FGSM est aussi
          logué correctement maintenant.
    """
    try:
        t0   = time.time()
        resp = hf_session.post(
            HF_FGSM_URL,
            json={"image": _encode_b64(face_crop), "target": target},
            timeout=20
        )
        ms = int((time.time() - t0) * 1000)
        resp.raise_for_status()
        result = resp.json()

        # Récupérer le nom reconnu (FIX : était None car mauvaise clé)
        recognized_as = result.get('name') or result.get('identity') or 'Inconnu'
        confidence    = result.get('confidence', 0.0)

        if DEBUG:
            print(f"[HF FGSM] {ms}ms | reconnu comme : {recognized_as} ({confidence:.4f})")

        # Récupérer l'image attaquée — accepte les deux noms de champ
        img_b64 = result.get('attacked_image') or result.get('image')
        if not img_b64:
            print("[HF FGSM] Aucun champ image dans la réponse")
            return None

        attacked = _decode_b64(img_b64)
        if attacked is None:
            print("[HF FGSM] Décodage de l'image attaquée échoué")
        return attacked

    except requests.exceptions.Timeout:
        print("[HF FGSM] Timeout")
        logging.warning("HF /fgsm timeout")
        return None
    except Exception as e:
        print(f"[HF FGSM] Erreur : {e}")
        logging.error(f"HF /fgsm : {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# THREAD HF WORKER — ne bloque JAMAIS le thread vidéo
# ─────────────────────────────────────────────────────────────────────────────
hf_queue = queue.Queue(maxsize=1)   # on garde 1 seul crop en attente

def _hf_worker():
    while True:
        try:
            face_crop = hf_queue.get(timeout=1)
            result    = send_to_hf_infer(face_crop)
            if result:
                name = result.get('name', 'Inconnu')
                with state_lock:
                    state.identity     = name
                    state.confidence   = result.get('confidence', 0.0)
                    state.access_level = result.get('access', 'DENIED')
                    state.permissions  = rights_manager.get_permissions(name)
        except queue.Empty:
            continue

threading.Thread(target=_hf_worker, daemon=True, name="hf-worker").start()

# ─────────────────────────────────────────────────────────────────────────────
# THREAD VIDÉO
# ─────────────────────────────────────────────────────────────────────────────
def _video_thread():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam inaccessible.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    fps_count   = 0
    fps_start   = time.time()

    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_count += 1
        fps_count   += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            with state_lock:
                state.fps = int(fps_count / elapsed)
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
                # FGSM déporté sur HF (toutes les FRAME_SKIP frames)
                if attack_active and frame_count % FRAME_SKIP == 0:
                    attacked = send_to_hf_fgsm(face_crop)
                    if attacked is not None:
                        face_crop = attacked

                # Défense locale si mode durci
                if current_mode == 'hardened':
                    face_crop = defender.apply_defense(face_crop, defense_type='gaussian')

                # Anomalie FFT local (~2ms)
                is_attacked, anom_score = anomaly_detector.analyze(face_crop)

                # Envoyer vers HF sans bloquer
                if frame_count % FRAME_SKIP == 0:
                    try:
                        hf_queue.put_nowait(face_crop)
                    except queue.Full:
                        pass  # on drop, pas de blocage

            # Overlay
            with state_lock:
                identity = state.identity
                conf     = state.confidence

            x1, y1, x2, y2 = bbox
            ui_cfg    = rights_manager.get_ui_config(identity)
            hex_color = ui_cfg['color'].lstrip('#')
            color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{identity} ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        with state_lock:
            state.anomaly_detected = is_attacked
            state.anomaly_score    = anom_score
            if is_attacked:
                logging.warning(f"ANOMALIE score={anom_score:.2f} identite={state.identity}")
            ok2, buf = cv2.imencode('.jpg', frame)
            if ok2:
                state.latest_frame = buf.tobytes()

threading.Thread(target=_video_thread, daemon=True, name="video-thread").start()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
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
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json or {}
    if 'active' not in data:
        return jsonify({"success": False, "error": "Parametre 'active' manquant."}), 400
    with state_lock:
        state.attack_active = bool(data['active'])
    status = "activée" if data['active'] else "désactivée"
    logging.info(f"Attaque {status}")
    return jsonify({"success": True, "message": f"Attaque {status}."})

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json or {}
    mode = data.get('mode', '')
    if mode not in ('standard', 'hardened'):
        return jsonify({"success": False, "error": "Mode invalide (standard|hardened)."}), 400
    try:
        face_recognizer.switch_mode(mode)
        with state_lock:
            state.model_mode = mode
        logging.info(f"Mode -> {mode}")
        return jsonify({"success": True, "message": f"Mode -> {mode}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes."}), 400
    file  = request.files['image']
    name  = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    if not file.filename:
        return jsonify({"success": False, "error": "Fichier vide."}), 400
    save_path = os.path.join(ENROLLED_DIR, secure_filename(f"{name}_{file.filename}"))
    file.save(save_path)
    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}."})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ── /api/analyze_static — FIX : send_crop_to_colab → send_to_hf_infer ────────
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

        bbox      = bboxes[0]
        face_crop = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        is_attacked, anom_score = anomaly_detector.analyze(face_crop)

        # FIX : send_to_hf_infer (plus send_crop_to_colab qui n'existe plus)
        result   = send_to_hf_infer(face_crop)
        identity = result.get('name', 'Inconnu') if result else 'Inconnu'
        conf     = float(result.get('confidence', 0.0)) if result else 0.0

        access_level = rights_manager.get_access_level(identity)
        permissions  = rights_manager.get_permissions(identity)

        x1, y1, x2, y2 = bbox
        ui_cfg    = rights_manager.get_ui_config(identity)
        hex_color = ui_cfg['color'].lstrip('#')
        color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        _, buf     = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buf).decode()

        return jsonify({"success": True, "results": {
            "identity":         identity,
            "confidence":       round(conf, 4),
            "access_level":     access_level,
            "permissions":      permissions,
            "anomaly_detected": is_attacked,
            "anomaly_score":    round(anom_score, 4),
            "image_base64":     img_base64,
        }})
    except Exception as e:
        logging.error(f"analyze_static: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ── /api/generate_glasses_attack — FIX : send_crop_to_colab → send_to_hf_infer
@app.route('/api/generate_glasses_attack', methods=['POST'])
def generate_glasses_attack():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Parametre 'image' manquant."}), 400
    target_name = request.form.get('target', 'Manager_Demo')
    target_emb  = face_recognizer.enrolled_embeddings.get(target_name)
    if target_emb is None:
        return jsonify({
            "success": False,
            "error":   f"'{target_name}' non enrôlé.",
            "enrolled": list(face_recognizer.enrolled_embeddings.keys())
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
            return jsonify({"success": False, "error": "Aucun visage détecté."}), 422

        bbox            = bboxes[0]
        x1, y1, x2, y2 = bbox
        face_crop       = face_detector.crop_face(img, bbox, size=FACE_CROP_SIZE)
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage."}), 500

        # FIX : send_to_hf_infer (plus send_crop_to_colab)
        r_before    = send_to_hf_infer(face_crop) or {}
        id_before   = r_before.get('name', 'Inconnu')
        conf_before = float(r_before.get('confidence', 0.0))

        attacked_crop = patch_attacker.attack(face_crop, target_emb)

        # FIX : send_to_hf_infer (plus send_crop_to_colab)
        r_after      = send_to_hf_infer(attacked_crop) or {}
        id_after     = r_after.get('name', 'Inconnu')
        conf_after   = float(r_after.get('confidence', 0.0))
        access_after = r_after.get('access', 'DENIED')

        is_anom, anom_score  = anomaly_detector.analyze(attacked_crop)
        permissions_after    = rights_manager.get_permissions(id_after)

        # Recoller le crop attaqué dans l'image originale
        result_img   = img.copy()
        img_h, img_w = img.shape[:2]
        rx1 = max(0, x1); ry1 = max(0, y1)
        rx2 = min(img_w, x2); ry2 = min(img_h, y2)
        result_img[ry1:ry2, rx1:rx2] = cv2.resize(attacked_crop, (rx2-rx1, ry2-ry1))

        ui_cfg    = rights_manager.get_ui_config(id_after)
        hex_color = ui_cfg['color'].lstrip('#')
        col_bgr   = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), col_bgr, 3)
        cv2.putText(result_img, f"{id_after} ({conf_after:.2f}) [PATCH]",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_bgr, 2)

        _, buf  = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode()

        logging.info(f"Patch: {id_before}({conf_before:.2f}) -> {id_after}({conf_after:.2f})")

        return jsonify({
            "success": True,
            "target":  target_name,
            "before":  {"identity": id_before, "confidence": round(conf_before, 4)},
            "after":   {
                "identity":         id_after,
                "confidence":       round(conf_after, 4),
                "access_level":     access_after,
                "permissions":      permissions_after,
                "anomaly_detected": is_anom,
                "anomaly_score":    round(anom_score, 4),
                "image_base64":     img_b64,
            }
        })
    except Exception as e:
        logging.error(f"generate_glasses_attack: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)