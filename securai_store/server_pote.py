"""
SecurAI — Serveur d'inférence GPU
A faire tourner sur le PC du pote (RTX 8Go).
Lance : python server_pote.py
Expose ensuite via Cloudflare Tunnel.
"""
import os, base64, time, logging, threading
import numpy as np
import torch
import cv2
from flask import Flask, request, jsonify
from facenet_pytorch import InceptionResnetV1, MTCNN

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING terminal
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('securai-gpu')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PORT           = 7860
FACE_CROP_SIZE = 160
FGSM_EPSILON   = 0.03
# Dossier contenant les photos des identités enrôlées
# Structure : enrolled/Manager_Demo.jpg, enrolled/Employee_Demo.jpg ...
ENROLLED_DIR   = './enrolled'

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Device : {device}")
if device.type == 'cuda':
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**2} Mo")

# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
log.info("Chargement MTCNN + FaceNet...")
mtcnn    = MTCNN(image_size=FACE_CROP_SIZE, device=device, keep_all=False, post_process=False)
facenet  = InceptionResnetV1(pretrained='vggface2').eval().to(device)
log.info("Modèles chargés.")

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE D'EMBEDDINGS (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
enrolled_embeddings: dict[str, torch.Tensor] = {}
enrolled_lock = threading.Lock()

def _get_embedding(img_rgb: np.ndarray) -> torch.Tensor | None:
    """
    Prend une image RGB numpy, détecte + aligne le visage via MTCNN,
    calcule l'embedding FaceNet. Retourne un tenseur (1, 512) ou None.
    """
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_rgb)
        face_tensor = mtcnn(pil_img)   # (3, 160, 160) ou None
        if face_tensor is None:
            return None
        face_tensor = face_tensor.unsqueeze(0).to(device)  # (1, 3, 160, 160)
        face_tensor = (face_tensor / 255.0 - 0.5) / 0.5   # normalisation [-1, 1]
        with torch.no_grad():
            emb = facenet(face_tensor)  # (1, 512)
        return emb
    except Exception as e:
        log.error(f"_get_embedding : {e}")
        return None

def _identify(emb: torch.Tensor, threshold: float = 0.9) -> tuple[str, float]:
    """
    Compare un embedding avec le registre.
    Retourne (name, confidence) ou ('Inconnu', distance_min).
    """
    with enrolled_lock:
        if not enrolled_embeddings:
            return 'Inconnu', 0.0
        best_name = 'Inconnu'
        best_dist = float('inf')
        for name, ref_emb in enrolled_embeddings.items():
            dist = torch.dist(emb, ref_emb).item()
            if dist < best_dist:
                best_dist = dist
                best_name = name
    # distance → confidence [0, 1] (plus c'est proche de 0, plus c'est confiant)
    confidence = max(0.0, 1.0 - best_dist / threshold)
    if best_dist > threshold:
        return 'Inconnu', round(confidence, 4)
    return best_name, round(confidence, 4)

# ─────────────────────────────────────────────────────────────────────────────
# ENRÔLEMENT AU DÉMARRAGE depuis ENROLLED_DIR
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(ENROLLED_DIR, exist_ok=True)

def _enroll_from_dir():
    count = 0
    for fname in os.listdir(ENROLLED_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        name = os.path.splitext(fname)[0].split('-')[0]
        img  = cv2.imread(os.path.join(ENROLLED_DIR, fname))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = _get_embedding(img_rgb)
        if emb is not None:
            with enrolled_lock:
                enrolled_embeddings[name] = emb
            log.info(f"  Enrôlé : {name}")
            count += 1
    log.info(f"{count} identité(s) enrôlée(s) depuis {ENROLLED_DIR}/")

_enroll_from_dir()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _decode_b64(b64_str: str) -> np.ndarray | None:
    try:
        raw    = base64.b64decode(b64_str.split(',')[-1])
        np_arr = np.frombuffer(raw, np.uint8)
        bgr    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        log.error(f"_decode_b64 : {e}")
        return None

def _encode_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# ─────────────────────────────────────────────────────────────────────────────
# FGSM SUR GPU
# ─────────────────────────────────────────────────────────────────────────────
def _fgsm_attack(img_rgb: np.ndarray, target_name: str, epsilon: float = FGSM_EPSILON):
    """
    Calcule une perturbation FGSM sur GPU pour faire confondre img_rgb
    avec target_name. Retourne (attacked_img_bgr, recognized_name, confidence).
    """
    target_emb = enrolled_embeddings.get(target_name)
    if target_emb is None:
        log.warning(f"[FGSM] Cible '{target_name}' non enrôlée")
        return None, 'Inconnu', 0.0

    try:
        from PIL import Image as PILImage
        pil_img     = PILImage.fromarray(img_rgb)
        face_tensor = mtcnn(pil_img)
        if face_tensor is None:
            return None, 'Inconnu', 0.0

        face_tensor = face_tensor.unsqueeze(0).to(device)
        face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
        face_tensor.requires_grad_(True)

        emb  = facenet(face_tensor)
        loss = torch.dist(emb, target_emb)  # minimiser la distance → ressembler à la cible
        loss.backward()

        with torch.no_grad():
            perturbed = face_tensor - epsilon * face_tensor.grad.sign()  # signe inversé
            perturbed = torch.clamp(perturbed, -1, 1)

        # Re-identifier
        new_emb              = facenet(perturbed)
        rec_name, rec_conf   = _identify(new_emb)

        # Convertir en image numpy BGR
        p_np  = perturbed.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        p_np  = ((p_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        p_bgr = cv2.cvtColor(p_np, cv2.COLOR_RGB2BGR)

        log.info(f"[FGSM] epsilon={epsilon} | résultat : {rec_name} (conf={rec_conf:.4f})")
        return p_bgr, rec_name, rec_conf

    except Exception as e:
        log.error(f"[FGSM] Erreur : {e}")
        return None, 'Inconnu', 0.0

# ─────────────────────────────────────────────────────────────────────────────
# FLASK API
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Compteurs pour les logs
_req_count = {'infer': 0, 'fgsm': 0, 'enroll': 0}

@app.route('/health', methods=['GET'])
def health():
    with enrolled_lock:
        n = len(enrolled_embeddings)
    return jsonify({
        'status':    'ok',
        'device':    str(device),
        'gpu':       torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none',
        'enrolled':  n,
        'requests':  _req_count,
    })

@app.route('/infer', methods=['POST'])
def infer():
    """
    Body JSON : { "image": "<data URI base64>" }
    Retourne   : { "name", "confidence", "access" }
    """
    _req_count['infer'] += 1
    t0 = time.time()

    payload = request.get_json(silent=True) or {}
    if 'image' not in payload:
        return jsonify({'error': "Champ 'image' manquant"}), 400

    img_rgb = _decode_b64(payload['image'])
    if img_rgb is None:
        return jsonify({'error': 'Décodage image échoué'}), 400

    emb = _get_embedding(img_rgb)
    if emb is None:
        log.info(f"[INFER #{_req_count['infer']}] Aucun visage détecté")
        return jsonify({'name': 'Inconnu', 'confidence': 0.0, 'access': 'DENIED'})

    name, confidence = _identify(emb)
    access = 'GRANTED' if name != 'Inconnu' else 'DENIED'
    ms = int((time.time() - t0) * 1000)

    symbol = "✅" if access == 'GRANTED' else "🚫"
    log.info(f"[INFER #{_req_count['infer']}] {symbol} {name} | conf={confidence:.2f} | {ms}ms")

    return jsonify({'name': name, 'confidence': confidence, 'access': access})

@app.route('/fgsm', methods=['POST'])
def fgsm():
    """
    Body JSON : { "image": "<data URI base64>", "target": "Manager_Demo", "epsilon": 0.03 }
    Retourne   : { "attacked_image": "<data URI>", "name", "confidence" }
    """
    _req_count['fgsm'] += 1
    t0 = time.time()

    payload = request.get_json(silent=True) or {}
    if 'image' not in payload:
        return jsonify({'error': "Champ 'image' manquant"}), 400

    img_rgb     = _decode_b64(payload['image'])
    target_name = payload.get('target', 'Manager_Demo')
    epsilon     = float(payload.get('epsilon', FGSM_EPSILON))

    if img_rgb is None:
        return jsonify({'error': 'Décodage image échoué'}), 400

    attacked_bgr, rec_name, rec_conf = _fgsm_attack(img_rgb, target_name, epsilon)
    ms = int((time.time() - t0) * 1000)

    if attacked_bgr is None:
        return jsonify({'error': 'FGSM échoué (visage non détecté ou cible inconnue)'}), 422

    log.info(f"[FGSM #{_req_count['fgsm']}] {ms}ms | {rec_name} (conf={rec_conf:.4f})")
    return jsonify({
        'attacked_image': _encode_b64(attacked_bgr),
        'name':           rec_name,
        'confidence':     rec_conf,
    })

@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Body JSON : { "name": "Manager_Demo", "image": "<data URI base64>" }
    Ajoute l'identité au registre en mémoire + sauvegarde la photo.
    """
    _req_count['enroll'] += 1
    payload = request.get_json(silent=True) or {}
    name    = payload.get('name', '').strip()
    if not name or 'image' not in payload:
        return jsonify({'error': "Champs 'name' et 'image' requis"}), 400

    img_rgb = _decode_b64(payload['image'])
    if img_rgb is None:
        return jsonify({'error': 'Décodage image échoué'}), 400

    emb = _get_embedding(img_rgb)
    if emb is None:
        return jsonify({'error': 'Aucun visage détecté dans l image'}), 422

    with enrolled_lock:
        enrolled_embeddings[name] = emb

    # Sauvegarde photo sur disque
    save_path = os.path.join(ENROLLED_DIR, f"{name}.jpg")
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)

    log.info(f"[ENROLL] {name} enrôlé et sauvegardé -> {save_path}")
    return jsonify({'success': True, 'name': name})

@app.route('/enrolled', methods=['GET'])
def list_enrolled():
    with enrolled_lock:
        names = list(enrolled_embeddings.keys())
    return jsonify({'enrolled': names, 'count': len(names)})

# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info(f"Serveur GPU SecurAI sur http://0.0.0.0:{PORT}")
    log.info("Lance ensuite : cloudflared tunnel --url http://localhost:7860")
    # threaded=True pour gérer plusieurs requêtes simultanées (multi-workers ton PC)
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)
