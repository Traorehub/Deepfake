"""
Backend Flask pour SecurAI Store.
Expose le flux vidéo MJPEG et les API de contrôle avec multithreading.
"""
import os
import cv2
import time
import numpy as np
import threading
from dataclasses import dataclass
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import des modules d'intelligence
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.fgsm_attacker import FGSMAttacker
from modules.defender import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ENROLLED_DIR = os.path.join(BASE_DIR, 'data', 'enrolled')
os.makedirs(ENROLLED_DIR, exist_ok=True)

# --- ÉTAT GLOBAL (Thread-safe) ---
@dataclass
class SystemState:
    identity: str = "Aucun"
    access_level: str = "DENIED"
    permissions: dict = None
    anomaly_detected: bool = False
    anomaly_score: float = 0.0
    attack_active: bool = False
    model_mode: str = "standard"
    fps: int = 0
    confidence: float = 0.0
    latest_frame: np.ndarray = None

state = SystemState()
state.permissions = {'entrance': False, 'stock': False, 'cashier': False, 'server': False}
state_lock = threading.Lock()

# --- INITIALISATION DES MODULES ---
print("Initialisation des modules IA en cours...")
face_detector = FaceDetector()
rights_manager = RightsManager()

# On ne charge plus class_names.json car on utilise les embeddings
face_recognizer = FaceRecognizer(mode='standard')
fgsm_attacker = FGSMAttacker(face_recognizer.model, epsilon=0.03)

# Auto-enrôlement des images présentes dans data/enrolled/
print("Enrôlement des visages connus...")
for filename in os.listdir(ENROLLED_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Le nom complet sans extension (ex: Manager_Demo)
        # On ignore ce qui suit un tiret '-' pour permettre plusieurs photos (ex: Manager_Demo-1.jpg)
        base_name = os.path.splitext(filename)[0]
        identity_name = base_name.split('-')[0]
        
        filepath = os.path.join(ENROLLED_DIR, filename)
        img = cv2.imread(filepath)
        if img is not None:
            # On détecte le visage dans l'image
            bboxes = face_detector.detect(img)
            if bboxes:
                face_crop = face_detector.crop_face(img, bboxes[0], size=160)
                if face_crop is not None:
                    face_recognizer.enroll_face(identity_name, face_crop)
                    
print(f"{len(face_recognizer.enrolled_embeddings)} identités enrôlées.")

defender = Defender()
anomaly_detector = AnomalyDetector()
print("Modules prêts.")

# --- THREAD DE TRAITEMENT VIDEO ---
def video_processing_thread():
    camera = cv2.VideoCapture(0)
    # Résolution 640x480
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
            
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            with state_lock:
                state.fps = int(frame_count / elapsed)
            frame_count = 0
            start_time = time.time()
            
        bboxes = face_detector.detect(frame)
        
        current_identity = "Inconnu"
        current_conf = 0.0
        is_attacked = False
        anom_score = 0.0
        
        with state_lock:
            attack_active = state.attack_active
            current_mode = state.model_mode
            
        for bbox in bboxes:
            face_crop = face_detector.crop_face(frame, bbox, size=128)
            if face_crop is None or face_recognizer is None:
                continue
                
            # Attaque
            if attack_active and frame_count % 3 == 0:
                # Si on connaît l'embedding du Manager_Demo, on fait une attaque ciblée vers lui
                target_emb = face_recognizer.enrolled_embeddings.get('Manager_Demo')
                face_crop = fgsm_attacker.attack(face_crop, target_emb)
                
            # Anomalie
            is_attacked_flag, anom_score = anomaly_detector.analyze(face_crop)
            is_attacked = is_attacked or is_attacked_flag
            
            # Défense
            if current_mode == 'hardened':
                face_crop = defender.apply_defense(face_crop, defense_type='gaussian')
                
            # Reconnaissance
            identity, conf = face_recognizer.predict(face_crop)
            if conf > current_conf:
                current_identity = identity
                current_conf = conf
                
            # Overlay Bbox
            x1, y1, x2, y2 = bbox
            ui_config = rights_manager.get_ui_config(identity)
            color_hex = ui_config['color'].lstrip('#')
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{identity} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Mise à jour de l'état global
        with state_lock:
            state.identity = current_identity
            state.confidence = current_conf
            state.anomaly_detected = is_attacked
            state.anomaly_score = anom_score
            state.access_level = rights_manager.get_access_level(current_identity)
            state.permissions = rights_manager.get_permissions(current_identity)
            
            # On stocke l'image encodée pour le flux MJPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                state.latest_frame = buffer.tobytes()

# Démarrage du thread
t = threading.Thread(target=video_processing_thread, daemon=True)
t.start()

# --- ROUTES FLASK ---

@app.route('/')
def index():
    return render_template('EntranceControl.html')

@app.route('/static_analysis')
def static_analysis():
    return render_template('StaticAnalysis.html')

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400

    try:
        # Lecture de l'image depuis la requête
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Format d'image invalide"}), 400
            
        # Détection du visage
        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({
                "success": True, 
                "results": {"identity": "Aucun visage", "access_level": "DENIED", "confidence": 0, "anomaly_detected": False, "anomaly_score": 0.0}
            })
            
        # On prend le premier visage trouvé
        bbox = bboxes[0]
        face_crop = face_detector.crop_face(img, bbox, size=160)
        
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage"}), 500
            
        # Détection d'anomalie (Spoofing)
        is_attacked, anom_score = anomaly_detector.analyze(face_crop)
        
        # Reconnaissance
        identity, conf = face_recognizer.predict(face_crop)
        
        # Récupération des droits
        access_level = rights_manager.get_access_level(identity)
        permissions = rights_manager.get_permissions(identity)
        
        # Overlay bbox sur l'image d'origine pour le retour
        x1, y1, x2, y2 = bbox
        ui_config = rights_manager.get_ui_config(identity)
        color_hex = ui_config['color'].lstrip('#')
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        
        # Encodage de l'image résultante en base64 pour affichage frontend
        import base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "results": {
                "identity": identity,
                "confidence": conf,
                "access_level": access_level,
                "permissions": permissions,
                "anomaly_detected": is_attacked,
                "anomaly_score": anom_score,
                "image_base64": img_base64
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def generate_mjpeg():
    while True:
        with state_lock:
            frame = state.latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # ~30fps max

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify({
            'identity': state.identity,
            'access_level': state.access_level,
            'permissions': state.permissions,
            'anomaly_detected': state.anomaly_detected,
            'anomaly_score': state.anomaly_score,
            'attack_active': state.attack_active,
            'model_mode': state.model_mode,
            'fps': state.fps,
            'confidence': state.confidence
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json
    if 'active' in data:
        with state_lock:
            state.attack_active = data['active']
            status = "activée" if state.attack_active else "désactivée"
        return jsonify({"success": True, "message": f"Attaque {status}."})
    return jsonify({"success": False, "error": "Paramètre 'active' manquant."}), 400

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json
    if 'mode' in data and data['mode'] in ['standard', 'hardened']:
        mode = data['mode']
        try:
            if face_recognizer:
                face_recognizer.switch_mode(mode)
            with state_lock:
                state.model_mode = mode
            return jsonify({"success": True, "message": f"Mode changé vers {mode}."})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Mode invalide."}), 400

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes"}), 400
        
    file = request.files['image']
    name = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400
        
    filename = secure_filename(f"{name}_{file.filename}")
    save_path = os.path.join(ENROLLED_DIR, filename)
    file.save(save_path)
    
    # Ajout dynamique au RightsManager
    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}"})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
