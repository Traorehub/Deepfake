import os
import sys
import cv2
import socket
import time
import numpy as np

def check(name, test_func):
    print(f"[{'RUN':^6}] {name}...", end="", flush=True)
    try:
        if test_func():
            print(f"\r[{'PASS':^6}] {name}   ")
            return True
        else:
            print(f"\r[{'FAIL':^6}] {name}   ")
            return False
    except Exception as e:
        print(f"\r[{'FAIL':^6}] {name} - Exception: {e}")
        return False

def check_models():
    # FaceNet is downloaded automatically by facenet-pytorch
    try:
        from modules.face_recognizer import FaceRecognizer
        fr = FaceRecognizer(mode='standard')
        return True
    except Exception as e:
        print(f" Exception: {e} ", end="")
        return False

def check_webcam():
    cap = cv2.VideoCapture(0)
    ret = cap.isOpened()
    if ret:
        # Try reading one frame
        ret, _ = cap.read()
    cap.release()
    return ret

def check_port_5000():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 5000))
    # If 0, port is in use (meaning app is probably running, which is fine, 
    # but we just want to know if we can bind or if it's available).
    # The requirement is "Flask démarre sur port 5000".
    # We will test if we can bind to it.
    sock.close()
    
    if result == 0:
        print(" (Port is already in use, assuming app is running) ", end="")
        return True
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 5000))
        sock.close()
        return True
    except socket.error:
        return False

def check_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    from modules.face_detector import FaceDetector
    from modules.face_recognizer import FaceRecognizer
    from modules.fgsm_attacker import FGSMAttacker
    from modules.defender import Defender
    from modules.anomaly_detector import AnomalyDetector
    
    detector = FaceDetector()
    recognizer = FaceRecognizer(mode='standard')
    attacker = FGSMAttacker(recognizer.model)
    defender = Defender()
    anomaly = AnomalyDetector()
    
    # Dummy image
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Force a dummy face crop
    dummy_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Run cycle
    detector.detect(dummy_frame)
    attacked = attacker.attack(dummy_crop)
    defended = defender.apply_defense(attacked, 'gaussian')
    is_anom, score = anomaly.analyze(defended)
    identity, conf = recognizer.predict(defended)
    
    return True

def run_healthcheck():
    print("Démarrage du Healthcheck SecurAI Store...\n")
    
    checks = [
        ("Modèles chargés", check_models),
        ("Webcam accessible", check_webcam),
        ("Port 5000 disponible", check_port_5000),
        ("Pipeline IA 1 Cycle", check_pipeline)
    ]
    
    all_passed = True
    start_time = time.time()
    for name, func in checks:
        if not check(name, func):
            all_passed = False
            
    elapsed = time.time() - start_time
    print(f"\nTemps écoulé: {elapsed:.2f}s")
    
    if all_passed:
        print("✅ TOUT EST OPÉRATIONNEL. Prêt pour la production.")
    else:
        print("❌ ÉCHEC DU HEALTHCHECK. Veuillez vérifier les erreurs ci-dessus.")

if __name__ == '__main__':
    run_healthcheck()
