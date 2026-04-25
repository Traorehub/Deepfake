import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='yolov8n-face.pt', frame_skip=3):
        """
        Initialise le détecteur YOLOv8 optimisé pour les visages.
        frame_skip: nombre de frames à sauter pour économiser le CPU.
        """
        # Note: 'yolov8n-face.pt' est une version spécifique pour les visages
        # Si absent, on peut utiliser 'yolov8n.pt' qui détecte aussi les personnes (classe 0)
        try:
            self.model = YOLO(model_path)
        except:
            print(f"Modèle {model_path} non trouvé, repli sur yolov8n.pt")
            self.model = YOLO('yolov8n.pt')
            
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_results = []

    def detect(self, frame: np.ndarray):
        """
        Détecte les visages avec un système de frame skipping pour le CPU.
        """
        self.frame_count += 1
        
        # On ne traite qu'une frame sur N
        if self.frame_count % self.frame_skip == 0 or not self.last_results:
            results = self.model(frame, verbose=False, conf=0.5)
            self.last_results = []
            
            for r in results:
                for box in r.boxes:
                    # Si on utilise yolov8n.pt standard, on filtre la classe 0 (personne)
                    # Si on utilise yolov8n-face.pt, toutes les boxes sont des visages
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    self.last_results.append((x1, y1, x2, y2))
                    
        return self.last_results

    def crop_face(self, frame, bbox, size=128):
        """
        Découpe et redimensionne le visage.
        """
        x1, y1, x2, y2 = bbox
        # Assurer que les coordonnées sont dans l'image
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
            
        face_resized = cv2.resize(face, (size, size))
        return face_resized
