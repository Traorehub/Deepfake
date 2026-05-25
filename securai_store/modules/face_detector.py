import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='yolov8n-face.pt', frame_skip=3):
        """
        Initialise le détecteur YOLOv8 dédié aux visages.
        * **model_path** – chemin du poids « yolov8n-face.pt ». Aucun fallback :
          si le fichier est absent ou trop petit, on lève une exception explicite.
        * **frame_skip** – nombre de frames à ignorer pour alléger le CPU.
        """
        # ------------------------------------------------------------------------
        # Vérification stricte et chargement du modèle.
        # Aucun fallback vers un modèle d'objets généraliste (ex: YOLOv5-n) n'est autorisé.
        # ------------------------------------------------------------------------
        import pathlib
        
        model_file = pathlib.Path(model_path)
        loaded = False
        
        # Liste des chemins potentiels à tester
        paths_to_try = [model_file]
        if model_path == 'yolov8n-face.pt':
            base_dir = pathlib.Path(__file__).parent.parent
            # On cherche dans 'models/' ou directement à la racine de securai_store
            paths_to_try.append(base_dir / 'models' / 'yolov8n-face.pt')
            paths_to_try.append(base_dir / 'yolov8n-face.pt')
            
        for path_opt in paths_to_try:
            if path_opt.is_file() and path_opt.stat().st_size >= 1_000_000:
                try:
                    self.model = YOLO(str(path_opt))
                    loaded = True
                    print(f"[FaceDetector] Modèle chargé avec succès depuis : {path_opt.absolute()}")
                    break
                except Exception as e:
                    print(f"[FaceDetector] Tentative de chargement depuis {path_opt} échouée : {e}")
                    
        if not loaded:
            raise FileNotFoundError(
                f"Impossible de charger un modèle YOLOv8-face valide depuis les chemins testés : {[str(p.absolute()) for p in paths_to_try]}. "
                "Le fichier est peut-être absent, corrompu (KeyError: 'model'), ou incomplet.\n"
                "Veuillez exécuter 'python securai_store/download.py' pour télécharger et installer le modèle correct."
            )
            
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
            results = self.model(frame, verbose=False, conf=0.5, imgsz=160)
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
