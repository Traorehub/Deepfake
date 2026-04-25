import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import cv2
import os

class FaceRecognizer:
    def __init__(self, mode='standard'):
        """
        Initialise le reconnaisseur avec FaceNet (InceptionResnetV1 pré-entraîné).
        Utilise la méthode des embeddings.
        """
        self.device = torch.device("cpu")
        self.mode = mode
        
        print("Chargement de FaceNet (InceptionResnetV1)...")
        # Le modèle retourne un vecteur 512D
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)), # FaceNet attend du 160x160
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalisation spécifique à FaceNet
        ])
        
        self.enrolled_embeddings = {}
        self.threshold = 0.65 # Seuil de similarité cosinus

    def preprocess(self, face_crop: np.ndarray):
        """Prépare l'image pour FaceNet (RGB + 160x160 + Tenseur)."""
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def get_embedding(self, face_crop: np.ndarray):
        """Retourne l'embedding 512D pour un visage donné."""
        tensor = self.preprocess(face_crop)
        embedding = self.model(tensor)
        # Normalisation L2 du vecteur
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def enroll_face(self, name: str, face_crop: np.ndarray):
        """Enregistre l'empreinte d'un visage."""
        if face_crop is None:
            return False
        emb = self.get_embedding(face_crop)
        # Si on a déjà des images pour cette personne, on fait la moyenne
        if name in self.enrolled_embeddings:
            self.enrolled_embeddings[name] = F.normalize(self.enrolled_embeddings[name] + emb, p=2, dim=1)
        else:
            self.enrolled_embeddings[name] = emb
        print(f"[FaceRecognizer] Visage enrôlé pour {name}")
        return True

    @torch.no_grad()
    def predict(self, face_crop: np.ndarray):
        """
        Compare le visage avec la base de données via Cosine Similarity.
        """
        if face_crop is None or len(self.enrolled_embeddings) == 0:
            return "Inconnu", 0.0
            
        emb = self.get_embedding(face_crop)
        
        best_name = "Inconnu"
        best_score = -1.0
        
        for name, saved_emb in self.enrolled_embeddings.items():
            # Similarité cosinus entre -1 et 1 (1 = identique)
            score = F.cosine_similarity(emb, saved_emb).item()
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_score < self.threshold:
            return "Inconnu", max(0.0, best_score)
            
        return best_name, best_score

    def switch_mode(self, mode: str):
        """Pour la compatibilité avec l'API existante. Dans ce nouveau paradigme, le mode
        hardened est géré en amont par le Defender, pas par un changement de poids."""
        self.mode = mode
        print(f"[FaceRecognizer] Mode changé vers : {mode}")
