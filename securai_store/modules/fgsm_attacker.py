import torch
import torch.nn.functional as F
import numpy as np
import cv2

class FGSMAttacker:
    def __init__(self, model, epsilon=0.03):
        """
        Génère des attaques FGSM sur le modèle FaceNet (InceptionResnetV1).
        L'attaque vise à manipuler l'embedding (vecteur 512D).
        """
        self.model = model
        self.epsilon = epsilon
        # FaceNet normalization
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def preprocess(self, face_crop: np.ndarray):
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # Resize to 160x160 as expected by FaceNet
        face_resized = cv2.resize(face_rgb, (160, 160))
        img_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        # Normalize
        img_tensor = (img_tensor - self.mean.squeeze(0)) / self.std.squeeze(0)
        return img_tensor.unsqueeze(0)

    def attack(self, face_crop: np.ndarray, target_embedding=None, steps=10):
        """
        Applique l'attaque I-FGSM (Iterative FGSM / PGD) sur un crop BGR.
        FaceNet est très robuste, une attaque itérative est nécessaire.
        """
        img_tensor_orig = self.preprocess(face_crop)
        img_tensor = img_tensor_orig.clone().detach().requires_grad_(True)
        
        # On augmente l'epsilon car FaceNet est coriace et la distance est grande
        self.epsilon = 0.15 
        alpha = 0.02 # Taille du pas par itération
        
        for step in range(steps):
            embedding = self.model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            
            if target_embedding is not None:
                # Attaque ciblée
                loss = 1.0 - F.cosine_similarity(embedding, target_embedding)
                sign = -1.0
            else:
                # Attaque non ciblée
                loss = embedding.sum()
                sign = 1.0
                
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Avancer d'un petit pas dans la direction du gradient
                adv_img = img_tensor + sign * alpha * img_tensor.grad.sign()
                
                # S'assurer qu'on ne dépasse pas la limite de perturbation maximale (epsilon)
                eta = torch.clamp(adv_img - img_tensor_orig, min=-self.epsilon, max=self.epsilon)
                img_tensor = img_tensor_orig + eta
                
            img_tensor.requires_grad = True
            
        # Denormalize
        perturbed_image = img_tensor * self.std + self.mean
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Retour BGR
        perturbed_np = perturbed_image.squeeze(0).detach().numpy().transpose(1, 2, 0)
        perturbed_bgr = cv2.cvtColor((perturbed_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        h, w = face_crop.shape[:2]
        return cv2.resize(perturbed_bgr, (w, h))

    def get_perturbation_visual(self, face_crop):
        """Retourne uniquement le bruit ajouté (amplifié x10) pour l'UI."""
        img_tensor = self.preprocess(face_crop).requires_grad_(True)
        
        embedding = self.model(img_tensor)
        loss = embedding.sum()
        self.model.zero_grad()
        loss.backward()
        
        noise = self.epsilon * img_tensor.grad.data.sign()
        noise = torch.clamp(noise * 10, -1, 1) 
        
        noise_np = noise.squeeze(0).detach().numpy().transpose(1, 2, 0)
        return ((noise_np + 1) / 2 * 255).astype(np.uint8)
