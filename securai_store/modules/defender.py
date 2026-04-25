import cv2
import numpy as np

class Defender:
    """
    Fournit des filtres de prétraitement pour neutraliser le bruit adverse.
    """
    def __init__(self):
        pass

    def preprocess_gaussian(self, img: np.ndarray, kernel_size=3):
        """
        Applique un flou gaussien. Le bruit FGSM est très haute fréquence, 
        un léger flou peut le 'casser'.
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def preprocess_median(self, img: np.ndarray):
        """
        Le filtre médian est excellent pour supprimer le bruit de type 'sel et poivre'
        ou les perturbations pixel-par-pixel.
        """
        return cv2.medianBlur(img, 3)

    def apply_defense(self, img, defense_type='gaussian'):
        """Applique la défense sélectionnée."""
        if defense_type == 'gaussian':
            return self.preprocess_gaussian(img)
        elif defense_type == 'median':
            return self.preprocess_median(img)
        else:
            return img
