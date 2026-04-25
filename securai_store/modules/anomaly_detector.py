import numpy as np
import cv2

class AnomalyDetector:
    def __init__(self, threshold=0.35):
        """
        Détecte les anomalies dans les hautes fréquences de l'image.
        Les attaques FGSM ajoutent souvent un bruit 'artificiel' invisible à l'œil 
        mais très présent mathématiquement.
        """
        self.threshold = threshold

    def analyze(self, img: np.ndarray):
        """
        Analyse l'image via une Transformée de Fourier Rapide (FFT).
        """
        if img is None: return False, 0.0
        
        # 1. Conversion en gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. FFT pour passer du domaine spatial au domaine fréquentiel
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 3. Calcul de l'énergie des hautes fréquences (les bords du spectre)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # On crée un masque pour ignorer le centre (basses fréquences = formes globales)
        # et ne garder que les bords (hautes fréquences = détails/bruit)
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (center_x, center_y), 30, 0, -1) # Rayon de 30 au centre
        
        high_freq_energy = np.mean(magnitude_spectrum[mask == 1])
        
        # Score normalisé simple (à ajuster selon tes tests)
        anomaly_score = float(np.clip(high_freq_energy / 100, 0, 1))
        is_attacked = bool(anomaly_score > self.threshold)
        
        return is_attacked, anomaly_score

    def get_frequency_visualization(self, img):
        """Retourne une image du spectre FFT pour le dashboard."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(mag, cv2.COLORMAP_JET)
