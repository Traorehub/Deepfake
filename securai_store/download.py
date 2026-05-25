# download_yolo_face.py  (remplacez le contenu actuel)
import pathlib
import urllib.request
import sys
from ultralytics import YOLO

MODEL_PATH = pathlib.Path(r"C:\Users\MOH\Desktop\Vision_Project\securai_store\models\yolov8n-face.pt")

# S'assurer que le dossier parent existe
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Vérifier si le modèle existant est valide
download_needed = False
if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size < 1_000_000:
    download_needed = True
else:
    try:
        # Tenter de charger le modèle pour vérifier sa validité (évite les fichiers corrompus/state_dicts incorrects)
        _ = YOLO(str(MODEL_PATH))
        print("Modèle YOLOv8-face déjà présent, valide et de taille correcte :", MODEL_PATH)
    except Exception as e:
        print(f"Modèle présent mais invalide ou corrompu ({e}). Nouveau téléchargement requis.")
        download_needed = True

if download_needed:
    print("Téléchargement du modèle YOLOv8-face depuis GitHub Releases...")
    url = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
    try:
        # Configuration d'un user-agent pour éviter d'être bloqué par GitHub
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        # Validation après téléchargement
        try:
            _ = YOLO(str(MODEL_PATH))
            print("Modèle YOLOv8-face téléchargé, validé et sauvegardé avec succès :", MODEL_PATH)
        except Exception as e:
            print(f"Le fichier téléchargé est invalide ou corrompu ({e}).")
            raise e
    except Exception as e:
        print(f"Erreur de téléchargement : {e}")
        print("Veuillez télécharger manuellement le modèle à l'adresse suivante :")
        print(url)
        print(f"Et placez-le sous le nom 'yolov8n-face.pt' dans le dossier : {MODEL_PATH.parent}")
        sys.exit(1)
