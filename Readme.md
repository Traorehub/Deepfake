# SecurAI - Simulateur de Vulnérabilité Biométrique 👁️🛡️

**SecurAI** est un Proof of Concept (POC) professionnel démontrant les capacités et les vulnérabilités d'un système de reconnaissance faciale moderne (FaceNet) face aux attaques adversariales (Spoofing mathématique).

Ce projet a été conçu pour illustrer comment un réseau de neurones peut être trompé (Attaque PGD) et comment se défendre via l'analyse de fréquence (FFT) et le filtrage spatial.

---

## 🚀 Fonctionnalités Principales

1. **Reconnaissance Faciale Haute Précision (FaceNet)**
   - Utilisation d'**InceptionResnetV1** pré-entraîné sur VGGFace2.
   - Extraction de signatures vectorielles (Embeddings 512D) pour une reconnaissance par similarité cosinus.
   - Zéro entraînement requis : le système s'auto-enrôle à partir du dossier `/data/enrolled/`.

2. **Attaque Adversariale (I-FGSM / PGD)**
   - Implémentation d'une attaque ciblée pour usurper l'identité de l'Administrateur.
   - Calcul des gradients (Projected Gradient Descent) pour générer un bruit mathématique invisible ou minimal, capable de franchir le seuil de similarité biométrique.
   - Simulation d'une attaque *Man-in-the-Middle* sur le flux vidéo.

3. **Mécanismes de Défense Anti-Spoofing**
   - **Détection d'Anomalie Spectrale (FFT)** : Analyse en temps réel des fréquences de l'image. Le bruit adverse crée une explosion dans les hautes fréquences, immédiatement détectée par la barre d'anomalie du système.
   - **Filtrage Spatial** : Utilisation de filtres Gaussiens/Médians appliqués avant le réseau de neurones pour détruire mathématiquement la perturbation adversariale sans altérer les traits du visage.

4. **Interface Control Room (Web GUI)**
   - Tableau de bord immersif "Cyber-Security" en plein écran.
   - Retour vidéo en temps réel avec Bounding Boxes et scores de confiance.
   - Gestion de l'affichage par code couleur (Vert = Accès Total, Jaune = Accès Limité, Rouge = Refusé).
   - Mode "Analyse Statique" pour auditer des images via glisser-déposer.

---

## 📂 Architecture du Projet

Le projet a été nettoyé et consolidé dans l'architecture **V2** suivante :

```text
Deepfake_Project/
└── securai_store/           # Cœur du POC
    ├── app.py               # Serveur backend (Flask, multithreading vidéo)
    ├── rights_manager.py    # Gestionnaire des rôles (Admin, Employé, Inconnu)
    ├── data/
    │   └── enrolled/        # Photos des personnes autorisées (format: Role_Nom-X.jpg)
    ├── modules/
    │   ├── anomaly_detector.py # Détection des hautes fréquences (Spoofing)
    │   ├── defender.py         # Filtres de défense (Gaussian Blur)
    │   ├── face_detector.py    # Détection des visages (YOLOv8)
    │   ├── face_recognizer.py  # Extraction d'embeddings (FaceNet InceptionResnetV1)
    │   └── fgsm_attacker.py    # Moteur d'attaque adversariale (I-FGSM/PGD)
    └── templates/
        ├── EntranceControl.html # Dashboard vidéo en temps réel
        └── StaticAnalysis.html  # Interface de test d'images fixes
```

---

## 🛠️ Installation & Lancement

1. **Environnement Virtuel**
   Assurez-vous d'utiliser l'environnement virtuel inclus ou installez les dépendances via `requirements.txt` situé dans `securai_store/`.
   ```bash
   pip install -r securai_store/requirements.txt
   ```
   *(Note : Pour des performances fluides lors de l'attaque PGD, PyTorch avec support CUDA est fortement recommandé).*

2. **Démarrage du Système**
   ```bash
   cd securai_store
   python app.py
   ```

3. **Accès au Tableau de Bord**
   Ouvrez un navigateur (Google Chrome recommandé) et rendez-vous sur : `http://localhost:5000`

---

## 👤 Gestion des Utilisateurs (Enrôlement)

Pour ajouter une personne reconnue par le système :
1. Prenez une photo claire de son visage.
2. Renommez le fichier selon la convention : `[Role]_[Nom]-[Numero].jpg`.
   * Exemple pour un Administrateur : `Manager_Elon-1.jpg`
   * Exemple pour un Employé : `Employee_Bob-1.jpg`
3. Placez l'image dans le dossier `securai_store/data/enrolled/`.
4. Relancez l'application. Le système fusionnera automatiquement les embeddings s'il y a plusieurs photos pour la même personne.

---
*Ce projet est à but purement éducatif dans le cadre d'une démonstration des vulnérabilités de l'IA (Adversarial Machine Learning).*
