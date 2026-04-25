import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_report():
    file_path = os.path.join(BASE_DIR, 'benchmark.json')
    if not os.path.exists(file_path):
        print("Erreur: benchmark.json introuvable. Exécutez 'python simulate.py --mode benchmark' d'abord.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    print("\n" + "="*60)
    print(" " * 15 + "RAPPORT DE RÉSILIENCE SecurAI")
    print("="*60)
    print(f"{'Métrique':<35} | {'Standard':<10} | {'Hardened':<10}")
    print("-" * 60)
    print(f"{'Précision de Reconnaissance (Clean)':<35} | {data['accuracy_standard']:<9.1f}% | {data['accuracy_hardened']:<9.1f}%")
    print(f"{'Taux de Succès Attaque (FGSM)':<35} | {data['attack_success_rate_standard']:<9.1f}% | {data['attack_success_rate_hardened']:<9.1f}%")
    print("-" * 60)
    print(f"{'Précision Détection Anomalie':<35} | {'N/A':<10} | {data['anomaly_detection_precision']:<9.1f}%")
    print(f"{'Temps Inférence Moyen (CPU)':<35} | {'N/A':<10} | {data['avg_inference_time_ms']:<9.1f} ms")
    print("="*60)
    
    print("\nConclusion:")
    if data['attack_success_rate_hardened'] < data['attack_success_rate_standard']:
        drop = data['attack_success_rate_standard'] - data['attack_success_rate_hardened']
        print(f"[SUCCÈS] Le modèle Hardened réduit la vulnérabilité aux attaques de {drop:.1f}%.")
    else:
        print("[ÉCHEC] Le modèle Hardened n'est pas plus résilient.")

if __name__ == '__main__':
    generate_report()
