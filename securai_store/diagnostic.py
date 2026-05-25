"""
Diagnostic complet Hugging Face Space.
Lance depuis le dossier securai_store/ :
    python diagnostic_hf.py
"""
import requests, cv2, base64, numpy as np, time, json, sys

HF_BASE = 'https://demimolchabite-securai-api.hf.space'

# ── Session réutilisable (évite re-négociation TLS) ──
session = requests.Session()
session.headers.update({'Content-Type': 'application/json'})

def encode(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

dummy_160 = np.zeros((160, 160, 3), dtype=np.uint8)
dummy_b64 = encode(dummy_160)

def separator(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print('═'*55)

# ════════════════════════════════════════════════════════
# 1. HEALTH CHECK — device GPU ou CPU ?
# ════════════════════════════════════════════════════════
separator("1. HEALTH CHECK — GPU ou CPU ?")
try:
    t0   = time.time()
    r    = session.get(f"{HF_BASE}/health", timeout=10)
    ms   = int((time.time()-t0)*1000)
    data = r.json()
    print(f"  Status  : {r.status_code}")
    print(f"  Latence : {ms}ms")
    print(f"  Device  : {data.get('device', '???')}  ← GPU=cuda / CPU=cpu")
    print(f"  Enrolled: {data.get('enrolled', [])}")
    print(f"  Réponse complète : {json.dumps(data, indent=4)}")
except Exception as e:
    print(f"  ❌ ERREUR : {e}")
    sys.exit(1)

# ════════════════════════════════════════════════════════
# 2. LATENCE /infer — 10 appels avec session réutilisée
# ════════════════════════════════════════════════════════
separator("2. LATENCE /infer (10 appels, session réutilisée)")
times_session = []
for i in range(10):
    t0 = time.time()
    r  = session.post(f"{HF_BASE}/infer", json={"image": dummy_b64}, timeout=10)
    ms = (time.time()-t0)*1000
    times_session.append(ms)
    print(f"  Appel {i+1:02d} : {ms:.0f}ms → {r.json()}")

avg = sum(times_session)/len(times_session)
print(f"\n  Moy={avg:.0f}ms | Min={min(times_session):.0f}ms | Max={max(times_session):.0f}ms")
print(f"  FPS théorique (1 appel/frame) : {1000/avg:.1f}")
print(f"  FPS théorique (mode attaque)  : {1000/(avg*2):.1f}  (2 appels/frame)")

# ════════════════════════════════════════════════════════
# 3. LATENCE /infer — 10 appels SANS session (nouvelle connexion à chaque fois)
# ════════════════════════════════════════════════════════
separator("3. LATENCE /infer (10 appels, SANS session — comparaison)")
times_nosession = []
for i in range(10):
    t0 = time.time()
    r  = requests.post(f"{HF_BASE}/infer", json={"image": dummy_b64}, timeout=10)
    ms = (time.time()-t0)*1000
    times_nosession.append(ms)
    print(f"  Appel {i+1:02d} : {ms:.0f}ms")

avg2 = sum(times_nosession)/len(times_nosession)
print(f"\n  Moy={avg2:.0f}ms | Min={min(times_nosession):.0f}ms | Max={max(times_nosession):.0f}ms")
print(f"  Gain session réutilisée : {avg2-avg:.0f}ms / appel")

# ════════════════════════════════════════════════════════
# 4. TEST /fgsm — latence calcul adversarial
# ════════════════════════════════════════════════════════
separator("4. LATENCE /fgsm (5 appels)")
for i in range(5):
    t0 = time.time()
    r  = session.post(f"{HF_BASE}/fgsm",
                      json={"image": dummy_b64, "target": "Manager_Demo"},
                      timeout=20)
    ms = (time.time()-t0)*1000
    data = r.json()
    print(f"  Appel {i+1} : {ms:.0f}ms | success={data.get('success')} | "
          f"reconnu={data.get('recognized_as', data.get('name', '?'))} "
          f"conf={data.get('confidence', '?')}")

# ════════════════════════════════════════════════════════
# 5. VERDICT
# ════════════════════════════════════════════════════════
separator("5. VERDICT & DIAGNOSTIC")
avg_infer = sum(times_session)/len(times_session)

if avg_infer < 100:
    verdict = "🟢 EXCELLENT — latence normale pour GPU distant"
elif avg_infer < 250:
    verdict = "🟡 CORRECT — acceptable pour une démo"
elif avg_infer < 500:
    verdict = "🟠 LENT — probablement CPU ou réseau Maroc→US"
else:
    verdict = "🔴 TRÈS LENT — CPU confirmé ou problème réseau"

print(f"\n  Latence moyenne /infer : {avg_infer:.0f}ms")
print(f"  Verdict : {verdict}")
print(f"\n  Causes possibles si lent :")
print(f"  1. HF Space tourne sur CPU (vérifier 'device' ci-dessus)")
print(f"  2. Réseau Maroc → serveurs HF USA : +100-200ms incompressible")
print(f"  3. HF Space en 'sleep' mode (cold start ~5s au premier appel)")
print(f"  4. GPU payant pas encore activé sur le Space")
print(f"\n  Pour activer GPU sur HF Space :")
print(f"  → huggingface.co/spaces/demimolchabite/securai-api")
print(f"  → Settings → Hardware → T4 small (0.60$/h)")
print()