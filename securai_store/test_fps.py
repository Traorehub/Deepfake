
import requests, cv2, base64, numpy as np, time

dummy = np.zeros((160,160,3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', dummy)
b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

# Test 10 appels /infer
times = []
for i in range(10):
    t0 = time.time()
    requests.post('http://vision.api.near-u-api.org/infer', json={'image': b64}, timeout=5)
    times.append((time.time()-t0)*1000)

print(f'Min  : {min(times):.0f}ms')
print(f'Max  : {max(times):.0f}ms')
print(f'Moy  : {sum(times)/len(times):.0f}ms')
print(f'FPS théorique max : {1000/(sum(times)/len(times)):.1f}')
