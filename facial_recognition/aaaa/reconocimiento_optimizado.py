import cv2
import numpy as np
from pymongo import MongoClient
import time
import sys

# ============================================================
# Conexión a MongoDB y lista de personas
# ============================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

imagePaths = [p["nombre"] for p in personas.find().sort("nombre", 1)]
print("Personas registradas:", imagePaths)

# ============================================================
# Cargar modelo LBPH (verifica que exista)
# ============================================================
MODEL_PATH = "modeloLBPHFace.xml"
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)
except Exception as e:
    print("Error cargando el modelo LBPH. Asegúrate de que", MODEL_PATH, "exista y que OpenCV tenga contrib.")
    print("Detalle:", e)
    sys.exit(1)

# ============================================================
# Parámetros (ajusta si quieres)
# ============================================================
UMBRAL_CONFIANZA = 60
FRAME_SKIP = 2         # detectar cada N frames
RESIZE_WIDTH = 640     # ancho de procesamiento (menor = más rápido)
FPS_DISPLAY_SMOOTH = 0.9

# ============================================================
# Inicializar cámara
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    sys.exit(1)

# intenta fijar resolución de captura (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================================
# Cargar Haar cascades
# ============================================================
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# ============================================================
# Variables de estado
# ============================================================
frame_count = 0
prev_faces = []            # cache de detecciones (lista de tuples)
name_memory = {}           # conteo temporal de detecciones por nombre
stable_name = "Desconocido"
last_time = time.time()
fps_display = 0.0

print("Reconocimiento facial iniciado. Presione ESC para salir.")

# ------------------------------------------------------------
# Helpers: normaliza salida de detectMultiScale a lista de tuplas
# ------------------------------------------------------------
def to_list_of_tuples(detect):
    """Convierte la salida de detectMultiScale a lista de (x,y,w,h)."""
    if detect is None:
        return []
    # Si es ndarray con shape (N,4)
    try:
        return [tuple(map(int, d)) for d in detect]
    except Exception:
        # Si ya es lista/iterable de tuplas
        return list(detect)

# ------------------------------------------------------------
# Bucle principal
# ------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    # reducir tamaño para procesar menos pixeles
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1
    detected_faces = []

    # Ejecutar detección completa solo cada FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        # 1) Frontal
        faces_front = to_list_of_tuples(frontal_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90)))
        if len(faces_front) > 0:
            detected_faces = faces_front
        else:
            # 2) Perfil izquierdo
            faces_left = to_list_of_tuples(profile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90)))

            # 3) Perfil derecho (imagen volteada)
            gray_flipped = cv2.flip(gray, 1)
            found_right = to_list_of_tuples(profile_face.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90)))
            faces_right = []
            for (x, y, w, h) in found_right:
                x_real = gray.shape[1] - x - w
                faces_right.append((x_real, y, w, h))

            detected_faces = faces_left + faces_right

        # actualizar cache
        prev_faces = detected_faces
    else:
        # reutilizar última detección para no volver a costear detectMultiScale
        detected_faces = prev_faces or []

    # Reconocimiento sobre las detecciones
    for (x, y, w, h) in detected_faces:
        # asegurarse de que las coordenadas estén dentro del frame
        x = max(0, x); y = max(0, y)
        w = max(1, w); h = max(1, h)
        if x + w > gray.shape[1] or y + h > gray.shape[0]:
            continue

        rostro = gray[y:y+h, x:x+w]
        # proteger contra recortes vacíos
        if rostro.size == 0:
            continue

        rostro_resized = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)

        # predict (esto es lo más costoso)
        try:
            label, confidence = face_recognizer.predict(rostro_resized)
        except Exception as e:
            # en caso de fallo en predict, continuar
            print("Predict error:", e)
            continue

        if confidence < UMBRAL_CONFIANZA and 0 <= label < len(imagePaths):
            name = imagePaths[label]
        else:
            name = "Desconocido"

        # Actualizar memoria temporal (decay implícito más abajo)
        name_memory[name] = name_memory.get(name, 0) + 1

        # Si un nombre alcanza umbral de estabilidad, fijarlo
        if name_memory.get(name, 0) > 5:
            stable_name = name

        color = (0, 255, 0) if stable_name != "Desconocido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{stable_name} ({confidence:.1f})", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Decaimiento ligero de name_memory para evitar saturación
    # (reduce todos los contadores en 1 cada 30 frames)
    if frame_count % 30 == 0:
        keys = list(name_memory.keys())
        for k in keys:
            name_memory[k] = max(0, name_memory[k] - 1)
            if name_memory[k] == 0:
                del name_memory[k]

    # FPS suavizado
    now = time.time()
    elapsed = now - last_time if (now - last_time) > 1e-6 else 1e-6
    instant_fps = 1.0 / elapsed
    fps_display = FPS_DISPLAY_SMOOTH * fps_display + (1 - FPS_DISPLAY_SMOOTH) * instant_fps
    last_time = now
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Reconocimiento Facial (corregido)", frame)

    # manejo de tecla (no bloquear con sleep)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# liberar recursos
cap.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
