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
UMBRAL_FUERTE = 55     # súper confiable
UMBRAL_DEBIL = 65      # por encima de esto = desconocido

FRAME_SKIP = 2         # detectar cada N frames
RESIZE_WIDTH = 640     # ancho de procesamiento (menor = más rápido)
FPS_DISPLAY_SMOOTH = 0.9

# ============================================================
# Inicializar cámara
# ============================================================
# --- OPCIONES DE CÁMARA ---
# 1️⃣ Cámara del PC (predeterminada)
cap = cv2.VideoCapture(0)

# 2️⃣ Cámara externa USB (por ejemplo, iPhone con Iriun o EpocCam por cable)
#cap = cv2.VideoCapture(2)  # o prueba con 2 si hay varias cámaras

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
frames_estables = 0
ultima_persona = None
frames_sin_reconocer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break
   
    # Reducir tamaño para procesar menos pixeles
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

        # Actualizar cache
        prev_faces = detected_faces
    else:
        detected_faces = prev_faces or []

    # Reconocimiento sobre las detecciones
    for (x, y, w, h) in detected_faces:
        x = max(0, x); y = max(0, y)
        w = max(1, w); h = max(1, h)
        if x + w > gray.shape[1] or y + h > gray.shape[0]:
            continue

        rostro = gray[y:y+h, x:x+w]
        if rostro.size == 0:
            continue

        rostro_resized = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_AREA)

        try:
            label, confidence = face_recognizer.predict(rostro_resized)
        except Exception as e:
            print("Predict error:", e)
            continue

        # ----------- DOBLE UMBRAL DE RECONOCIMIENTO -----------
        if confidence < UMBRAL_FUERTE and 0 <= label < len(imagePaths):
            name = imagePaths[label]
            estado = "."
            color_estado = (0, 255, 0)

        elif confidence < UMBRAL_DEBIL and 0 <= label < len(imagePaths):
            name = imagePaths[label]
            estado = "."
            color_estado = (0, 255, 0)

        else:
            name = "Desconocido"
            color_estado = (0, 0, 255)


        # -------------------------------------
        # --- MENSAJE DE DETECCIÓN DE USUARIO ---
        # -------------------------------------
        if name != "Desconocido":
            if name == ultima_persona:
                frames_estables += 1
            else:
                frames_estables = 1
                ultima_persona = name

            # Si la persona ha sido reconocida durante varios frames seguidos
            if frames_estables >= 20:
                print(f"✅ Acceso autorizado para: {name}")
                cv2.putText(frame, f"ACCESO AUTORIZADO: {name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Reconocimiento Facial (corregido)", frame)
                cv2.waitKey(2000)  # muestra 2 segundos el mensaje
                frames_estables = 0
                ultima_persona = None
                continue
        else:
            frames_estables = 0
            ultima_persona = None

        # --- Control cuando no se reconoce a nadie ---
        if stable_name == "Desconocido":
            frames_sin_reconocer += 1
        else:
            frames_sin_reconocer = 0

        if frames_sin_reconocer >= 100:
            print("⚠️ Persona no reconocida")
            cv2.putText(frame, "PERSONA NO RECONOCIDA", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
            cv2.imshow("Reconocimiento Facial (corregido)", frame)
            cv2.waitKey(2000)  # muestra 2 segundos el mensaje
            frames_sin_reconocer = 0
            continue

        # Actualizar memoria temporal
        name_memory[name] = name_memory.get(name, 0) + 1
        if name_memory.get(name, 0) > 5:
            stable_name = name

        color = (0, 255, 0) if stable_name != "Desconocido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name}  [{confidence:.1f}]  {estado}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color_estado, 2)

    # Decaimiento ligero de memoria
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
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    # Mostrar frame
    cv2.imshow("Reconocimiento Facial (corregido)", frame)

    # --- Centrar ventana al inicio ---
    if frame_count == 1:
        window_name = "Reconocimiento Facial (corregido)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) 
        import tkinter as tk
        root_tk = tk.Tk()
        screen_width = root_tk.winfo_screenwidth()
        screen_height = root_tk.winfo_screenheight()
        root_tk.destroy()
        win_w = frame.shape[1]
        win_h = frame.shape[0]
        x = max((screen_width - win_w) // 2, 0)
        y = max((screen_height - win_h) // 2, 0)
        cv2.moveWindow(window_name, x, y)

        
    # Salir con ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Reconocimiento Facial (corregido)", cv2.WND_PROP_VISIBLE) < 1:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
