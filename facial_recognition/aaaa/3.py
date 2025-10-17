import cv2
import numpy as np
from pymongo import MongoClient

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# --- Cargar lista de personas en orden alfabético ---
imagePaths = [p["nombre"] for p in personas.find().sort("nombre", 1)]
print("Personas registradas:", imagePaths)

# --- Cargar modelo LBPH entrenado ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("modeloLBPHFace.xml")

# --- Parámetros de confiabilidad ---
UMBRAL_CONFIANZA = 60   # cuanto menor, más estricto

# ==============================================================
#  OPCIONES DE CÁMARA
# ==============================================================

# 1️⃣ Cámara del PC (predeterminada)
# cap = cv2.VideoCapture(0)

# 2️⃣ Cámara externa USB (por ejemplo, iPhone con Iriun o EpocCam por cable)
cap = cv2.VideoCapture(1)  # o prueba con 2 si hay varias cámaras

# 3️⃣ Cámara del iPhone por Wi-Fi (modo IP Webcam o Iriun Wi-Fi)
#     Abre la app en tu iPhone y revisa la IP que muestra, por ejemplo:
#     "http://192.168.1.5:4747/video"
#     Cambia la IP de abajo por la tuya
#cap = cv2.VideoCapture("http://172.20.10.1:8080/video")


# ==============================================================
# --- Inicializar cámara ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

print(" Reconocimiento facial iniciado. Presione ESC para salir.")

# --- Variables de estabilidad ---
name_memory = {}
stable_name = "Desconocido"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_show = frame.copy()
    recognized = False

    # --- Buscar rostro frontal ---
    faces = frontal_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    if len(faces) == 0:
        # Perfil derecho
        faces = profile_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

        if len(faces) == 0:
            # Perfil izquierdo
            gray_flipped = cv2.flip(gray, 1)
            faces = profile_face.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))
            for (x, y, w, h) in faces:
                x = frame.shape[1] - x - w
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                label, confidence = face_recognizer.predict(rostro)

                if confidence < UMBRAL_CONFIANZA and 0 <= label < len(imagePaths):
                    name = imagePaths[label]
                else:
                    name = "Desconocido"

                # Acumular detecciones estables
                name_memory[name] = name_memory.get(name, 0) + 1
                if name_memory[name] > 5:
                    stable_name = name

                color = (0, 255, 0) if stable_name != "Desconocido" else (0, 0, 255)
                cv2.putText(frame_show, f"{stable_name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
                cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
                recognized = True

        else:
            for (x, y, w, h) in faces:
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                label, confidence = face_recognizer.predict(rostro)

                if confidence < UMBRAL_CONFIANZA and 0 <= label < len(imagePaths):
                    name = imagePaths[label]
                else:
                    name = "Desconocido"

                name_memory[name] = name_memory.get(name, 0) + 1
                if name_memory[name] > 5:
                    stable_name = name

                color = (0, 255, 0) if stable_name != "Desconocido" else (0, 0, 255)
                cv2.putText(frame_show, f"{stable_name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
                cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
                recognized = True
    else:
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            label, confidence = face_recognizer.predict(rostro)

            if confidence < UMBRAL_CONFIANZA and 0 <= label < len(imagePaths):
                name = imagePaths[label]
            else:
                name = "Desconocido"

            name_memory[name] = name_memory.get(name, 0) + 1
            if name_memory[name] > 5:
                stable_name = name

            color = (0, 255, 0) if stable_name != "Desconocido" else (0, 0, 255)
            cv2.putText(frame_show, f"{stable_name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
            recognized = True

    cv2.imshow("Reconocimiento Facial", frame_show)

    k = cv2.waitKey(1)
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
