import cv2
import numpy as np
from pymongo import MongoClient

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# --- Obtener lista de personas registradas ---
imagePaths = [p["nombre"] for p in personas.find()]
print(" Personas registradas en MongoDB:", imagePaths)

# --- Cargar modelo entrenado (LBPH) ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("modeloLBPHFace.xml")

# --- Inicializar cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

# --- Clasificadores de rostros ---
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

print("🎥 Reconocimiento facial iniciado. Presione ESC para salir.")

# --- Bucle principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_show = frame.copy()
    recognized = False

    # --- Paso 1: Buscar rostro frontal ---
    faces = frontal_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    if len(faces) == 0:
        # --- Paso 2: Buscar perfil derecho ---
        faces = profile_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

        if len(faces) == 0:
            # --- Paso 3: Buscar perfil izquierdo ---
            gray_flipped = cv2.flip(gray, 1)
            faces = profile_face.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))
            for (x, y, w, h) in faces:
                x = frame.shape[1] - x - w  # invertir coordenadas para el perfil izquierdo
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                label, confidence = face_recognizer.predict(rostro)

                if confidence < 65 and 0 <= label < len(imagePaths):
                    name = imagePaths[label]
                    color = (0, 255, 0)
                else:
                    name = "Desconocido"
                    color = (0, 0, 255)

                cv2.putText(frame_show, f"{name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
                cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
                recognized = True
        else:
            # --- Rostros de perfil derecho ---
            for (x, y, w, h) in faces:
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                label, confidence = face_recognizer.predict(rostro)

                if confidence < 65 and 0 <= label < len(imagePaths):
                    name = imagePaths[label]
                    color = (0, 255, 0)
                else:
                    name = "Desconocido"
                    color = (0, 0, 255)

                cv2.putText(frame_show, f"{name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
                cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
                recognized = True
    else:
        # --- Rostros frontales ---
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            label, confidence = face_recognizer.predict(rostro)

            if confidence < 65 and 0 <= label < len(imagePaths):
                name = imagePaths[label]
                color = (0, 255, 0)
            else:
                name = "Desconocido"
                color = (0, 0, 255)

            cv2.putText(frame_show, f"{name} ({confidence:.2f})", (x, y - 10), 2, 0.7, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame_show, (x, y), (x + w, y + h), color, 2)
            recognized = True

    # --- Mostrar resultado ---
    cv2.imshow("Reconocimiento Facial (Frontal y Perfil)", frame_show)

    # --- Salida: tecla ESC ---
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Reconocimiento finalizado.")
