import cv2
import mediapipe as mp
import numpy as np
from pymongo import MongoClient
import imutils
import time

# ==============================
# 🔹 Configuración de MongoDB
# ==============================
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# Pedir nombre
personName = input("Ingrese el nombre de la persona: ")

# ==============================
# 🔹 Inicialización de MediaPipe
# ==============================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.6,
                                  min_tracking_confidence=0.6)

# ==============================
# 🔹 Configurar cámara
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: no se pudo acceder a la cámara.")
    exit()

# ==============================
# 🔹 Orientaciones esperadas
# ==============================
orientaciones = [
    ("Frontal", (0, 0)),
    ("Derecha", (25, 0)),
    ("Izquierda", (-25, 0)),
    ("Arriba", (0, -20)),
    ("Abajo", (0, 20)),
    ("Diagonal arriba derecha", (25, -20)),
    ("Diagonal arriba izquierda", (-25, -20)),
    ("Diagonal abajo derecha", (25, 20)),
    ("Diagonal abajo izquierda", (-25, 20))
]

actual_orientacion = 0
count = 0
limite_fotos_por_angulo = 15

# ==============================
# 🔹 Función para guardar rostro
# ==============================
def guardar_rostro(nombre, rostro, orientacion, count):
    _, buffer = cv2.imencode(".jpg", rostro)
    rostro_bytes = buffer.tobytes()
    personas.update_one(
        {"nombre": nombre},
        {"$push": {"rostros": {"imagen_id": count, "orientacion": orientacion, "data": rostro_bytes}}},
        upsert=True
    )

# ==============================
# 🔹 Función para estimar orientación
# ==============================
def calcular_angulo_pose(landmarks, ancho, alto):
    # Puntos clave: nariz, ojos, orejas
    nariz = landmarks[1]
    ojo_izq = landmarks[33]
    ojo_der = landmarks[263]
    menton = landmarks[199]

    # Calcular desplazamientos
    yaw = (ojo_der.x - ojo_izq.x) * 100  # izquierda-derecha
    pitch = (menton.y - nariz.y) * 100   # arriba-abajo
    return yaw, pitch

# ==============================
# 🔹 Bucle principal
# ==============================
print("\n📸 Siga las instrucciones para capturar diferentes ángulos del rostro.")
time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            alto, ancho, _ = frame.shape
            yaw, pitch = calcular_angulo_pose(face_landmarks.landmark, ancho, alto)

            # Orientación actual
            orient_name, (yaw_obj, pitch_obj) = orientaciones[actual_orientacion]

            # Mostrar instrucciones
            cv2.putText(frame, f"Mueve tu rostro: {orient_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Dibuja malla facial
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # Capturar si está cerca del ángulo objetivo
            if abs(yaw - yaw_obj) < 8 and abs(pitch - pitch_obj) < 8:
                x_coords = [int(lm.x * ancho) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * alto) for lm in face_landmarks.landmark]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                rostro = frame[y1:y2, x1:x2]
                if rostro.size > 0:
                    rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                    rostro_gray = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)
                    guardar_rostro(personName, rostro_gray, orient_name, count)
                    count += 1
                    cv2.putText(frame, f"✅ {orient_name} capturado ({count})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    time.sleep(0.5)

                    # Cambiar al siguiente ángulo
                    if count % limite_fotos_por_angulo == 0:
                        actual_orientacion += 1
                        if actual_orientacion >= len(orientaciones):
                            cap.release()
                            cv2.destroyAllWindows()
                            print(f"\n✅ Capturas completas ({count} fotos de {personName})")
                            exit()
                        time.sleep(1)

    cv2.imshow("Captura Facial con MediaPipe", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
