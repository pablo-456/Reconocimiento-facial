import cv2
import numpy as np
from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# Lista de nombres de personas
imagePaths = [p["nombre"] for p in personas.find()]
print("Personas registradas en Mongo:", imagePaths)

# Cargar modelo entrenado
face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.read("modeloFisherFace.xml")

# Cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

# Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        label, confidence = face_recognizer.predict(rostro)

        if confidence < 500 and 0 <= label < len(imagePaths):
            name = imagePaths[label]
            color = (0, 255, 0)
        else:
            name = "Desconocido"
            color = (0, 0, 255)

        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 2, 0.7, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Reconocimiento", frame)

    k = cv2.waitKey(1)
    if k == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
