import cv2
import numpy as np
from pymongo import MongoClient

# Conexión a MongoDB para obtener los nombres
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# Lista de personas (equivalente a las carpetas antes)
imagePaths = [p["nombre"] for p in personas.find()]
print("Personas registradas en Mongo:", imagePaths)

# Cargar el modelo entrenado
face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read("modeloFisherFace.xml")
# face_recognizer.read("modeloLBPHFace.xml")

# Video de entrada o cámara
# cap = cv2.VideoCapture("video_pablo.mp4")
cap = cv2.VideoCapture(0)  # si quieres cámara en vivo

# Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Predicción
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, "{}".format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # Con FisherFace
        if result[1] < 500:
            cv2.putText(frame, "{}".format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Desconocido", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        ''' #Con LBPH
        if result[1] < 70:
            cv2.putText(frame, "{}".format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Desconocido", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        '''

    cv2.imshow("Reconocimiento", frame)

    k = cv2.waitKey(1)
    if k == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
