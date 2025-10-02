import cv2
import imutils
from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# Nombre de la persona
personName = 'pablo'

# Video de entrada
cap = cv2.VideoCapture('video_pablo.mp4')

# Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

def guardar_rostro(nombre, rostro, count):
    _, buffer = cv2.imencode(".jpg", rostro)
    rostro_bytes = buffer.tobytes()
    personas.update_one(
        {"nombre": nombre},
        {"$push": {"rostros": {"imagen_id": count, "data": rostro_bytes}}},
        upsert=True
    )

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)

        # Guardar en MongoDB
        guardar_rostro(personName, rostro, count)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 200:  # ESC o 200 capturas
        break

cap.release()
cv2.destroyAllWindows()
