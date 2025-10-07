import cv2
import numpy as np
from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

labels = []
facesData = []
label = 0
peopleList = []  # Para saber el nombre asignado a cada etiqueta

print("Leyendo datos desde MongoDB...")

# Recorremos cada persona en la colección
for persona in personas.find():
    nombre = persona["nombre"]
    peopleList.append(nombre)
    print(f"Procesando persona: {nombre}")

    if "rostros" in persona:
        for rostro_doc in persona["rostros"]:
            data = rostro_doc["data"]

            # Convertir los bytes a imagen OpenCV
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                facesData.append(img)
                labels.append(label)

    label += 1

print("Lista de personas:", peopleList)

# Entrenamiento del reconocedor
face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # alternativa

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo
face_recognizer.write("modeloFisherFace.xml")
# face_recognizer.write("modeloLBPHFace.xml")

print("Modelo almacenado con éxito")
