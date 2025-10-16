import cv2
import numpy as np
from pymongo import MongoClient

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

labels = []
facesData = []
label = 0
peopleList = []

print(" Leyendo datos desde MongoDB...")

# Ordenar por nombre para coherencia entre modelo y reconocimiento
for persona in personas.find().sort("nombre", 1):
    nombre = persona["nombre"]
    peopleList.append(nombre)
    print(f"Procesando persona: {nombre}")

    if "rostros" in persona:
        for rostro_doc in persona["rostros"]:
            data = rostro_doc["data"]
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Normalizar contraste para mayor robustez
                img = cv2.equalizeHist(img)
                facesData.append(img)
                labels.append(label)

    label += 1

print("Lista de personas:", peopleList)

# --- Entrenamiento del modelo LBPH ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print(" Entrenando modelo, espere unos segundos...")
face_recognizer.train(facesData, np.array(labels))

# --- Guardar modelo y etiquetas ---
face_recognizer.write("modeloLBPHFace.xml")
np.save("labels.npy", np.array(peopleList))

print("Modelo almacenado con éxito: modeloLBPHFace.xml")
