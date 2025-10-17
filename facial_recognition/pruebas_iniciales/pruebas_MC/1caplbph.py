import cv2
import imutils
from pymongo import MongoClient

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# --- Solicitar nombre de la persona ---
personName = input("Ingrese el nombre de la persona: ")

# Verificar si ya tiene suficientes rostros almacenados
persona_existente = personas.find_one({"nombre": personName})
if persona_existente and "rostros" in persona_existente and len(persona_existente["rostros"]) >= 600:
    print(f"{personName} ya tiene suficientes rostros registrados.")
    exit()

# --- Activar cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

# --- Clasificadores de rostro frontal y perfil ---
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

count = 0

# --- Función para guardar rostro en MongoDB ---
def guardar_rostro(nombre, rostro, count):
    _, buffer = cv2.imencode(".jpg", rostro)
    rostro_bytes = buffer.tobytes()
    personas.update_one(
        {"nombre": nombre},
        {"$push": {"rostros": {"imagen_id": count, "data": rostro_bytes}}},
        upsert=True
    )

# --- Bucle principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

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
                x = frame.shape[1] - x - w  # invertir coordenadas
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                guardar_rostro(personName, rostro, count)
                count += 1
                print(f" Foto {count} guardada (perfil izquierdo)")

        else:
            # --- Dibujar y guardar perfil derecho ---
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                guardar_rostro(personName, rostro, count)
                count += 1
                print(f" Foto {count} guardada (perfil derecho)")

    else:
        # --- Dibujar y guardar rostro frontal ---
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            guardar_rostro(personName, rostro, count)
            count += 1
            print(f" Foto {count} guardada (frontal)")

    cv2.imshow('Capturando Rostros (Frontal y Perfil)', frame)

    # --- Salida: ESC o #fotos finalizado ---
    k = cv2.waitKey(1)
    if k == 27 or count >= 500:
        break

cap.release()
cv2.destroyAllWindows()

print(f" Se guardaron {count} fotos de {personName}")
