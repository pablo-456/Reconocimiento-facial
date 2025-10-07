import cv2
import imutils
from pymongo import MongoClient

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# Pedir el nombre de la persona
personName = input("Ingrese el nombre de la persona: ")

# Verificar si ya tiene suficientes rostros almacenados
persona_existente = personas.find_one({"nombre": personName})
if persona_existente and "rostros" in persona_existente and len(persona_existente["rostros"]) >= 200:
    print(f"⚠️ {personName} ya tiene suficientes rostros registrados.")
    exit()

# Activar cámara (0 = cámara predeterminada)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: no se pudo acceder a la cámara.")
    exit()
    
# Video de entrada
#cap = cv2.VideoCapture('video_pablo.mp4')

# Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

# Función para guardar rostro en MongoDB
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
        print("Error al leer la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()  # ✅ ahora los rostros se guardan en escala de grises

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Guardar en MongoDB
        guardar_rostro(personName, rostro, count)
        count += 1
        print(f"📸 Foto {count} guardada")

    cv2.imshow('Capturando Rostros', frame)

    # Terminar si se presiona ESC o se alcanzan 200 fotos
    k = cv2.waitKey(1)
    if k == 27 or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Se guardaron {count} fotos de {personName}")
