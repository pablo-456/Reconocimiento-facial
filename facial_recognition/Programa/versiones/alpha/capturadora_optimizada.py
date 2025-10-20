import cv2
import imutils
import mediapipe as mp
from pymongo import MongoClient
import numpy as np
import sys
import os

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

# --- OPCIONES DE CÁMARA ---
# 1️⃣ Cámara del PC (predeterminada)
cap = cv2.VideoCapture(0)

# 2️⃣ Cámara externa USB (por ejemplo, iPhone con Iriun o EpocCam por cable)
#cap = cv2.VideoCapture(2)  # o prueba con 2 si hay varias cámaras

# --- Verificación de la cámara ---
if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Obtener datos desde argumentos (enviados por Tkinter) ---
if len(sys.argv) >= 4:
    cc = sys.argv[1].strip()
    personName = sys.argv[2].strip().lower()
    programa = " ".join(sys.argv[3:]).strip()
else:
    print("⚠️ No se recibieron suficientes argumentos. Finalizando...")
    sys.exit(1)

# --- Buscar si la persona ya existe ---
persona_existente = personas.find_one({"cc": cc})

if persona_existente:
    print(f"Persona existente encontrada: {personName}")
    person_id = persona_existente["_id"]
else:
    print(f"Registrando nueva persona: {personName}")
    person_id = personas.insert_one({
        "cc": cc,
        "nombre": personName,
        "programa": programa,
        "rostros": []
    }).inserted_id

# --- Comprobación de límite de rostros ---
persona_existente = personas.find_one({"_id": person_id})
if "rostros" in persona_existente and len(persona_existente["rostros"]) >= 600:
    print(f"{personName} ya tiene suficientes rostros registrados.")
    exit()

# --- Clasificadores ---
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

count = 0
Maxfotos = 500

# --- FaceMesh (Mediapipe) ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Función para guardar rostro ---
def guardar_rostro(person_id, rostro, count):
    _, buffer = cv2.imencode(".jpg", rostro)
    rostro_bytes = buffer.tobytes()
    personas.update_one(
        {"_id": person_id},
        {"$push": {"rostros": {"imagen_id": count, "data": rostro_bytes}}}
    )

# --- Captura principal ---
print("📸 Capturando rostros (Presione ESC para salir)")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = frontal_face.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))
    if len(faces) == 0:
        faces = profile_face.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))
        if len(faces) == 0:
            flipped = cv2.flip(gray, 1)
            faces = profile_face.detectMultiScale(flipped, 1.2, 6, minSize=(80, 80))
            for (x, y, w, h) in faces:
                x = frame.shape[1] - x - w
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150))
                guardar_rostro(person_id, rostro, count)
                count += 1
                print(f"📷 Foto {count} guardada (perfil izquierdo)")
                if count >= Maxfotos:
                    break
        else:
            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150))
                guardar_rostro(person_id, rostro, count)
                count += 1
                print(f"📷 Foto {count} guardada (perfil derecho)")
                if count >= Maxfotos:
                    break
    else:
        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150))
            guardar_rostro(person_id, rostro, count)
            count += 1
            print(f"📷 Foto {count} guardada (frontal)")
            if count >= Maxfotos:
                break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

    cv2.imshow("Capturando Rostros", frame)
    k = cv2.waitKey(1)
    if k == 27 or count >= Maxfotos:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Se guardaron {count} fotos de {personName}")

# --- Entrenamiento automático ---
def entrenar_modelo_incremental():
    from pymongo import MongoClient
    import numpy as np
    import cv2
    import os

    client = MongoClient("mongodb://localhost:27017/")
    db = client["rostrosDB"]
    personas = db["personas"]

    labels = []
    facesData = []
    label = 0
    peopleList = []

    for persona in personas.find().sort("nombre", 1):
        nombre = persona["nombre"]
        peopleList.append(nombre)
        if "rostros" in persona:
            for rostro_doc in persona["rostros"]:
                data = rostro_doc["data"]
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    facesData.append(img)
                    labels.append(label)
        label += 1

    model_path = "modeloLBPHFace.xml"
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(model_path):
        face_recognizer.read(model_path)
        face_recognizer.update(facesData, np.array(labels))
        print("Modelo actualizado con nuevos rostros.")
    else:
        face_recognizer.train(facesData, np.array(labels))
        print("Modelo creado desde cero.")
    face_recognizer.write(model_path)
    np.save("labels.npy", np.array(peopleList))
    print("Modelo guardado correctamente.")

entrenar_modelo_incremental()
