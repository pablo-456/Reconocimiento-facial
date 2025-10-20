import cv2
import imutils
import mediapipe as mp
from pymongo import MongoClient
import numpy as np
import os

# --- Conexión a MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
personas = db["personas"]

#---   OPCIONES DE CÁMARA ---

# 1️⃣ Cámara del PC (predeterminada)
cap = cv2.VideoCapture(0)

# 2️⃣ Cámara externa USB (por ejemplo, iPhone con Iriun o EpocCam por cable)
#cap = cv2.VideoCapture(2)  # o prueba con 2 si hay varias cámaras

if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Solicitar nombre ---
personName = input("Ingrese el nombre de la persona: ").strip().capitalize()
persona_existente = personas.find_one({"nombre": personName})
if persona_existente and "rostros" in persona_existente and len(persona_existente["rostros"]) >= 600:
    print(f"{personName} ya tiene suficientes rostros registrados.")
    exit()

# --- Clasificadores de rostro ---
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

count = 0
Maxfotos = 500

# --- FaceMesh (Mediapipe) ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Función para guardar rostro ---
def guardar_rostro(nombre, rostro, count):
    _, buffer = cv2.imencode(".jpg", rostro)
    rostro_bytes = buffer.tobytes()
    personas.update_one(
        {"nombre": nombre},
        {"$push": {"rostros": {"imagen_id": count, "data": rostro_bytes}}},
        upsert=True
    )

# --- Entrenamiento incremental ---
def entrenar_modelo_incremental():
    from pymongo import MongoClient
    import numpy as np
    import cv2

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

    # Si ya existe, actualizar modelo existente
    model_path = "modeloLBPHFace.xml"
    if os.path.exists(model_path):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(model_path)
        face_recognizer.update(facesData, np.array(labels))
        print("Modelo actualizado con nuevos rostros.")
    else:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        print("Modelo creado desde cero.")

    face_recognizer.write(model_path)
    np.save("labels.npy", np.array(peopleList))
    print("Modelo guardado correctamente.")

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

    # --- Detección frontal ---
    faces = frontal_face.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))

    if len(faces) == 0:
        # --- Perfil derecho ---
        faces = profile_face.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))

        if len(faces) == 0:
            # --- Perfil izquierdo (imagen volteada) ---
            flipped = cv2.flip(gray, 1)
            faces = profile_face.detectMultiScale(flipped, 1.2, 6, minSize=(80, 80))
            for (x, y, w, h) in faces:
                x = frame.shape[1] - x - w
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                guardar_rostro(personName, rostro, count)
                count += 1
                print(f"📷 Foto {count} guardada (perfil izquierdo)")
                if count >= Maxfotos:
                    break
        else:
            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                guardar_rostro(personName, rostro, count)
                count += 1
                print(f"📷 Foto {count} guardada (perfil derecho)")
                if count >= Maxfotos:
                    break
    else:
        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            guardar_rostro(personName, rostro, count)
            count += 1
            print(f"📷 Foto {count} guardada (frontal)")
            if count >= Maxfotos:
                break

    # --- Máscara facial (en todo el frame, no solo ROI) ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Tesselation cubre barbilla y mejillas
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )
            # Contornos para definición facial
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

# --- Liberar recursos ---
cap.release()
cv2.destroyAllWindows()
print(f"✅ Se guardaron {count} fotos de {personName}")

# --- Entrenar automáticamente al finalizar ---
entrenar_modelo_incremental()
