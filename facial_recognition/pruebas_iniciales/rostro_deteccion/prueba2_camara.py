import cv2
import imutils

# Cargar clasificadores
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Paso 1: Buscar rostro frontal ---
    faces = frontal_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    if len(faces) == 0:
        # --- Paso 2: Si no hay rostro frontal, buscar perfil derecho ---
        faces = profile_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

        if len(faces) == 0:
            # --- Paso 3: Si no hay perfil derecho, buscar perfil izquierdo ---
            gray_flipped = cv2.flip(gray, 1)
            faces = profile_face.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))
            # invertir coordenadas del perfil izquierdo
            for (x, y, w, h) in faces:
                x = frame.shape[1] - x - w
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Dibujar perfil derecho
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Dibujar rostro frontal
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detección de Rostros", frame)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
