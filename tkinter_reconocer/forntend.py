# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import imutils

# --- Cargamos el reconocedor de caras preentrenado ---
clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    r'haarcascade_frontalface_default.xml'
)

captura = None


# --- Método que reconoce las caras en el vídeo ---
def reconocimientoFacial(ventana):
    gris = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
    caras = clasificador.detectMultiScale(gris, 1.3, 5)
    for (x, y, ancho, alto) in caras:
        ventana = cv2.rectangle(ventana, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
    return ventana


# --- Método que carga el vídeo de entrada ---
def videoDeEntrada():
    global captura
    if seleccionado.get() == 1:
        # Si se selecciona un vídeo desde el ordenador
        ruta_video = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")]
        )
        if len(ruta_video) > 0:
            boton.configure(state="active")
            btnRadio1.configure(state="disabled")
            btnRadio2.configure(state="disabled")
            ruta_video_entrada = "..." + ruta_video[-20:]
            lblInformacionRutaVideo.configure(text=ruta_video_entrada)
            captura = cv2.VideoCapture(ruta_video)
            visualizarVideo()

    elif seleccionado.get() == 2:
        # Si se selecciona la cámara
        boton.configure(state="active")
        btnRadio1.configure(state="disabled")
        btnRadio2.configure(state="disabled")
        lblInformacionRutaVideo.configure(text="Webcam en uso")
        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        visualizarVideo()


# --- Método que muestra el vídeo en la interfaz ---
def visualizarVideo():
    global captura
    if captura is not None:
        ret, ventana = captura.read()
        if ret:
            ventana = imutils.resize(ventana, width=640)
            ventana = reconocimientoFacial(ventana)
            ventana = cv2.cvtColor(ventana, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(ventana)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizarVideo)
        else:
            limpiarPantalla()


# --- Método que finaliza la reproducción ---
def finalizarYLimpiar():
    global captura
    if captura is not None:
        captura.release()
        captura = None
    limpiarPantalla()


# --- Método auxiliar para limpiar la interfaz ---
def limpiarPantalla():
    lblVideo.image = ""
    lblInformacionRutaVideo.configure(text="")
    btnRadio1.configure(state="active")
    btnRadio2.configure(state="active")
    seleccionado.set(0)
    boton.configure(state="disabled")


# --- Interfaz gráfica ---
root = Tk()
root.title("Reproductor de vídeo avanzado")

# Etiqueta superior
lblInformacion1 = Label(root, text="VÍDEO DE ENTRADA", font="bold")
lblInformacion1.grid(column=0, row=0, columnspan=2)

# Radiobuttons
seleccionado = IntVar()
btnRadio1 = Radiobutton(root, text="Elegir vídeo", width=20, value=1, variable=seleccionado, command=videoDeEntrada)
btnRadio2 = Radiobutton(root, text="Vídeo en directo", width=20, value=2, variable=seleccionado, command=videoDeEntrada)
btnRadio1.grid(column=0, row=1)
btnRadio2.grid(column=1, row=1)

# Etiqueta de la ruta del vídeo
lblInformacionRutaVideo = Label(root, text="", width=40)
lblInformacionRutaVideo.grid(column=0, row=2, columnspan=2)

# Etiqueta donde se mostrará el vídeo
lblVideo = Label(root)
lblVideo.grid(column=0, row=3, columnspan=2)

# Botón para finalizar
boton = Button(root, text="Finalizar visualización y limpiar", state="disabled", command=finalizarYLimpiar)
boton.grid(column=0, row=4, columnspan=2, pady=10)

# Iniciar ventana
root.mainloop()
