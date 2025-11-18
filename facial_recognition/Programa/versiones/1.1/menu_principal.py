import tkinter as tk
import os
from PIL import Image, ImageTk
from utilidades import RUTA_BASE
from reconocimiento_vista import reconocer_rostro
import login_vista


def abrir_login(root):
    """Oculta el menú principal y abre la ventana de login"""
    root.withdraw()
    login_vista.mostrar_login(root)


def crear_menu_principal():
    """Crea y retorna la ventana principal sin ejecutarla"""

    root = tk.Tk()
    root.title("Access Smart")
    root.attributes("-fullscreen", True)

    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()

    imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "menu.png"))
    imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
    fondo_tk = ImageTk.PhotoImage(imagen_fondo)

    label_fondo = tk.Label(root, image=fondo_tk)
    label_fondo.image = fondo_tk  # evitar recolección de basura
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    def crear_boton(texto, comando, color_fondo, color_hover, color_texto, fuente, relx, rely, ancho, alto):
        btn = tk.Button(
            root,
            text=texto,
            command=comando,
            bg=color_fondo,
            fg=color_texto,
            activeforeground=color_texto,
            font=fuente,
            width=ancho,
            height=alto,
            relief="solid",
            bd=2,
            highlightthickness=0,
            cursor="hand2"
        )

        def on_enter(e):
            btn.config(bg=color_hover, relief="raised", bd=3)

        def on_leave(e):
            btn.config(bg=color_fondo, relief="solid", bd=2)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.place(relx=relx, rely=rely, anchor="center")
        return btn

    # --- Botones ---
    crear_boton(
        "ENTRAR",
        lambda: reconocer_rostro(lambda: None),
        "#4cc9f0", "#72d6f9", "black",
        ("Arial", 14, "bold"),
        relx=0.833, rely=0.475, ancho=28, alto=2
    )

    crear_boton(
        "ENTRAR",
        lambda: abrir_login(root),
        "#1b6fd0", "#3b83e3", "black",
        ("Arial", 14, "bold"),
        relx=0.833, rely=0.775, ancho=28, alto=2
    )

    crear_boton(
        "SALIR",
        root.destroy,
        "#e63946", "#ff6b6b", "white",
        ("Arial", 12, "bold"),
        relx=0.95, rely=0.95, ancho=10, alto=1
    )

    return root
