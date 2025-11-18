import tkinter as tk
import os
from PIL import Image, ImageTk
from utilidades import RUTA_BASE
from registro_vista import registrar_rostro
from inventario_usuarios import ver_inventario
from registro_admin import mostrar_registro  

def mostrar_menu_admin(root_principal):
    # Crea una nueva ventana para el menú administrador
    root_admin = tk.Toplevel()
    root_admin.title("Administración")
    root_admin.attributes("-fullscreen", True)

    ancho_pantalla = root_admin.winfo_screenwidth()
    alto_pantalla = root_admin.winfo_screenheight()

    imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "administrador.png"))
    imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
    fondo_tk = ImageTk.PhotoImage(imagen_fondo)

    # Para evitar que se borre la imagen
    root_admin.imagen_fondo = fondo_tk

    label_fondo = tk.Label(root_admin, image=fondo_tk)
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    # --- Función general para crear botones con hover ---
    def crear_boton(root, texto, comando, color_fondo, color_hover, color_texto, fuente, relx, rely, ancho, alto):
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

    # --- Botones principales ---
    crear_boton(
        root_admin, "Usuarios", lambda: ver_inventario(root_admin, lambda: None),
        color_fondo="#ffffff", color_hover="#f2f2f2", color_texto="black",
        fuente=("Arial", 14, "bold"),
        relx=0.5, rely=0.45, ancho=69, alto=2
    )

    crear_boton(
        root_admin, "Registrar persona", lambda: registrar_rostro(root_admin, lambda: None),
        color_fondo="#ffffff", color_hover="#f2f2f2", color_texto="black",
        fuente=("Arial", 14, "bold"),
        relx=0.5, rely=0.59, ancho=69, alto=2
    )

    #crear_boton(
    #    root_admin, "Movimientos", lambda: None,
    #    color_fondo="#ffffff", color_hover="#f2f2f2", color_texto="black",
    #    fuente=("Arial", 14, "bold"),
    #    relx=0.5, rely=0.85, ancho=69, alto=2
    #)

    crear_boton(
        root_admin, "Registrar admin", lambda: mostrar_registro(root_admin, root_principal),
        color_fondo="#ffffff", color_hover="#f2f2f2", color_texto="black",
        fuente=("Arial", 14, "bold"),
        relx=0.5, rely=0.73, ancho=69, alto=2
    )

    # --- Botón Volver ---
    def volver_menu_principal():
        root_admin.destroy()
        root_principal.deiconify()  # vuelve a mostrar el menú principal sin reiniciarlo

    crear_boton(
        root_admin, "Volver", volver_menu_principal,
        color_fondo="#e63946", color_hover="#ff6b6b", color_texto="white",
        fuente=("Arial", 12, "bold"),
        relx=0.5, rely=0.95, ancho=10, alto=1
    )
