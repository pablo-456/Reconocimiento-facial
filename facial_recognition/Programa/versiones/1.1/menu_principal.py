import tkinter as tk
import os
from PIL import Image, ImageTk
from utilidades import RUTA_BASE
from reconocimiento_vista import reconocer_rostro
import menu_administrador  # 👈 Importamos el otro módulo directamente


def abrir_menu_admin():
    # Oculta la ventana actual y abre el menú de administrador
    root.withdraw()
    menu_administrador.mostrar_menu_admin(root)


# ---------------- MENÚ PRINCIPAL ----------------
root = tk.Tk()
root.title("Access Smart")
root.attributes("-fullscreen", True)

ancho_pantalla = root.winfo_screenwidth()
alto_pantalla = root.winfo_screenheight()

imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "menu.png"))
imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
fondo_tk = ImageTk.PhotoImage(imagen_fondo)

label_fondo = tk.Label(root, image=fondo_tk)
label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

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


btn_verificacion = crear_boton(
    root, "ENTRAR", lambda: reconocer_rostro(lambda: None),
    color_fondo="#4cc9f0", color_hover="#72d6f9", color_texto="black",
    fuente=("Arial", 14, "bold"),
    relx=0.833, rely=0.475, ancho=25, alto=2
)

btn_admin = crear_boton(
    root, "ENTRAR", abrir_menu_admin,
    color_fondo="#1b6fd0", color_hover="#3b83e3", color_texto="black",
    fuente=("Arial", 14, "bold"),
    relx=0.833, rely=0.775, ancho=25, alto=2
)

btn_salir = crear_boton(
    root, "SALIR", root.destroy,
    color_fondo="#e63946", color_hover="#ff6b6b", color_texto="white",
    fuente=("Arial", 12, "bold"),
    relx=0.95, rely=0.95, ancho=10, alto=1
)

root.mainloop()
