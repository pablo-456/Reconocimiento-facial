import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from utilidades import RUTA_BASE
from conexion_mongo import coleccion_admins  # ✅ conexión a MongoDB


# -------- REGISTRO DE NUEVOS ADMINS --------
def mostrar_registro(ventana_login, ventana_principal):
    registro = tk.Toplevel()
    registro.title("Registrar nuevo administrador")
    registro.attributes("-fullscreen", True)

    # Dimensiones pantalla
    ancho_pantalla = registro.winfo_screenwidth()
    alto_pantalla = registro.winfo_screenheight()

    # Imagen de fondo
    imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "base.png"))
    imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
    fondo_tk = ImageTk.PhotoImage(imagen_fondo)

    label_fondo = tk.Label(registro, image=fondo_tk)
    label_fondo.image = fondo_tk
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    # -------- FRAME PRINCIPAL BLANCO --------
    frame_blanco = tk.Frame(
        registro,
        bg="white",
        bd=3,
        relief="ridge",
        highlightbackground="black",
        highlightthickness=2
    )
    frame_blanco.place(relx=0.5, rely=0.5, anchor="center", width=600, height=450)

    # ---- TÍTULO ----
    tk.Label(
        frame_blanco,
        text="Registrar nuevo administrador",
        font=("Arial", 22, "bold"),
        fg="#1b6fd0",
        bg="white"
    ).pack(pady=30)

    # ---- CAMPOS ----
    contenedor_campos = tk.Frame(frame_blanco, bg="white")
    contenedor_campos.pack(pady=10)

    # Usuario
    tk.Label(contenedor_campos, text="Usuario", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=0, sticky="e", padx=10, pady=8)
    entrada_usuario = tk.Entry(contenedor_campos, font=("Arial", 14), width=25, relief="solid", bd=1)
    entrada_usuario.grid(row=0, column=1, padx=10, pady=8)

    # Contraseña
    tk.Label(contenedor_campos, text="Contraseña", font=("Arial", 14, "bold"), bg="white").grid(row=1, column=0, sticky="e", padx=10, pady=8)
    entrada_contra = tk.Entry(contenedor_campos, font=("Arial", 14), show="*", width=25, relief="solid", bd=1)
    entrada_contra.grid(row=1, column=1, padx=10, pady=8)

    # Email
    tk.Label(contenedor_campos, text="Correo electrónico", font=("Arial", 14, "bold"), bg="white").grid(row=2, column=0, sticky="e", padx=10, pady=8)
    entrada_email = tk.Entry(contenedor_campos, font=("Arial", 14), width=25, relief="solid", bd=1)
    entrada_email.grid(row=2, column=1, padx=10, pady=8)

    # ---- FUNCIÓN DE REGISTRO ----
    def registrar_admin():
        usuario = entrada_usuario.get().strip()
        contra = entrada_contra.get().strip()
        email = entrada_email.get().strip()

        # --- Validaciones personalizadas ---
        if not usuario or not contra or not email:
            messagebox.showwarning("Campos vacíos", "Por favor complete todos los campos.")
            return

        #Usuario con espacios
        if " " in usuario:
            messagebox.showerror("Error en usuario", "El nombre de usuario no puede contener espacios.")
            return

        #Contraseña muy corta
        if len(contra) < 8:
            messagebox.showerror("Error en contraseña", "La contraseña debe tener al menos 8 caracteres.")
            return

        # Correo no institucional
        if not email.endswith("@unisabaneta.edu.co"):
            messagebox.showerror("Error en correo", "Debe ingresar un correo institucional @unisabaneta.edu.co.")
            return

        # --- Validar si el usuario ya existe ---
        existente = coleccion_admins.find_one({"usuario": usuario})
        if existente:
            messagebox.showerror("Error", "El usuario ya existe.")
            return

        nuevo_admin = {
            "usuario": usuario,
            "contrasena": contra,
            "email": email
        }

        try:
            coleccion_admins.insert_one(nuevo_admin)
            messagebox.showinfo("Registro exitoso", "Administrador registrado correctamente.")
            registro.destroy()
            ventana_login.deiconify()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo registrar el administrador.\n{e}")

    # ---- FUNCIÓN VOLVER ----
    def volver_login():
        registro.destroy()
        ventana_login.deiconify()

    # ---- FUNCIÓN PARA CREAR BOTONES ----
    def crear_boton(root, texto, comando, color_fondo, color_hover, color_texto, fuente):
        btn = tk.Button(
            root,
            text=texto,
            command=comando,
            bg=color_fondo,
            fg=color_texto,
            activeforeground=color_texto,
            font=fuente,
            width=15,
            height=1,
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
        return btn

    # ---- BOTONES ----
    contenedor_botones = tk.Frame(frame_blanco, bg="white")
    contenedor_botones.pack(pady=30)

    crear_boton(
        contenedor_botones, "REGISTRAR", registrar_admin,
        color_fondo="#1b6fd0", color_hover="#3b83e3", color_texto="white",
        fuente=("Arial", 13, "bold")
    ).grid(row=0, column=0, padx=20)

    crear_boton(
        contenedor_botones, "VOLVER", volver_login,
        color_fondo="#e63946", color_hover="#ff6b6b", color_texto="white",
        fuente=("Arial", 12, "bold")
    ).grid(row=0, column=1, padx=20)

    registro.mainloop()
