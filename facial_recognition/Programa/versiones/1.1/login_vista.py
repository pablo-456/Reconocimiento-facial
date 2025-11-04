import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from utilidades import RUTA_BASE
from conexion_mongo import coleccion_admins
import menu_administrador
from recuperacion_contrasena import mostrar_recuperacion  # 🔹 nuevo import

def mostrar_login(ventana_principal):
    # --- Ventana principal del login ---
    login = tk.Toplevel()
    login.title("Inicio de Sesión - Administrador")
    login.attributes("-fullscreen", True)
    login.configure(bg="#a6c7e9")

    # --- Imagen de fondo ---
    ancho_pantalla = login.winfo_screenwidth()
    alto_pantalla = login.winfo_screenheight()
    imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "base.png"))
    imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
    fondo_tk = ImageTk.PhotoImage(imagen_fondo)

    label_fondo = tk.Label(login, image=fondo_tk)
    label_fondo.image = fondo_tk
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    # --- Contenedor central ---
    frame_central = tk.Frame(
        login,
        bg="white",
        bd=3,
        relief="solid",
        highlightbackground="#000000",
        highlightthickness=2
    )
    frame_central.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.4, relheight=0.55)

    # --- Título ---
    tk.Label(
        frame_central,
        text="Inicio de Sesión",
        font=("Segoe UI", 26, "bold"),
        bg="white",
        fg="#0d1b2a"
    ).pack(pady=(40, 20))

    # --- Campos de entrada ---
    def crear_campo(texto, show=None):
        tk.Label(
            frame_central,
            text=texto,
            font=("Segoe UI", 13, "bold"),
            bg="white",
            fg="#222",
            anchor="w"
        ).pack(fill="x", padx=60, pady=(10, 0))

        entry = tk.Entry(
            frame_central,
            font=("Segoe UI", 13),
            width=35,
            bd=2,
            relief="solid",
            show=show,
            highlightbackground="#000000",
            highlightthickness=1
        )
        entry.pack(pady=5)
        return entry

    entrada_usuario = crear_campo("Usuario:")
    entrada_contra = crear_campo("Contraseña:", show="*")

    # --- Función para efecto hover ---
    def estilo_boton(widget, color_base, color_hover):
        def on_enter(e): widget.config(bg=color_hover)
        def on_leave(e): widget.config(bg=color_base)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # --- Funciones de acción ---
    def verificar_login():
        usuario = entrada_usuario.get().strip()
        contra = entrada_contra.get().strip()

        if not usuario or not contra:
            messagebox.showwarning("Campos vacíos", "Por favor complete todos los campos.")
            return
            # 🔹 Verificar primero el usuario y contraseña predeterminados
        if usuario == "Admin00" and contra == "12345678":
            messagebox.showinfo("Acceso concedido", "Bienvenido, administrador principal.")
            login.destroy()
            menu_administrador.mostrar_menu_admin(ventana_principal)
            return
        admin = coleccion_admins.find_one({"usuario": usuario, "contrasena": contra})
        if admin:
            messagebox.showinfo("Acceso concedido", f"Bienvenido, {usuario}.")
            login.destroy()
            menu_administrador.mostrar_menu_admin(ventana_principal)
        else:
            messagebox.showerror("Error", "Usuario o contraseña incorrectos.")

    def volver_menu():
        login.destroy()
        ventana_principal.deiconify()

    def abrir_recuperacion():
        login.withdraw()
        mostrar_recuperacion(login, ventana_principal)

    # --- Botón de ingresar ---
    btn_ingresar = tk.Button(
        frame_central,
        text="Ingresar",
        command=verificar_login,
        bg="#1a73e8",
        fg="white",
        font=("Segoe UI", 13, "bold"),
        width=15,
        height=1,
        relief="solid",
        bd=2,
        cursor="hand2"
    )
    btn_ingresar.pack(pady=(40, 10))
    estilo_boton(btn_ingresar, "#1a73e8", "#1558a6")



    # --- Texto de recuperar contraseña ---
    lbl_recuperar = tk.Label(
        frame_central,
        text="¿Olvidaste tu contraseña?",
        font=("Segoe UI", 11, "underline"),
        fg="#1a73e8",
        bg="white",
        cursor="hand2"
    )
    lbl_recuperar.pack()
    lbl_recuperar.bind("<Button-1>", lambda e: abrir_recuperacion())

    # --- Separador visual ---
    tk.Frame(frame_central, height=2, bg="#ccc").pack(fill="x", pady=(15, 25))

    # --- Botón rojo (Volver) en la esquina inferior derecha ---
    btn_volver = tk.Button(
        login,  # 🔹 Está directamente en la ventana, no dentro del frame blanco
        text="Volver",
        command=volver_menu,
        bg="#c62828",
        fg="white",
        font=("Segoe UI", 12, "bold"),
        width=10,
        height=1,
        relief="solid",
        bd=2,
        cursor="hand2"
    )

    # 🔹 Lo colocamos en la esquina inferior derecha
    btn_volver.place(relx=0.90, rely=0.93)  # Ajusta si lo quieres más al borde (ej. 0.92 / 0.95)

    # --- Efecto hover ---
    estilo_boton(btn_volver, "#c62828", "#a31818")

    login.update()
    login.lift()
    login.focus_force()
    login.mainloop()
