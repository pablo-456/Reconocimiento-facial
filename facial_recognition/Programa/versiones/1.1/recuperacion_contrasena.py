import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
from conexion_mongo import coleccion_admins
from utilidades import RUTA_BASE  # Asegúrate de tener esta ruta definida

def mostrar_recuperacion(login, ventana_principal):
    rec = tk.Toplevel()
    rec.title("Recuperar Contraseña")
    rec.attributes("-fullscreen", True)
    rec.configure(bg="#a6c7e9")

    # --- Imagen de fondo ---
    ancho_pantalla = rec.winfo_screenwidth()
    alto_pantalla = rec.winfo_screenheight()
    imagen_fondo = Image.open(os.path.join(RUTA_BASE, "imagenes", "base.png"))
    imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla))
    fondo_tk = ImageTk.PhotoImage(imagen_fondo)

    label_fondo = tk.Label(rec, image=fondo_tk)
    label_fondo.image = fondo_tk
    label_fondo.place(x=0, y=0, relwidth=1, relheight=1)

    # --- Frame central ---
    frame_rec = tk.Frame(
        rec,
        bg="white",
        bd=3,
        relief="solid",
        highlightbackground="#000000",
        highlightthickness=2
    )
    frame_rec.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.35, relheight=0.4)

    # --- Título ---
    tk.Label(
        frame_rec,
        text="Recuperar Contraseña",
        font=("Segoe UI", 22, "bold"),
        bg="white",
        fg="#0d1b2a"
    ).pack(pady=(35, 15))

    # --- Campo de entrada ---
    tk.Label(
        frame_rec,
        text="Ingrese su usuario:",
        font=("Segoe UI", 13, "bold"),
        bg="white",
        fg="#222"
    ).pack(pady=(10, 5))

    entry_usuario = tk.Entry(
        frame_rec,
        font=("Segoe UI", 13),
        width=30,
        bd=2,
        relief="solid",
        highlightbackground="#000000",
        highlightthickness=1
    )
    entry_usuario.pack(pady=10, ipady=5)

    # --- Función para efecto hover ---
    def estilo_boton(widget, color_base, color_hover):
        def on_enter(e): widget.config(bg=color_hover)
        def on_leave(e): widget.config(bg=color_base)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # --- Recuperar contraseña ---
    def recuperar():
        usuario = entry_usuario.get().strip()
        if not usuario:
            messagebox.showwarning("Advertencia", "Ingrese su usuario.")
            return

        admin = coleccion_admins.find_one({"usuario": usuario})
        if admin:
            messagebox.showinfo("Recuperación", f"Tu contraseña es: {admin.get('contrasena')}")
            rec.destroy()
            login.deiconify()  # 🔹 Vuelve automáticamente al login
        else:
            messagebox.showerror("Error", "Usuario no encontrado.")

    # --- Botón recuperar ---
    btn_recuperar = tk.Button(
        frame_rec,
        text="Recuperar",
        command=recuperar,
        bg="#42a5f5",
        fg="white",
        font=("Segoe UI", 13, "bold"),
        width=15,
        height=1,
        relief="solid",
        bd=2,
        cursor="hand2"
    )
    btn_recuperar.pack(pady=(25, 10))
    estilo_boton(btn_recuperar, "#42a5f5", "#1e88e5")

    # --- Botón volver ---
    def volver():
        rec.destroy()
        login.deiconify()

    btn_volver = tk.Button(
        frame_rec,
        text="Volver",
        command=volver,
        bg="#c62828",
        fg="white",
        font=("Segoe UI", 12, "bold"),
        width=10,
        height=1,
        relief="solid",
        bd=2,
        cursor="hand2"
    )
    btn_volver.pack(pady=(10, 20))
    estilo_boton(btn_volver, "#c62828", "#a31818")

    rec.update()
    rec.lift()
    rec.focus_force()
    rec.mainloop()
