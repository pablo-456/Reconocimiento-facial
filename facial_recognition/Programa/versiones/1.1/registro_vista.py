import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import sys
import os
from utilidades import RUTA_BASE
from pymongo import MongoClient

# --- Conexión directa a MongoDB ---
try:
    cliente = MongoClient("mongodb://localhost:27017/")
    db = cliente["rostrosDB"]
    usuarios_collection = db["personas"]
except Exception as e:
    print("❌ Error al conectar con MongoDB:", e)
    usuarios_collection = None


def registrar_rostro(root, abrir_menu_principal):
    # Ocultar la ventana principal
    root.withdraw()

    # Crear ventana registro
    ventana_registro = tk.Toplevel(root)
    ventana_registro.title("Registro de Usuario")
    ventana_registro.configure(bg="#a6c7e9")
    ventana_registro.attributes("-fullscreen", True)

    # ---------- CONTENEDOR CENTRAL ----------
    frame_central = tk.Frame(
        ventana_registro,
        bg="white",
        bd=3,
        relief="solid",
        highlightbackground="#000000",
        highlightthickness=2
    )
    frame_central.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.45, relheight=0.75)

    # ---------- TÍTULO ----------
    tk.Label(
        frame_central,
        text="Registro de Estudiante",
        font=("Segoe UI", 24, "bold"),
        bg="white",
        fg="#111"
    ).pack(pady=(40, 15))

    # ---------- CAMPOS ----------
    def crear_campo(texto, variable):
        tk.Label(
            frame_central, text=texto,
            font=("Segoe UI", 13, "bold"),
            bg="white", fg="#222", anchor="w"
        ).pack(fill="x", padx=60, pady=(20, 0))
        tk.Entry(
            frame_central, textvariable=variable,
            font=("Segoe UI", 13),
            width=40, bd=2, relief="solid",
            highlightbackground="#000000",
            highlightthickness=1
        ).pack(pady=5)

    cc_var = tk.StringVar()
    nombre_var = tk.StringVar()
    programa_var = tk.StringVar()

    crear_campo("C.C / Identificación:", cc_var)
    crear_campo("Nombre completo:", nombre_var)

    # ---------- COMBOBOX ----------
    tk.Label(
        frame_central, text="Programa de estudio:",
        font=("Segoe UI", 13, "bold"),
        bg="white", fg="#222", anchor="w"
    ).pack(fill="x", padx=60, pady=(20, 0))

    programas = [
        "Ing. Informática",
        "Ing. Aeronáutica",
        "Administración de Empresas",
        "Contaduría",
        "Derecho",
    ]

    estilo_combo = ttk.Style()
    estilo_combo.theme_use("clam")
    estilo_combo.configure(
        "Custom.TCombobox",
        fieldbackground="white",
        background="white",
        bordercolor="black",
        arrowcolor="black",
        foreground="black",
        font=("Segoe UI", 13)
    )
    estilo_combo.map(
        "Custom.TCombobox",
        fieldbackground=[("readonly", "white")],
        foreground=[("readonly", "black")]
    )

    programa_combo = ttk.Combobox(
        frame_central,
        textvariable=programa_var,
        values=programas,
        state="readonly",
        width=38,
        font=("Segoe UI", 13),
        style="Custom.TCombobox"
    )
    programa_combo.pack(pady=5)
    programa_combo.set("Seleccione un programa")

    # ---------- MENSAJE DE ESTADO ----------
    mensaje_estado = tk.Label(
        frame_central,
        text="",
        bg="white",
        fg="#007acc",
        font=("Segoe UI", 12)
    )
    mensaje_estado.pack(pady=15)

    # ---------- FUNCIÓN ESTILO BOTONES ----------
    def estilo_boton(widget, color_base, color_hover):
        def on_enter(e):
            widget.config(bg=color_hover)
        def on_leave(e):
            widget.config(bg=color_base)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # ---------- BOTÓN INICIAR ----------
    btn_iniciar = tk.Button(
        frame_central,
        text="Iniciar Registro",
        bg="#1a73e8",
        fg="black",
        font=("Segoe UI", 13, "bold"),
        width=15, height=1,
        relief="solid",
        bd=2,
        highlightbackground="#000000",
        cursor="hand2"
    )
    btn_iniciar.pack(pady=(20, 10))
    estilo_boton(btn_iniciar, "#1a73e8", "#1558a6")

    # ---------- BOTÓN VOLVER ----------
    def volver_menu_admin():
        ventana_registro.destroy()
        root.deiconify()  # Muestra el menú admin al instante

    btn_volver = tk.Button(
        frame_central,
        text="Volver al Menú",
        command=volver_menu_admin,
        bg="#c62828",
        fg="black",
        font=("Segoe UI", 13, "bold"),
        width=15, height=1,
        relief="solid",
        bd=2,
        highlightbackground="#000000",
        cursor="hand2"
    )
    btn_volver.pack(pady=(10, 20))
    estilo_boton(btn_volver, "#c62828", "#a31818")

    # ---- FUNCIÓN DE REGISTRO ----
    def iniciar_registro():
        cc = cc_var.get().strip()
        nombre = nombre_var.get().strip()
        programa = programa_var.get().strip()

        if not cc or not nombre or programa == "Seleccione un programa":
            messagebox.showwarning("Advertencia", "Debe completar todos los campos.")
            return

        if usuarios_collection.find_one({"cc": cc}):
            messagebox.showerror("Error", "Ya existe un usuario con esta C.C.")
            return

        messagebox.showinfo(
            "Instrucciones",
            "Por favor, mire a la cámara y realice diferentes gestos:\n\n"
            "✅ Mire al frente\n"
            "✅ Gire la cabeza hacia ambos lados\n"
            "✅ Sonría o haga expresiones faciales\n\n"
            "Presione ESC para terminar la captura."
        )

        btn_iniciar.config(state="disabled")
        btn_volver.config(state="disabled")
        mensaje_estado.config(text="⏳ Cargando... Espere mientras se realiza el registro...")

        def ejecutar_script():
            try:
                ruta_script = os.path.join(RUTA_BASE, "capturadora_optimizada.py")
                subprocess.run([sys.executable, ruta_script, cc, nombre, programa], check=True)
                messagebox.showinfo("Finalizado", f"✅ Usuario {nombre} registrado correctamente.")
            except FileNotFoundError:
                messagebox.showerror("Error", f"No se encontró el archivo:\n{ruta_script}")
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Error", f"Ocurrió un error durante el registro:\n{e}")
            except Exception as db_error:
                messagebox.showerror("Error", f"Error al guardar en MongoDB:\n{db_error}")
            finally:
                ventana_registro.destroy()
                root.deiconify()  # vuelve al menú admin al terminar

        threading.Thread(target=ejecutar_script).start()

    btn_iniciar.config(command=iniciar_registro)

    # --- Mostrar correctamente ---
    ventana_registro.update()
    ventana_registro.lift()
    ventana_registro.focus_force()
