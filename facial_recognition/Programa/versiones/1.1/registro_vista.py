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
    cliente = MongoClient("mongodb://localhost:27017/")  # Cambia si usas otro host o puerto
    db = cliente["registro_usuarios"]  # Nombre de tu base de datos
    usuarios_collection = db["usuarios"]  # Nombre de tu colección
except Exception as e:
    print("❌ Error al conectar con MongoDB:", e)
    usuarios_collection = None

def registrar_rostro(root, abrir_menu_principal):
    # Ocultar ventana principal
    # Configuración de ventana (pantalla completa)
    
    ventana_registro = tk.Toplevel(root)
    ventana_registro.title("Registro de Usuario")
    ventana_registro.configure(bg="#a6c7e9")
    ventana_registro.attributes("-fullscreen", True)
    
    # ---------- CONTENEDOR CENTRAL (más grande) ----------
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

        # --- Estilo personalizado del Combobox ---
    estilo_combo = ttk.Style()
    estilo_combo.theme_use("clam")
    estilo_combo.configure(
        "Custom.TCombobox",
        fieldbackground="white",
        background="white",
        bordercolor="black",
        lightcolor="black",
        darkcolor="black",
        arrowcolor="black",
        foreground="black",
        selectbackground="white",
        selectforeground="black",
        font=("Segoe UI", 13)
    )
    # Evitar el sombreado azul al seleccionar
    estilo_combo.map(
        "Custom.TCombobox",
        fieldbackground=[("readonly", "white")],
        foreground=[("readonly", "black")],
        selectbackground=[("readonly", "white")],
        selectforeground=[("readonly", "black")]
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
        bg="#1a73e8",     # Azul fuerte
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
    btn_volver = tk.Button(
        frame_central,
        text="Volver al Menú",
        command=lambda: [abrir_menu_principal(), ventana_registro.destroy()],
        bg="#c62828",     # Rojo fuerte
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

    # ---- Función de registro ----
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
                abrir_menu_principal()

        threading.Thread(target=ejecutar_script).start()

    btn_iniciar.config(command=iniciar_registro)
    
    
    # --- Forzar que la ventana se dibuje y se muestre encima del root ---
    ventana_registro.update()
    ventana_registro.lift()          # Traer al frente
    ventana_registro.focus_force()   # Dar foco a la ventana

    # Ocultar root **después de que todo se haya dibujado**
    root.withdraw()