import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import threading
import sys
import os
from pymongo import MongoClient


# ---------------- CONFIGURACIÓN MONGO ----------------
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["rostrosDB"]
    usuarios_collection = db["personas"]
except Exception as e:
    messagebox.showerror("Error", f"No se pudo conectar a MongoDB:\n{e}")
    sys.exit()

# ---------------- FUNCION AUXILIAR ----------------
def centrar_ventana(ventana, ancho, alto):
    """Centra una ventana en la pantalla según el tamaño indicado."""
    ventana.update_idletasks()
    x = (ventana.winfo_screenwidth() // 2) - (ancho // 2)
    y = (ventana.winfo_screenheight() // 2) - (alto // 2)
    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")

# ---------------- RUTA BASE RELATIVA ----------------
# Detecta automáticamente la carpeta donde está este archivo .py
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))

def abrir_menu_principal():
    """Reabre el menú principal."""
    root.deiconify()

# ---------------- REGISTRO DE USUARIO ----------------
def registrar_rostro():
    root.withdraw()
    ventana_registro = tk.Toplevel(root)
    ventana_registro.title("Registrar Usuario")
    centrar_ventana(ventana_registro, 500, 500)
    ventana_registro.configure(bg="#20232a")

    tk.Label(
        ventana_registro, text="Registro de nuevo usuario",
        font=("Arial", 16, "bold"), bg="#20232a", fg="white"
    ).pack(pady=15)

    # Campo: Cédula
    tk.Label(
        ventana_registro, text="C.C / Identificación:",
        font=("Arial", 12), bg="#20232a", fg="white"
    ).pack(pady=5)
    cc_var = tk.StringVar()
    tk.Entry(ventana_registro, textvariable=cc_var, font=("Arial", 12), width=30).pack(pady=5)

    # Campo: Nombre
    tk.Label(
        ventana_registro, text="Nombre completo:",
        font=("Arial", 12), bg="#20232a", fg="white"
    ).pack(pady=5)
    nombre_var = tk.StringVar()
    tk.Entry(ventana_registro, textvariable=nombre_var, font=("Arial", 12), width=30).pack(pady=5)

    # Campo: Programa de estudio (desplegable)
    tk.Label(
        ventana_registro, text="Programa de estudio:",
        font=("Arial", 12), bg="#20232a", fg="white"
    ).pack(pady=5)

    programas = [
        "Ing. Informática",
        "Ing. Aeronáutica",
        "Administración de Empresas",
        "Contaduría",
        "Derecho",
    ]

    programa_var = tk.StringVar()
    programa_combo = ttk.Combobox(
        ventana_registro, textvariable=programa_var,
        values=programas, font=("Arial", 12), state="readonly", width=28
    )
    programa_combo.pack(pady=5)
    programa_combo.set("Seleccione un programa")

    mensaje_estado = tk.Label(ventana_registro, text="", bg="#20232a", fg="#61afef", font=("Arial", 12))
    mensaje_estado.pack(pady=10)

    # Botones
    btn_iniciar = tk.Button(
        ventana_registro, text="Iniciar Registro",
        bg="#61afef", fg="white", font=("Arial", 12, "bold"), width=20
    )
    btn_iniciar.pack(pady=20)

    btn_volver = tk.Button(
        ventana_registro, text="Volver al Menú",
        command=lambda: [ventana_registro.destroy(), abrir_menu_principal()],
        bg="#e06c75", fg="white", font=("Arial", 12, "bold"), width=20
    )
    btn_volver.pack()

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

        # Deshabilitar botones mientras se ejecuta
        btn_iniciar.config(state="disabled")
        btn_volver.config(state="disabled")
        mensaje_estado.config(text="⏳ Cargando... Espere mientras se realiza el registro...")

        def ejecutar_script():
            try:
                ruta_script = os.path.join(RUTA_BASE, "capturadora_optimizada.py")

                # Enviar los 3 argumentos al script
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

    # Asociar comando después de definirlo
    btn_iniciar.config(command=iniciar_registro)


# ---------------- RECONOCIMIENTO ----------------
def reconocer_rostro():
    messagebox.showinfo(
        "Reconocimiento Facial",
        "Se iniciará la cámara para detectar rostros."
    )

    def ejecutar_reconocimiento():
        try:
            ruta_script = os.path.join(RUTA_BASE, "reconocimiento_optimizado.py")
            subprocess.run([sys.executable, ruta_script], check=True)
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el archivo:\n{ruta_script}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el reconocimiento:\n{e}")
        finally:
            abrir_menu_principal()

    threading.Thread(target=ejecutar_reconocimiento).start()

# ---------------- MENÚ PRINCIPAL ----------------
root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
centrar_ventana(root, 500, 400)
root.configure(bg="#1e1e2f")

tk.Label(
    root,
    text="Sistema de Registro y Reconocimiento Facial",
    font=("Arial", 16, "bold"),
    bg="#1e1e2f", fg="white", wraplength=400
).pack(pady=30)

tk.Button(
    root, text="Registrar Rostro", command=registrar_rostro,
    bg="#98c379", fg="black", font=("Arial", 12, "bold"), width=20, height=2
).pack(pady=15)

tk.Button(
    root, text="Reconocer Rostro", command=reconocer_rostro,
    bg="#56b6c2", fg="black", font=("Arial", 12, "bold"), width=20, height=2
).pack(pady=15)

tk.Button(
    root, text="Salir", command=root.destroy,
    bg="#e06c75", fg="white", font=("Arial", 12, "bold"), width=10, height=1
).pack(pady=30)

root.mainloop()
