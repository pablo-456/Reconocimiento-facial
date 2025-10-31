import subprocess
import threading
import sys
import os
from tkinter import messagebox
from utilidades import RUTA_BASE

def reconocer_rostro(abrir_menu_principal):
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
