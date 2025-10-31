import os

def centrar_ventana(ventana, ancho, alto):
    """Centra una ventana en la pantalla según el tamaño indicado."""
    ventana.update_idletasks()
    x = (ventana.winfo_screenwidth() // 2) - (ancho // 2)
    y = (ventana.winfo_screenheight() // 2) - (alto // 2)
    ventana.geometry(f"{ancho}x{alto}+{x}+{y}")

# Detecta automáticamente la carpeta base donde está el archivo principal
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
