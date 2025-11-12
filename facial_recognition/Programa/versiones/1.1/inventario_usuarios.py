import tkinter as tk
from tkinter import ttk, messagebox
from pymongo import MongoClient
from PIL import Image, ImageTk
import os
from utilidades import RUTA_BASE  # ✅ Igual que en tu login


# --- Conexión directa a MongoDB ---
try:
    cliente = MongoClient("mongodb://localhost:27017/")
    db = cliente["rostrosDB"]
    usuarios_collection = db["personas"]
except Exception as e:
    print("❌ Error al conectar con MongoDB:", e)
    usuarios_collection = None


def ver_inventario(root, abrir_menu_principal):
    # Ocultar el menú administrador actual
    root.withdraw()

    # Crear ventana del inventario
    ventana_inv = tk.Toplevel()
    ventana_inv.title("Inventario de Usuarios")
    ventana_inv.configure(bg="#a6c7e9")
    ventana_inv.attributes("-fullscreen", True)

    # ---------- 💠 IMAGEN DE FONDO (usando RUTA_BASE + carpeta imagenes) ----------
    try:
        # Obtener tamaño de pantalla
        sw = ventana_inv.winfo_screenwidth()
        sh = ventana_inv.winfo_screenheight()

        # Ruta absoluta de la imagen base
        ruta_fondo = os.path.join(RUTA_BASE, "imagenes", "base_usuario.png")

        # Cargar y redimensionar
        img = Image.open(ruta_fondo)
        img = img.resize((sw, sh))
        fondo_img = ImageTk.PhotoImage(img)

        fondo_label = tk.Label(ventana_inv, image=fondo_img)
        fondo_label.image = fondo_img  # evitar recolección
        fondo_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print(f"⚠️ No se pudo cargar la imagen de fondo: {e}")

    # ---------- CONTENEDOR CENTRAL ----------
    frame_central = tk.Frame(
        ventana_inv,
        bg="white",
        bd=3,
        relief="solid",
        highlightbackground="#000000",
        highlightthickness=2
    )
    frame_central.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.75, relheight=0.75)

    # ---------- TÍTULO ----------
    tk.Label(
        frame_central,
        text="Inventario de Estudiantes Registrados",
        font=("Segoe UI", 24, "bold"),
        bg="white",
        fg="#111"
    ).pack(pady=(40, 15))

    # ======= FRAME DE FILTRO =======
    filtro_frame = tk.Frame(frame_central, bg="white")
    filtro_frame.pack(pady=(5, 10))

    tk.Label(
        filtro_frame, text="Filtrar por programa:",
        bg="white", font=("Segoe UI", 12, "bold")
    ).pack(side="left", padx=(10, 5))

    combo_programas = ttk.Combobox(filtro_frame, state="readonly", width=40, font=("Segoe UI", 11))
    combo_programas.pack(side="left", padx=5)

    # ---------- TABLA ----------
    columnas = ("cc", "nombre", "programa", "accion")
    tabla = ttk.Treeview(frame_central, columns=columnas, show="headings", height=10)
    tabla.pack(padx=40, pady=20, fill="both", expand=True)

    tabla.heading("cc", text="C.C / Identificación")
    tabla.heading("nombre", text="Nombre Completo")
    tabla.heading("programa", text="Programa de Estudio")
    tabla.heading("accion", text="Acción")

    tabla.column("cc", width=200)
    tabla.column("nombre", width=300)
    tabla.column("programa", width=250)
    tabla.column("accion", width=120, anchor="center")

    # ---------- ESTILO ----------
    estilo = ttk.Style()
    estilo.theme_use("clam")
    estilo.configure(
        "Treeview",
        background="white",
        foreground="black",
        rowheight=30,
        fieldbackground="white",
        font=("Segoe UI", 12)
    )
    estilo.configure(
        "Treeview.Heading",
        background="#4c9aff",
        foreground="black",
        font=("Segoe UI", 13, "bold")
    )

    # ---------- CARGAR DATOS ----------
    def cargar_datos(programa_filtrado=None):
        tabla.delete(*tabla.get_children())
        if usuarios_collection is not None:
            filtro = {}
            if programa_filtrado and programa_filtrado != "Todos":
                filtro["programa"] = programa_filtrado
            usuarios = list(usuarios_collection.find(filtro, {"_id": 0, "cc": 1, "nombre": 1, "programa": 1}))
            if usuarios:
                for usuario in usuarios:
                    tabla.insert("", "end", values=(
                        usuario.get("cc", ""),
                        usuario.get("nombre", ""),
                        usuario.get("programa", ""),
                        "Editar"  # <-- texto que simula botón
                    ))
            else:
                messagebox.showinfo("Sin datos", "No hay usuarios registrados en la base de datos.")
        else:
            messagebox.showerror("Error", "No se pudo conectar con la base de datos.")

    def cargar_programas():
        programas = usuarios_collection.distinct("programa") if usuarios_collection is not None else []
        combo_programas["values"] = ["Todos"] + sorted([p for p in programas if p])
        combo_programas.current(0)

    def filtrar_programa(event):
        seleccionado = combo_programas.get()
        cargar_datos(seleccionado)

    combo_programas.bind("<<ComboboxSelected>>", filtrar_programa)

    # --- Simular botón "Editar" con evento de doble clic ---
    def click_en_tabla(event):
        item = tabla.identify_row(event.y)
        columna = tabla.identify_column(event.x)
        if columna == "#4" and item:  # Columna 'accion'
            usuario = tabla.item(item, "values")
            messagebox.showinfo("Editar", f"Hiciste clic en Editar para:\n{usuario[1]}")

    tabla.bind("<Double-1>", click_en_tabla)

    # Cargar datos iniciales
    cargar_programas()
    cargar_datos()

    # ---------- FUNCIÓN VOLVER ----------
    def volver_menu_admin():
        ventana_inv.destroy()
        root.deiconify()

    # ---------- ESTILO BOTÓN ----------
    def estilo_boton(widget, color_base, color_hover):
        def on_enter(e): widget.config(bg=color_hover)
        def on_leave(e): widget.config(bg=color_base)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # ---------- BOTÓN VOLVER ----------
    btn_volver = tk.Button(
        ventana_inv,  # 👈 colocado sobre la ventana principal
        text="Volver al menú principal",
        command=volver_menu_admin,
        bg="#c62828",
        fg="white",
        font=("Arial", 14, "bold"),
        width=25,
        height=1,
        bd=2,
        relief="solid",
        cursor="hand2",
        highlightthickness=0
    )
    btn_volver.place(relx=0.5, rely=0.93, anchor="center")  # 👈 centrado al fondo
    estilo_boton(btn_volver, "#c62828", "#ff6b6b")

    ventana_inv.protocol("WM_DELETE_WINDOW", volver_menu_admin)
    ventana_inv.mainloop()
