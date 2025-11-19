import tkinter as tk
from tkinter import ttk, messagebox
from pymongo import MongoClient
from PIL import Image, ImageTk
import os
from utilidades import RUTA_BASE  


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

    # ----------  IMAGEN DE FONDO ----------
    try:
        sw = ventana_inv.winfo_screenwidth()
        sh = ventana_inv.winfo_screenheight()
        ruta_fondo = os.path.join(RUTA_BASE, "imagenes", "base_usuario.png")
        img = Image.open(ruta_fondo)
        img = img.resize((sw, sh))
        fondo_img = ImageTk.PhotoImage(img)
        fondo_label = tk.Label(ventana_inv, image=fondo_img)
        fondo_label.image = fondo_img
        fondo_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print(f"No se pudo cargar la imagen de fondo: {e}")

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

    tk.Label(
        frame_central,
        text="Inventario de Usuarios Registrados",
        font=("Segoe UI", 24, "bold"),
        bg="white",
        fg="#111"
    ).pack(pady=(40, 15))

    # ======= FILTRO =======
    filtro_frame = tk.Frame(frame_central, bg="white")
    filtro_frame.pack(pady=(5, 10))

    tk.Label(
        filtro_frame, text="Filtrar por programa o cargo:",
        bg="white", font=("Segoe UI", 12, "bold")
    ).pack(side="left", padx=(10, 5))

    combo_programas = ttk.Combobox(filtro_frame, state="readonly", width=40, font=("Segoe UI", 11))
    combo_programas.pack(side="left", padx=5)

    # ---------- TABLA ----------
    columnas = ("cc", "nombre", "programa")
    tabla = ttk.Treeview(frame_central, columns=columnas, show="headings", height=10)
    tabla.pack(padx=40, pady=20, fill="both", expand=True)

    tabla.heading("cc", text="C.C / Identificación")
    tabla.heading("nombre", text="Nombre Completo")
    tabla.heading("programa", text="Programa de Estudio/Cargo")

    tabla.column("cc", width=200)
    tabla.column("nombre", width=300)
    tabla.column("programa", width=250)

    # ---------- ESTILO ----------
    estilo = ttk.Style()
    estilo.theme_use("clam")
    estilo.configure("Treeview", background="white", foreground="black",
                     rowheight=30, fieldbackground="white", font=("Segoe UI", 12))
    estilo.configure("Treeview.Heading", background="#4c9aff", foreground="black",
                     font=("Segoe UI", 13, "bold"))

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
                    ))
            else:
                messagebox.showinfo("Sin datos", "No hay usuarios registrados en la base de datos.")
        else:
            messagebox.showerror("Error", "No se pudo conectar con la base de datos.")

    def cargar_programas():
        programas = usuarios_collection.distinct("programa") if usuarios_collection is not None else []
        combo_programas["values"] = ["Todos"] + sorted([p for p in programas if p])
        combo_programas.current(0)

    combo_programas.bind("<<ComboboxSelected>>", lambda e: cargar_datos(combo_programas.get()))

    # ---- EDITOR EN LÍNEA ----
    editor_widget = None
    boton_guardar = None
    item_editando = None
    col_editando = None

    programas = [
        "Ing. Informática",
        "Ing. Aeronáutica",
        "Administración de Empresas",
        "Contaduría",
        "Derecho",
        "Profesor",
        "Otro Cargo"
    ]

    def iniciar_edicion(event):
        nonlocal editor_widget, boton_guardar, item_editando, col_editando

        # Detectar celda
        item = tabla.identify_row(event.y)
        col = tabla.identify_column(event.x)

        if not item:
            return

        col_index = int(col.replace("#", "")) - 1  # 0=cc,1=nombre,2=programa

        # ❌ La cédula NO se edita
        if col_index == 0:
            return

        # Limpiar ediciones anteriores
        cancelar_edicion()

        item_editando = item
        col_editando = col_index

        valores = list(tabla.item(item, "values"))
        x, y, ancho, alto = tabla.bbox(item, col)

        # -------------------------
        # ✔ NOMBRE → Entry
        # -------------------------
        if col_index == 1:
            editor_widget = tk.Entry(tabla, font=("Segoe UI", 12))
            editor_widget.insert(0, valores[col_index])

        # -------------------------
        # ✔ PROGRAMA → ComboBox
        # -------------------------
        elif col_index == 2:
            editor_widget = ttk.Combobox(
                tabla,
                values=programas,
                state="readonly",
                font=("Segoe UI", 12)
            )
            editor_widget.set(valores[col_index])

        editor_widget.place(x=x, y=y, width=ancho, height=alto)
        editor_widget.focus()

        # Crear botón guardar debajo de la celda
        boton_guardar = tk.Button(
            frame_central,
            text="Guardar cambios",
            font=("Segoe UI", 12, "bold"),
            bg="#4caf50",
            fg="black",
            command=guardar_desde_boton
        )
        boton_guardar.pack(pady=5)

    def cancelar_edicion():
        nonlocal editor_widget, boton_guardar, item_editando, col_editando
        if editor_widget:
            editor_widget.destroy()
            editor_widget = None
        if boton_guardar:
            boton_guardar.destroy()
            boton_guardar = None
        item_editando = None
        col_editando = None

    def guardar_desde_boton():
        nonlocal editor_widget, boton_guardar, item_editando, col_editando

        if not editor_widget or item_editando is None:
            return

        nuevo_valor = editor_widget.get()
        valores = list(tabla.item(item_editando, "values"))
        cc_actual = valores[0]  # cédula NO cambia

        valores[col_editando] = nuevo_valor
        tabla.item(item_editando, values=valores)

        campo_modificado = ["cc", "nombre", "programa"][col_editando]

        usuarios_collection.update_one(
            {"cc": cc_actual},
            {"$set": {campo_modificado: nuevo_valor}}
        )

        messagebox.showinfo("Éxito", "Cambios guardados correctamente.")

        cancelar_edicion()

    # Activar edición con un solo clic
    tabla.bind("<Button-1>", iniciar_edicion)


    # Cargar datos iniciales
    cargar_programas()
    cargar_datos()

    # ---------- VOLVER ----------
    def volver_menu_admin():
        ventana_inv.destroy()
        root.deiconify()

    # ---------- BOTÓN ----------
    def estilo_boton(widget, color_base, color_hover):
        def on_enter(e): widget.config(bg=color_hover)
        def on_leave(e): widget.config(bg=color_base)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    btn_volver = tk.Button(
        ventana_inv,
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
    btn_volver.place(relx=0.5, rely=0.93, anchor="center")
    estilo_boton(btn_volver, "#c62828", "#ff6b6b")

    ventana_inv.protocol("WM_DELETE_WINDOW", volver_menu_admin)
    ventana_inv.mainloop()
