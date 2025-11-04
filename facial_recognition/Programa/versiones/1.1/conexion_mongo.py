from pymongo import MongoClient

try:
    # 🔹 Conexión local a MongoDB
    cliente = MongoClient("mongodb://localhost:27017/")

    # 🔹 Nombre de la base de datos
    db = cliente["rostrosDB"]

    # 🔹 Colección para los administradores
    coleccion_admins = db["admins"]

    print("✅ Conexión exitosa a MongoDB")

except Exception as e:
    print("❌ Error al conectar con MongoDB:", e)
    coleccion_admins = None
