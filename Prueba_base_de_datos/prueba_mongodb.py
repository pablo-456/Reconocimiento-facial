from pymongo import MongoClient

# Conexión local
client = MongoClient("mongodb://localhost:27017/")

# Seleccionar la base de datos
db = client["rostrosDB"]

# Seleccionar la colección
personas = db["personas"]

# Documento de prueba
doc = {
    "nombre": "miguel",
    "edad": 25,
    "descripcion": "Documento de prueba"
}

# Insertar el documento en la colección
resultado = personas.insert_one(doc)

print("Documento insertado con _id:", resultado.inserted_id)
