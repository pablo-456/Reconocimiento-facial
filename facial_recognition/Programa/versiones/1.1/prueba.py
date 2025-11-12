from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["rostrosDB"]
coleccion = db["personas"]
"""
# CREATE
cliente = {"nombre": "Juan", "edad": 25}
insertado = coleccion.insert_one(cliente)
print("Creado con ID:", insertado.inserted_id)

# READ
for c in coleccion.find():
    print(c)

# UPDATE
coleccion.update_one({"nombre": "Juan"}, {"$set": {"edad": 26}})
print("Cliente actualizado")


"""
# DELETE
coleccion.delete_one({"nombre": "Juan"})
print("Cliente eliminado")
