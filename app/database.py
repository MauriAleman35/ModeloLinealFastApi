from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "retail_prediction")

# Conexión a MongoDB con manejo de errores
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Verificar conexión
    client.admin.command('ping')
    print(f"✅ Conexión exitosa a MongoDB: {DB_NAME}")
    
except Exception as e:
    print(f"❌ Error al conectar a MongoDB: {e}")
    # Continuar con la aplicación aunque falle la conexión
    # En producción, podrías querer manejar esto de manera diferente
    client = None
    db = None

# Colecciones (con manejo defensivo)
if db:
    ventas = db.ventas
    productos = db.productos
    categorias = db.categorias
    detalles_ventas = db.detalles_ventas
else:
    # Objetos vacíos para evitar errores si la conexión falla
    class EmptyCollection:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                return []
            return method
    
    empty = EmptyCollection()
    ventas = productos = categorias = detalles_ventas = empty