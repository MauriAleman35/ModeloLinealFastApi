import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("database")

# Variables de conexión
MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("DB_NAME", "EcommerML")

# Cliente asíncrono para operaciones API
async def get_async_client():
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        yield client[DB_NAME]
        logger.info("Conexión asíncrona a MongoDB establecida")
    except Exception as e:
        logger.error(f"Error conectando a MongoDB: {str(e)}")
        raise

# Cliente sincrónico para operaciones de modelo
def get_sync_client():
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        logger.info("Conexión sincrónica a MongoDB establecida")
        return db
    except Exception as e:
        logger.error(f"Error conectando a MongoDB: {str(e)}")
        raise