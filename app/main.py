import logging
import os
import json
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime

from app.routers.prediction_router import router as prediction_router
from app.routers.training_router import router as training_router
from app.routers.database_router import router as database_router

from app.utils.json_encoder import JSONEncoder, convert_mongo_objects

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# Crear aplicación FastAPI
app = FastAPI(
    title="Modelo Lineal API",
    description="API para predicción de ventas usando modelos lineales",
    version="1.0.0"
)

# Middleware para convertir ObjectIds en respuestas
@app.middleware("http")
async def convert_objectids_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Procesar solo respuestas JSON
    if isinstance(response, JSONResponse):
        # Obtener contenido
        response_body = json.loads(response.body)
        
        # Convertir tipos de MongoDB
        converted_body = convert_mongo_objects(response_body)
        
        # Crear nueva respuesta
        return JSONResponse(
            content=converted_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
    return response

# Configurar CORS para permitir solicitudes desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear directorios necesarios
os.makedirs("models", exist_ok=True)
os.makedirs("app/static/charts", exist_ok=True)

# Montar directorio estático para servir gráficos
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Incluir routers
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(database_router)


@app.get("/", tags=["status"])
async def root():
    """Endpoint para verificar que la API está funcionando"""
    return {
        "status": "online",
        "mensaje": "API de Predicción de Ventas v1.0"
    }

@app.get("/health", tags=["status"])
async def health_check():
    """Health check para monitoreo"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Iniciando servidor en puerto {port}, debug={debug}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=debug)
# Middleware para convertir tipos no serializables
@app.middleware("http")
async def convert_objectids_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        
        # Procesar solo respuestas JSON
        if isinstance(response, JSONResponse):
            try:
                # Obtener contenido
                response_body = json.loads(response.body)
                
                # Convertir tipos no serializables
                converted_body = convert_mongo_objects(response_body)
                
                # Crear nueva respuesta
                return JSONResponse(
                    content=converted_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            except Exception as e:
                logger.error(f"Error en middleware de conversión: {str(e)}")
                # Retornar respuesta original en caso de error
                return response
                
        return response
    except Exception as e:
        logger.error(f"Error general en middleware: {str(e)}")
        # En caso de error severo, crear una respuesta de error
        return JSONResponse(
            content={"error": "Error interno en el servidor"},
            status_code=500
        )