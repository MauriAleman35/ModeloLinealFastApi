import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.routers.prediction import router as prediction_router

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

# Configurar CORS para permitir solicitudes desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar directorio estático para servir gráficos
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Incluir routers
app.include_router(prediction_router)

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