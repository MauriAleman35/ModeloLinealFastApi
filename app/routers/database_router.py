from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import traceback

from app.database.mongodb import get_sync_client

router = APIRouter(prefix="/api", tags=["database"])
logger = logging.getLogger("database_router")

@router.get("/load-data", response_model=Dict[str, Any])
async def load_data():
    """
    Carga datos desde MongoDB y retorna los esquemas de las colecciones
    """
    try:
        # Intentar conectar a MongoDB
        logger.info("Intentando conectar a MongoDB...")
        db = get_sync_client()
        logger.info(f"Conexión exitosa a la base de datos: {db.name}")
        
        # Obtener nombres de todas las colecciones
        collection_names = db.list_collection_names()
        logger.info(f"Colecciones encontradas: {collection_names}")
        
        # Contar documentos por colección
        collection_counts = {}
        schemas = {}
        
        for collection_name in collection_names:
            try:
                # Contar documentos
                count = db[collection_name].count_documents({})
                collection_counts[collection_name] = count
                
                # Obtener un documento de muestra para inferir esquema
                sample = db[collection_name].find_one()
                if sample:
                    # Crear copia del documento sin id
                    sample_clean = {}
                    for k, v in sample.items():
                        if k != "_id":
                            sample_clean[k] = str(v)
                    
                    # Guardar esquema (campos y tipos)
                    schemas[collection_name] = {
                        "fields": [{"name": k, "type": str(type(v).__name__)} for k, v in sample.items() if k != "_id"],
                        "sample": sample_clean
                    }
                else:
                    schemas[collection_name] = {"fields": [], "sample": {}}
                    
            except Exception as coll_error:
                logger.error(f"Error procesando colección {collection_name}: {str(coll_error)}")
                schemas[collection_name] = {"error": str(coll_error)}
        
        return {
            "status": "success",
            "database": db.name,
            "collections": collection_names,
            "document_counts": collection_counts,
            "schemas": schemas
        }
        
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error al cargar datos: {str(e)}"
        )