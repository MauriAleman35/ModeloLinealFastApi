from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
import logging
import os
from bson import ObjectId
from app.database.mongodb import get_sync_client
from app.services.data_service import cargar_datos, procesar_datos, obtener_categorias
from app.services.model_service import analizar_ventas_categoria, preparar_datos_modelo_lineal, entrenar_modelo_lineal

router = APIRouter(prefix="/api", tags=["training"])
logger = logging.getLogger("training_router")

class TrainingRequest(BaseModel):
    categoria: str
    force: bool = False

class TrainingResponse(BaseModel):
    status: str
    categoria: str
    mensaje: str
    detalles: Optional[Dict[str, Any]] = None
    
@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Entrena un modelo para una categoría específica
    
    - **categoria**: ID o Nombre de la categoría a entrenar
    - **force**: Si es True, reentrenará incluso si ya existe un modelo
    """
    try:
        # Obtener mapa de categorías (ID -> Nombre)
        db = get_sync_client()
        categorias_map = {}
        nombres_a_id = {}
        
        categorias_col = db.categorias.find({}, {"_id": 1, "titulo": 1, "nombre": 1})
        for cat in categorias_col:
            cat_id = str(cat.get("_id"))
            
            # Buscar nombre en diferentes campos posibles
            cat_nombre = None
            for field in ["titulo", "nombre", "name", "descripcion"]:
                if field in cat and cat[field]:
                    cat_nombre = cat[field]
                    break
                    
            # Guardar mapeo bidireccional
            if cat_nombre:
                categorias_map[cat_id] = cat_nombre
                nombres_a_id[cat_nombre] = cat_id
        
        # Identificar la categoría solicitada
        categoria_id = None
        categoria_nombre = None
        
        # Caso 1: Es un ID directo
        if request.categoria in categorias_map:
            categoria_id = request.categoria
            categoria_nombre = categorias_map[request.categoria]
            logger.info(f"Categoría encontrada por ID: {categoria_id} -> {categoria_nombre}")
            
        # Caso 2: Es un nombre
        elif request.categoria in nombres_a_id:
            categoria_id = nombres_a_id[request.categoria]
            categoria_nombre = request.categoria
            logger.info(f"Categoría encontrada por nombre: {categoria_nombre} -> {categoria_id}")
            
        # Caso 3: No encontrado
        else:
            # Buscar coincidencia parcial
            for nombre, id in nombres_a_id.items():
                if request.categoria.lower() in nombre.lower():
                    categoria_id = id
                    categoria_nombre = nombre
                    logger.info(f"Categoría encontrada por coincidencia parcial: {request.categoria} -> {categoria_nombre} ({categoria_id})")
                    break
                    
            # Si aún no encontramos, usar la primera
            if categoria_id is None and categorias_map:
                categoria_id = list(categorias_map.keys())[0]
                categoria_nombre = categorias_map[categoria_id]
                logger.warning(f"Categoría no encontrada. Usando primera disponible: {categoria_nombre} ({categoria_id})")
            
            # Si no hay categorías, error
            if categoria_id is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Categoría '{request.categoria}' no encontrada y no hay categorías disponibles."
                )
        
        # Comprobar si el modelo ya existe
        model_path = os.path.join("models", f"lineal_{categoria_id}.joblib")
        if os.path.exists(model_path) and not request.force:
            return TrainingResponse(
                status="skipped",
                categoria=categoria_nombre,
                mensaje=f"El modelo para '{categoria_nombre}' ya existe. Use 'force=True' para reentrenar."
            )
        
        # Iniciar entrenamiento en segundo plano
        background_tasks.add_task(
            _train_model_task,
            categoria_id,
            categoria_nombre
        )
        
        return TrainingResponse(
            status="started",
            categoria=categoria_nombre,
            mensaje=f"Entrenamiento para '{categoria_nombre}' iniciado en segundo plano"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error al iniciar entrenamiento: {str(e)}"
        )

def _train_model_task(categoria_id: str, categoria_nombre: str):
    """Tarea de entrenamiento en segundo plano"""
    try:
        logger.info(f"Iniciando entrenamiento para '{categoria_nombre}' (ID: {categoria_id})")
        
        # Cargar datos
        ventas, detalles, productos, categorias_data = cargar_datos()
        df_unificado = procesar_datos(ventas, detalles, productos, categorias_data)
        
        # Analizar ventas por categoría
        datos_analizados = analizar_ventas_categoria(df_unificado)
        
        # Preparar datos para el modelo
        # Usar el ID como identificador para el modelo
        datos_preparados = preparar_datos_modelo_lineal(datos_analizados, categoria_id)
        
        # Entrenar modelo
        resultado = entrenar_modelo_lineal(datos_preparados, categoria_id)
        
        logger.info(f"✅ Entrenamiento completado para '{categoria_nombre}' (ID: {categoria_id})")
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error en tarea de entrenamiento para {categoria_nombre}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None