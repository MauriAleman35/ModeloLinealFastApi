from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from app.database.mongodb import get_sync_client
from app.models.schemas import PredictionResponse, CategoryList
from app.services.model_service import predecir_ventas
from app.services.data_service import obtener_categorias
from app.utils.json_encoder import convert_mongo_objects
router = APIRouter(prefix="/api", tags=["prediction"])
logger = logging.getLogger("prediction_router")

class PredictionBodyRequest(BaseModel):
    categoria: str
    meses: int = Field(3, ge=1, le=12, description="Número de meses a predecir (1-12)")
    incluir_grafico: bool = Field(True, description="Incluir imagen del gráfico en la respuesta")

@router.get("/categories", response_model=Dict[str, Any])
async def get_categories():
    """Obtiene lista de categorías disponibles"""
    try:
        db = get_sync_client()
        categorias_list = []
        
        # Obtener categorías de la colección
        categorias_col = db.categorias.find({}, {"_id": 1, "titulo": 1, "nombre": 1})
        
        for cat in categorias_col:
            cat_id = cat.get("_id")
            
            # Buscar nombre en diferentes campos posibles
            cat_nombre = None
            for field in ["titulo", "nombre", "name", "descripcion"]:
                if field in cat and cat[field]:
                    cat_nombre = cat[field]
                    break
            
            # Si encontramos un nombre, asociarlo con el ID
            if cat_nombre:
                categorias_list.append({
                    "id": str(cat_id),
                    "nombre": cat_nombre
                })
            else:
                categorias_list.append({
                    "id": str(cat_id),
                    "nombre": f"Categoria {str(cat_id)[-6:]}"
                })
        
        return {
            "categorias": categorias_list,
            "total": len(categorias_list)
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo categorías: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/predict/{categoria}", response_model=PredictionResponse)
async def predict_sales(
    categoria: str, 
    meses: int = Query(3, ge=1, le=12, description="Número de meses a predecir")
):
    """
    Predice ventas para una categoría específica (GET)
    
    - **categoria**: Nombre de la categoría a predecir
    - **meses**: Número de meses futuros a predecir (1-12)
    """
    try:
        resultado = predecir_ventas(categoria, meses)
        
        if resultado is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No se pudo generar predicción para {categoria}"
            )
            
        # Construir respuesta
        response = PredictionResponse(
            categoria=resultado['categoria'],
            ultima_fecha=resultado['ultima_fecha'],
            predicciones=[],
            grafico_base64=f"data:image/png;base64,{resultado['grafico_base64']}",
            metricas=resultado['metricas']
        )
        
        # Incluir predicciones mensuales
        for pred_mensual in resultado['predicciones']:
            mes_idx = resultado['predicciones'].index(pred_mensual)
            productos_mes = []
            
            # Buscar productos para este mes
            if mes_idx < len(resultado['prediccion_productos']):
                for prod in resultado['prediccion_productos'][mes_idx]['productos']:
                    productos_mes.append({
                        'producto': prod['producto'],
                        'proporcion': prod['proporcion'],
                        'unidades': prod['unidades']
                    })
            
            # Añadir mes con sus productos
            response.predicciones.append({
                'año': pred_mensual['año'],
                'mes': pred_mensual['mes'],
                'mes_nombre': pred_mensual['mes_nombre'],
                'demanda_predicha': pred_mensual['demanda_predicha'],
                'rango_inferior': pred_mensual['rango_inferior'],
                'rango_superior': pred_mensual['rango_superior'],
                'productos': productos_mes
            })
            
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al generar predicción: {str(e)}"
        )

@router.post("/predict", response_model=Dict[str, Any])
async def predict_sales_body(request: PredictionBodyRequest):
    """
    Predice ventas para una categoría específica (POST)
    
    - **categoria**: ID o nombre de la categoría a predecir
    - **meses**: Número de meses futuros a predecir (1-12)
    - **incluir_grafico**: Si se debe incluir el gráfico en la respuesta
    """
    try:
        resultado = predecir_ventas(request.categoria, request.meses)
        
        if resultado is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No se pudo generar predicción para {request.categoria}"
            )
            
        # Convertir tipos de MongoDB a serializables
        resultado = convert_mongo_objects(resultado)
        
        # Construir respuesta
        response = {
            "categoria": resultado["categoria"],
            "ultima_fecha": resultado["ultima_fecha"],
            "predicciones": resultado["predicciones"],
            "productos_por_mes": resultado["prediccion_productos"],
            "metricas": resultado["metricas"]
        }
        
        # Incluir gráfico si se solicita
        if request.incluir_grafico:
            response["grafico_base64"] = f"data:image/png;base64,{resultado['grafico_base64']}"
            
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción (body): {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al generar predicción: {str(e)}"
        )

@router.get("/charts/{categoria}")
async def get_chart(
    categoria: str, 
    meses: int = Query(3, ge=1, le=12)
):
    """
    Obtiene gráfico de predicción para una categoría
    
    - **categoria**: Nombre de la categoría
    - **meses**: Número de meses a predecir
    """
    resultado = predecir_ventas(categoria, meses)
    
    if resultado is None or 'grafico_base64' not in resultado:
        raise HTTPException(
            status_code=404, 
            detail=f"No se pudo generar gráfico para {categoria}"
        )
            
    return {
        "categoria": categoria,
        "imagen_base64": f"data:image/png;base64,{resultado['grafico_base64']}"
    }