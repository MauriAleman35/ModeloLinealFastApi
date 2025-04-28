from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging

from app.models.schemas import PredictionRequest, PredictionResponse, CategoryList
from app.services.model_service import predecir_ventas
from app.services.data_service import obtener_categorias

router = APIRouter(prefix="/api", tags=["prediction"])
logger = logging.getLogger("prediction_router")

@router.get("/categories", response_model=CategoryList)
async def get_categories():
    """Obtiene lista de categorías disponibles"""
    categorias = obtener_categorias()
    
    if not categorias:
        raise HTTPException(status_code=404, detail="No se encontraron categorías")
        
    return {"categorias": categorias}

@router.get("/predict/{categoria}", response_model=PredictionResponse)
async def predict_sales(
    categoria: str, 
    meses: int = Query(3, ge=1, le=12, description="Número de meses a predecir")
):
    """
    Predice ventas para una categoría específica
    
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