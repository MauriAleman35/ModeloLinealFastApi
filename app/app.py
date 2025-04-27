from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import os

from schemas import (
    DataUploadResponse, 
    PredictRequest,
    PrediccionResponse,
    TopProductosResponse
)
from models import (
    procesar_datos,
    entrenar_modelos,
    predecir_demanda,
    obtener_top_productos,
    cargar_modelos_entrenados
)
from database import ventas, productos, categorias, detalles_ventas

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Demanda",
    description="Predice demanda futura por categoría y producto",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de raíz
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Demanda v1.0"}

# 1. Cargar datos
@app.post("/upload/data", response_model=DataUploadResponse)
async def cargar_datos(
    tipo_datos: str = Query(..., description="Tipo de datos: ventas, productos, categorias, detalles"),
    archivo: UploadFile = File(...)
):
    try:
        # Leer archivo
        contenido = await archivo.read()
        
        # Procesar según tipo de archivo
        if archivo.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.StringIO(contenido.decode('utf-8')))
        elif archivo.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(pd.io.BytesIO(contenido))
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
        
        # Seleccionar colección según tipo
        coleccion = None
        if tipo_datos == "ventas":
            coleccion = ventas
        elif tipo_datos == "productos":
            coleccion = productos
        elif tipo_datos == "categorias":
            coleccion = categorias
        elif tipo_datos == "detalles":
            coleccion = detalles_ventas
        else:
            raise HTTPException(status_code=400, detail="Tipo de datos no válido")
        
        # Convertir a diccionarios y guardar
        registros = df.to_dict('records')
        
        # Limpiar y cargar datos
        coleccion.delete_many({})
        if registros:
            coleccion.insert_many(registros)
        
        return DataUploadResponse(
            success=True,
            message=f"Datos de {tipo_datos} cargados exitosamente",
            records_processed=len(registros)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar datos: {str(e)}")

# 2. Entrenar modelos
@app.post("/train")
async def train_models():
    try:
        # Obtener datos desde MongoDB
        ventas_data = list(ventas.find({}, {'_id': 0}))
        productos_data = list(productos.find({}, {'_id': 0}))
        categorias_data = list(categorias.find({}, {'_id': 0}))
        detalles_data = list(detalles_ventas.find({}, {'_id': 0}))
        
        if not ventas_data or not productos_data or not categorias_data or not detalles_data:
            raise HTTPException(
                status_code=400, 
                detail="Faltan datos para entrenar modelos. Asegúrese de cargar todos los datasets."
            )
        
        # Procesar datos
        datos_procesados = procesar_datos(ventas_data, productos_data, categorias_data, detalles_data)
        
        # Entrenar modelos
        resultados = entrenar_modelos(datos_procesados)
        
        return {
            "success": True,
            "message": "Modelos entrenados exitosamente",
            "models_trained": len(resultados),
            "categories": [r["categoria"] for r in resultados],
            "metrics": {r["categoria"]: r["metricas"] for r in resultados}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelos: {str(e)}")

# 3. Predecir demanda por categoría
@app.get("/predict/{categoria}", response_model=PrediccionResponse)
async def predict_demand(
    categoria: str,
    meses: int = 3,
    incluir_productos: bool = True
):
    try:
        # Cargar modelos entrenados
        modelos = cargar_modelos_entrenados()
        
        if not modelos:
            raise HTTPException(
                status_code=404, 
                detail="No hay modelos entrenados. Debe entrenar modelos primero."
            )
        
        if categoria not in modelos:
            categorias_disponibles = list(modelos.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"No hay modelo para la categoría: {categoria}. Categorías disponibles: {categorias_disponibles}"
            )
        
        # Obtener datos necesarios
        productos_data = list(productos.find({}, {'_id': 0}))
        historico_ventas = list(detalles_ventas.find({}, {'_id': 0}))
        
        # Realizar predicción
        resultado = predecir_demanda(
            categoria=categoria,
            num_meses=meses,
            incluir_productos=incluir_productos,
            modelos=modelos,
            productos_data=productos_data,
            historico_ventas=historico_ventas
        )
        
        # Construir respuesta
        return PrediccionResponse(
            categoria=categoria,
            predicciones=resultado["predicciones"],
            metricas=resultado["metricas"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# 4. Obtener top productos
@app.get("/top-products", response_model=TopProductosResponse)
async def top_products(
    limit: int = 10,
    categoria: Optional[str] = None
):
    try:
        # Cargar modelos
        modelos = cargar_modelos_entrenados()
        
        if not modelos:
            raise HTTPException(
                status_code=404, 
                detail="No hay modelos entrenados. Debe entrenar modelos primero."
            )
        
        if categoria and categoria not in modelos:
            categorias_disponibles = list(modelos.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"No hay modelo para la categoría: {categoria}. Categorías disponibles: {categorias_disponibles}"
            )
        
        # Obtener datos
        productos_data = list(productos.find({}, {'_id': 0}))
        historico_ventas = list(detalles_ventas.find({}, {'_id': 0}))
        
        # Obtener top productos
        top_productos = obtener_top_productos(
            modelos=modelos,
            productos_data=productos_data,
            historico_ventas=historico_ventas,
            categoria=categoria,
            limit=limit
        )
        
        return TopProductosResponse(productos=top_productos)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener top productos: {str(e)}")

# Ejecutar con: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)