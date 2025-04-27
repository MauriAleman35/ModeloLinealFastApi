from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Schemas para respuestas de API
class DataUploadResponse(BaseModel):
    success: bool
    message: str
    records_processed: int

class ProductoPrediccion(BaseModel):
    producto: str
    demanda_predicha: float
    valor_estimado: float

class PrediccionMensual(BaseModel):
    año: int
    mes: int
    mes_nombre: str
    demanda_predicha: float
    rango_inferior: float
    rango_superior: float
    productos: Optional[List[ProductoPrediccion]] = None

class PrediccionResponse(BaseModel):
    categoria: str
    predicciones: List[PrediccionMensual]
    metricas: Dict[str, Any]

class TopProducto(BaseModel):
    categoria: str
    producto: str
    demanda_predicha: float
    valor_estimado: float

class TopProductosResponse(BaseModel):
    productos: List[TopProducto]

# Request schemas
class PredictRequest(BaseModel):
    categoria: str
    meses: int = 3
    incluir_productos: bool = True