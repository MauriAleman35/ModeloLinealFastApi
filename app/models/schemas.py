from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    categoria: str
    meses: int = Field(3, ge=1, le=12, description="Número de meses a predecir (1-12)")

class ProductoPrediction(BaseModel):
    producto: str
    proporcion: float
    unidades: int

class MesPrediction(BaseModel):
    año: int
    mes: int
    mes_nombre: str
    demanda_predicha: int
    rango_inferior: int
    rango_superior: int
    productos: List[ProductoPrediction] = []

class PredictionResponse(BaseModel):
    categoria: str
    ultima_fecha: datetime
    predicciones: List[MesPrediction]
    grafico_base64: Optional[str] = None
    metricas: Dict[str, Any] = {}

class CategoryList(BaseModel):
    categorias: List[str]