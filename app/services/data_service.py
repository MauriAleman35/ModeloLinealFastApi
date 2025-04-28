import pandas as pd
import numpy as np
from app.database.mongodb import get_sync_client
import logging

logger = logging.getLogger("data_service")

def cargar_datos():
    """Carga datos desde MongoDB"""
    try:
        db = get_sync_client()
        
        # Colecciones del eCommerce
        ventas = pd.DataFrame(list(db.ventas.find()))
        detalles = pd.DataFrame(list(db.detalles_venta.find()))
        productos = pd.DataFrame(list(db.productos.find()))
        categorias = pd.DataFrame(list(db.categorias.find()))
        
        logger.info(f"Datos cargados: {len(ventas)} ventas, {len(detalles)} detalles, "
                   f"{len(productos)} productos, {len(categorias)} categorías")
        
        return ventas, detalles, productos, categorias
    
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        return None, None, None, None

def procesar_datos(ventas, detalles, productos, categorias):
    """Unifica y procesa los datos para análisis"""
    try:
        if ventas is None or detalles is None or productos is None or categorias is None:
            logger.error("No se pueden procesar datos nulos")
            return None
        
        # Convertir fechas
        ventas['fecha'] = pd.to_datetime(ventas['fecha'])
        
        # Unir datasets
        df_detalle_producto = pd.merge(detalles, productos, on='producto_id')
        df_unificado = pd.merge(df_detalle_producto, ventas, on='venta_id')
        
        # Unir con categorías
        if 'categoria_id' in df_unificado.columns and not categorias.empty:
            df_unificado = pd.merge(df_unificado, categorias, on='categoria_id', how='left')
            
        # Columna de categoría
        if 'categoria_nombre' not in df_unificado.columns and 'categoria' in df_unificado.columns:
            df_unificado['categoria_nombre'] = df_unificado['categoria']
            
        # Columna de producto
        if 'producto_nombre' not in df_unificado.columns and 'producto' in df_unificado.columns:
            df_unificado['producto_nombre'] = df_unificado['producto']
        
        logger.info(f"Datos procesados: {len(df_unificado)} registros unificados")
        return df_unificado
    
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        return None

def obtener_categorias():
    """Obtiene lista de categorías disponibles"""
    try:
        ventas, detalles, productos, categorias = cargar_datos()
        df_unificado = procesar_datos(ventas, detalles, productos, categorias)
        
        if df_unificado is None:
            return []
        
        # Identificar columna de categoría
        if 'categoria_nombre' in df_unificado.columns:
            cat_column = 'categoria_nombre'
        else:
            cat_column = 'categoria'
            
        # Contar ventas por categoría
        categoria_ventas = df_unificado.groupby(cat_column)['cantidad'].sum().sort_values(ascending=False)
        
        return categoria_ventas.index.tolist()
    except Exception as e:
        logger.error(f"Error obteniendo categorías: {str(e)}")
        return []