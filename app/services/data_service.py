import pandas as pd
import numpy as np
from app.database.mongodb import get_sync_client
import logging
import random
from datetime import datetime, timedelta
from app.services.cache_service import cache

logger = logging.getLogger("data_service")

def cargar_datos():
    """Carga datos desde MongoDB"""
    try:
        db = get_sync_client()
        
        # Colecciones del eCommerce
        ventas = pd.DataFrame(list(db.ventas.find()))
        
        # Intentar cargar detalles de diferentes colecciones posibles
        detalles = pd.DataFrame()
        for col_name in ['ventadetalles', 'detalles_venta', 'venta_detalles', 'detalles']:
            try:
                if col_name in db.list_collection_names():
                    logger.info(f"Intentando cargar detalles desde colección: {col_name}")
                    detalles = pd.DataFrame(list(db[col_name].find()))
                    if len(detalles) > 0:
                        logger.info(f"Detalles cargados exitosamente desde {col_name}: {len(detalles)} registros")
                        break
            except Exception as e:
                logger.warning(f"Error al cargar detalles desde {col_name}: {str(e)}")
        
        productos = pd.DataFrame(list(db.productos.find()))
        categorias = pd.DataFrame(list(db.categorias.find()))
        
        logger.info(f"Datos cargados: {len(ventas)} ventas, {len(detalles)} detalles, "
                   f"{len(productos)} productos, {len(categorias)} categorías")
        
        # Si no hay detalles de venta, crearlos sintéticamente
        if len(detalles) == 0 and len(ventas) > 0 and len(productos) > 0:
            logger.warning("No se encontraron detalles de venta. Creando datos sintéticos para demostración...")
            detalles = generar_detalles_sinteticos(ventas, productos)
            logger.info(f"Se generaron {len(detalles)} detalles sintéticos")
        else:
            # Mapear columnas según el esquema detectado
            if 'producto' in detalles.columns and 'venta' in detalles.columns:
                logger.info("Adaptando esquema de ventadetalles...")
                detalles['producto_id'] = detalles['producto'].astype(str)
                detalles['venta_id'] = detalles['venta'].astype(str)
                
                # Si existe precio y cantidad, calcular subtotal
                if 'precio' in detalles.columns and 'cantidad' in detalles.columns:
                    detalles['precio_unitario'] = detalles['precio']
                    detalles['subtotal'] = detalles['precio'] * detalles['cantidad']
                    logger.info("Subtotal calculado a partir de precio y cantidad")
        
        return ventas, detalles, productos, categorias
    
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
def generar_detalles_sinteticos(ventas, productos):
    """Genera detalles de venta sintéticos para demostración"""
    try:
        # Preparar columnas necesarias
        if '_id' in ventas.columns:
            ventas['venta_id'] = ventas['_id'].astype(str)
        elif 'id' in ventas.columns:
            ventas['venta_id'] = ventas['id'].astype(str)
        
        if '_id' in productos.columns:
            productos['producto_id'] = productos['_id'].astype(str)
        elif 'id' in productos.columns:
            productos['producto_id'] = productos['id'].astype(str)
            
        # Preparar datos para detalles
        detalles_data = []
        
        # Convertir fecha de venta a datetime si es string
        if 'fecha' in ventas.columns:
            if isinstance(ventas['fecha'].iloc[0], str):
                ventas['fecha'] = pd.to_datetime(ventas['fecha'])
        
        # Generar detalles de venta aleatorios
        for _, venta in ventas.iterrows():
            # Para cada venta, generar entre 1 y 5 productos
            num_productos = random.randint(1, 5)
            productos_vendidos = productos.sample(n=min(num_productos, len(productos)))
            
            for _, producto in productos_vendidos.iterrows():
                cantidad = random.randint(1, 10)
                precio = float(random.randint(50, 2000))/10.0  # Precio entre 5 y 200
                
                detalle = {
                    'venta_id': venta['venta_id'],
                    'producto_id': producto['producto_id'],
                    'cantidad': cantidad,
                    'precio_unitario': precio,
                    'subtotal': cantidad * precio
                }
                
                # Añadir información de categoría si está disponible
                if 'categoria_id' in producto:
                    detalle['categoria_id'] = producto['categoria_id']
                
                detalles_data.append(detalle)
        
        # Crear DataFrame con los datos generados
        detalles_df = pd.DataFrame(detalles_data)
        
        return detalles_df
        
    except Exception as e:
        logger.error(f"Error generando datos sintéticos: {str(e)}")
        import traceback
        traceback.print_exc()
        # Retornar dataframe vacío
        return pd.DataFrame()

def procesar_datos(ventas, detalles, productos, categorias):
    """Unifica y procesa los datos para análisis"""
    try:
        if ventas is None or len(ventas) == 0:
            logger.error("No se encontraron datos de ventas")
            return None
            
        if detalles is None or len(detalles) == 0:
            logger.error("No se encontraron detalles de ventas")
            return None
            
        if productos is None or len(productos) == 0:
            logger.error("No se encontraron datos de productos")
            return None
        
        # Mapear columnas según el esquema detectado
        # Columna que conecta detalles con productos
        if 'producto_id' not in detalles.columns and 'producto' in detalles.columns:
            detalles['producto_id'] = detalles['producto'].astype(str)
            logger.info("Mapeando 'producto' a 'producto_id'")
            
        # Columna que conecta detalles con ventas
        if 'venta_id' not in detalles.columns and 'venta' in detalles.columns:
            detalles['venta_id'] = detalles['venta'].astype(str)
            logger.info("Mapeando 'venta' a 'venta_id'")
            
        # Ajustar columnas de productos
        if '_id' in productos.columns:
            productos['producto_id'] = productos['_id'].astype(str)
            logger.info("Mapeando '_id' de productos a 'producto_id'")
            
        # Ajustar columnas de ventas
        if '_id' in ventas.columns:
            ventas['venta_id'] = ventas['_id'].astype(str)
            logger.info("Mapeando '_id' de ventas a 'venta_id'")
            
        # Verificar fecha de venta
        if 'fecha' not in ventas.columns and 'createdAT' in ventas.columns:
            ventas['fecha'] = pd.to_datetime(ventas['createdAT'])
            logger.info("Usando 'createdAT' como fecha de venta")
        elif 'fecha' not in ventas.columns and 'createdAt' in ventas.columns:
            ventas['fecha'] = pd.to_datetime(ventas['createdAt'])
            logger.info("Usando 'createdAt' como fecha de venta")
        elif 'fecha' not in ventas.columns:
            logger.warning("No hay columna 'fecha' en ventas. Creando fecha sintética.")
            # Crear fechas sintéticas
            fechas = []
            fecha_base = datetime.now() - timedelta(days=365)
            for _ in range(len(ventas)):
                dias_aleatorios = random.randint(1, 365)
                fechas.append(fecha_base + timedelta(days=dias_aleatorios))
            ventas['fecha'] = pd.Series(fechas)
        
        # Resto del código existente...
        # Convertir fechas
        if 'fecha' in ventas.columns:
            try:
                ventas['fecha'] = pd.to_datetime(ventas['fecha'])
                logger.info("Fecha convertida exitosamente")
            except Exception as e:
                logger.warning(f"Error al convertir fechas: {e}")
                logger.warning("Creando fecha sintética.")
                fechas = []
                fecha_base = datetime.now() - timedelta(days=365)
                for _ in range(len(ventas)):
                    dias_aleatorios = random.randint(1, 365)
                    fechas.append(fecha_base + timedelta(days=dias_aleatorios))
                ventas['fecha'] = pd.Series(fechas)
            
        # Unir datasets
        try:
            logger.info("Uniendo detalles con productos...")
            df_detalle_producto = pd.merge(detalles, productos, on='producto_id', how='left')
            logger.info(f"JOIN detalle-producto: {len(df_detalle_producto)} filas")
            
            logger.info("Uniendo con ventas...")
            df_unificado = pd.merge(df_detalle_producto, ventas, on='venta_id', how='left')
            logger.info(f"JOIN completo: {len(df_unificado)} filas")
        except Exception as e:
            logger.error(f"Error en JOIN: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        # Manejar columnas de categoría
        if not 'categoria_id' in df_unificado.columns and 'categoria' in productos.columns:
            df_unificado['categoria_id'] = df_unificado['categoria']
            logger.info("Usando 'categoria' como 'categoria_id'")
            
        if 'categoria_id' in df_unificado.columns:
            # Convertir ObjectId a string si es necesario
            df_unificado['categoria_id'] = df_unificado['categoria_id'].astype(str)
            
            if categorias is not None and len(categorias) > 0:
                if '_id' in categorias.columns:
                    categorias['categoria_id'] = categorias['_id'].astype(str)
                    
                # Buscar columna con nombre de categoría
                nombre_col = None
                for col in ['nombre', 'name', 'titulo', 'descripcion']:
                    if col in categorias.columns:
                        nombre_col = col
                        break
                        
                if nombre_col:
                    # Unir con categorías para obtener nombre
                    categorias_map = dict(zip(categorias['categoria_id'], categorias[nombre_col]))
                    df_unificado['categoria_nombre'] = df_unificado['categoria_id'].map(categorias_map)
                    logger.info(f"Se asignaron nombres a {df_unificado['categoria_nombre'].notna().sum()} categorías")
        
        # Añadir columna de categoría si no existe
        if 'categoria_nombre' not in df_unificado.columns:
            if 'categoria' in df_unificado.columns:
                df_unificado['categoria_nombre'] = df_unificado['categoria']
            else:
                logger.warning("No se encontró columna de categoría. Asignando valores por defecto.")
                # Crear categorías aleatorias
                categorias_demo = ["Electrónica", "Ropa", "Accesorios", "Hogar", "Deportes"]
                df_unificado['categoria_nombre'] = [random.choice(categorias_demo) for _ in range(len(df_unificado))]
        
        logger.info(f"Procesamiento completado: {len(df_unificado)} registros")
        return df_unificado
    
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def obtener_categorias():
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
        
        logger.info(f"Se encontraron {len(categorias_list)} categorías")
        
        # Devolver solo los IDs (compatibilidad con el código existente)
        return [cat["id"] for cat in categorias_list]
        
    except Exception as e:
        logger.error(f"Error obteniendo categorías: {str(e)}")
        return []