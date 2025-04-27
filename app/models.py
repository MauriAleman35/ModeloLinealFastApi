import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb
from datetime import datetime, timedelta
import calendar
from typing import Dict, List, Any, Optional

# Directorio para modelos
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def procesar_datos(ventas, productos, categorias, detalles):
    """
    Procesa los datos para entrenar modelos de predicción
    """
    # Convertir a DataFrames
    ventas_df = pd.DataFrame(ventas)
    productos_df = pd.DataFrame(productos)
    categorias_df = pd.DataFrame(categorias)
    detalles_df = pd.DataFrame(detalles)
    
    # Convertir fechas
    ventas_df['fecha'] = pd.to_datetime(ventas_df['fecha'])
    
    # Unir datos
    datos_completos = detalles_df.merge(ventas_df, left_on='venta_id', right_on='id', suffixes=('', '_venta'))
    datos_completos = datos_completos.merge(productos_df, left_on='producto_id', right_on='id', suffixes=('', '_producto'))
    datos_completos = datos_completos.merge(categorias_df, left_on='categoria_id', right_on='id', suffixes=('', '_categoria'))
    
    # Extraer columnas de interés
    datos_procesados = datos_completos[
        ['fecha', 'nombre_categoria', 'nombre_producto', 'cantidad', 'precio_unitario']
    ].rename(columns={
        'nombre_categoria': 'categoria_nombre',
        'nombre_producto': 'producto_nombre',
        'precio_unitario': 'precio'
    })
    
    # Características temporales
    datos_procesados['año'] = datos_procesados['fecha'].dt.year
    datos_procesados['mes'] = datos_procesados['fecha'].dt.month
    datos_procesados['dia'] = datos_procesados['fecha'].dt.day
    datos_procesados['dia_semana'] = datos_procesados['fecha'].dt.dayofweek
    datos_procesados['dia_año'] = datos_procesados['fecha'].dt.dayofyear
    
    # Características cíclicas
    datos_procesados['mes_sen'] = np.sin(2 * np.pi * datos_procesados['mes'] / 12)
    datos_procesados['mes_cos'] = np.cos(2 * np.pi * datos_procesados['mes'] / 12)
    
    return datos_procesados

def preparar_caracteristicas(datos_por_categoria):
    """
    Prepara características para el modelo
    """
    # Agrupar por fecha
    datos_diarios = datos_por_categoria.groupby('fecha')['cantidad'].sum().reset_index()
    datos_diarios = datos_diarios.set_index('fecha').sort_index()
    
    # Características de series temporales
    datos_diarios['ma_7d'] = datos_diarios['cantidad'].rolling(window=7, min_periods=1).mean()
    datos_diarios['ma_14d'] = datos_diarios['cantidad'].rolling(window=14, min_periods=1).mean()
    datos_diarios['ma_30d'] = datos_diarios['cantidad'].rolling(window=30, min_periods=1).mean()
    
    # Características de rezago
    datos_diarios['lag_3d'] = datos_diarios['cantidad'].shift(3).fillna(0)
    datos_diarios['lag_7d'] = datos_diarios['cantidad'].shift(7).fillna(0)
    
    # Tendencia y volatilidad
    datos_diarios['volatilidad_7d'] = datos_diarios['cantidad'].rolling(window=7, min_periods=1).std().fillna(0)
    datos_diarios['tendencia'] = np.arange(len(datos_diarios))
    datos_diarios['tendencia_norm'] = (datos_diarios['tendencia'] - datos_diarios['tendencia'].min()) / \
                                     (datos_diarios['tendencia'].max() - datos_diarios['tendencia'].min() or 1)
    
    # Desviación de media móvil
    datos_diarios['desvio_ma7'] = (datos_diarios['cantidad'] - datos_diarios['ma_7d']).abs() / \
                                 (datos_diarios['ma_7d'].where(datos_diarios['ma_7d'] > 0, 1))
    
    # Detección de outliers
    umbral_outlier = 2.0
    datos_diarios['es_outlier'] = (datos_diarios['desvio_ma7'] > umbral_outlier).astype(int)
    
    # Características temporales
    datos_diarios['año'] = datos_diarios.index.year
    datos_diarios['mes'] = datos_diarios.index.month
    datos_diarios['dia_mes'] = datos_diarios.index.day
    datos_diarios['dia_semana'] = datos_diarios.index.dayofweek
    datos_diarios['dia_año'] = datos_diarios.index.dayofyear
    
    # Características cíclicas
    datos_diarios['mes_sen'] = np.sin(2 * np.pi * datos_diarios['mes'] / 12)
    datos_diarios['mes_cos'] = np.cos(2 * np.pi * datos_diarios['mes'] / 12)
    
    return datos_diarios

def entrenar_modelos(datos_procesados):
    """
    Entrena modelos por categoría
    """
    modelos_por_categoria = {}
    resultados = []
    
    # Obtener categorías únicas
    categorias = datos_procesados['categoria_nombre'].unique()
    
    for categoria in categorias:
        print(f"\nProcesando: {categoria}")
        
        # Filtrar datos de la categoría
        datos_categoria = datos_procesados[datos_procesados['categoria_nombre'] == categoria].copy()
        
        # Preparar características
        datos_diarios = preparar_caracteristicas(datos_categoria)
        
        # Definir features y target
        X = datos_diarios[[
            'ma_7d', 'ma_14d', 'ma_30d', 'lag_3d', 'lag_7d',
            'volatilidad_7d', 'tendencia', 'tendencia_norm', 'desvio_ma7',
            'es_outlier', 'mes', 'dia_mes', 'dia_semana', 'dia_año',
            'mes_sen', 'mes_cos'
        ]]
        y = datos_diarios['cantidad']
        
        # Entrenar modelo XGBoost
        modelo = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        
        modelo.fit(X, y)
        
        # Calcular métricas
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.model_selection import cross_val_score
        
        y_pred = modelo.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
        
        # Importancia de características
        importancias = modelo.feature_importances_
        features = X.columns
        
        top_features = pd.DataFrame({
            'caracteristica': features,
            'importancia': importancias
        }).sort_values(by='importancia', ascending=False).head(5).to_dict('records')
        
        # Guardar modelo
        categoria_slug = categoria.replace(' ', '_').replace('(', '').replace(')', '').lower()
        modelo_path = os.path.join(MODELS_DIR, f"{categoria_slug}_model.pkl")
        with open(modelo_path, 'wb') as f:
            pickle.dump(modelo, f)
        
        # Guardar información del modelo
        modelos_por_categoria[categoria] = {
            'modelo': modelo,
            'caracteristicas': list(X.columns),
            'metricas': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            },
            'top_features': top_features,
            'productos': datos_categoria['producto_nombre'].unique().tolist()
        }
        
        # Guardar resultados
        resultados.append({
            'categoria': categoria,
            'metricas': {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
                'cv_r2': f"{cv_scores.mean():.2f} (±{cv_scores.std():.2f})"
            },
            'top_features': top_features
        })
        
        print(f"  Métricas: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
        print(f"  CV R² promedio: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
    # Guardar modelos_por_categoria
    with open(os.path.join(MODELS_DIR, "modelos_por_categoria.pkl"), 'wb') as f:
        pickle.dump(modelos_por_categoria, f)
    
    return resultados

def cargar_modelos_entrenados():
    """
    Carga modelos previamente entrenados
    """
    modelo_path = os.path.join(MODELS_DIR, "modelos_por_categoria.pkl")
    if os.path.exists(modelo_path):
        with open(modelo_path, 'rb') as f:
            return pickle.load(f)
    return {}

def predecir_demanda(categoria, num_meses, incluir_productos, modelos, productos_data, historico_ventas):
    """
    Predice la demanda para una categoría
    """
    if categoria not in modelos:
        raise ValueError(f"No hay modelo disponible para: {categoria}")
    
    # Obtener últimos datos y modelo
    info_modelo = modelos[categoria]
    modelo = info_modelo['modelo']
    
    # Determinar última fecha disponible
    ultima_fecha = datetime.now() - timedelta(days=7)
    
    # Generar fechas futuras para predicción
    fechas_futuras = []
    for i in range(num_meses):
        mes = (ultima_fecha.month + i) % 12 + 1
        año = ultima_fecha.year + ((ultima_fecha.month + i) // 12)
        mes_nombre = calendar.month_name[mes]
        
        fechas_futuras.append({
            'año': año,
            'mes': mes,
            'mes_nombre': mes_nombre
        })
    
    # Datos de productos de esta categoría
    productos_df = pd.DataFrame(productos_data)
    productos_categoria = []
    
    # Intentar encontrar productos por categoría
    try:
        productos_categoria = productos_df[
            productos_df['categoria_id'] == categoria
        ].to_dict('records')
    except:
        # Usar los productos registrados durante el entrenamiento
        nombres_productos = info_modelo.get('productos', [])
        productos_categoria = [{'nombre': nombre} for nombre in nombres_productos[:7]]
    
    # Si aún no hay productos, crear algunos genéricos
    if not productos_categoria:
        productos_categoria = [
            {'nombre': f'Producto {i+1} - {categoria}', 'precio': 100} 
            for i in range(5)
        ]
    
    # Realizar predicciones mensuales
    predicciones = []
    
    for fecha in fechas_futuras:
        # Crear características para predicción
        X_pred = pd.DataFrame({
            'ma_7d': [70],  # Valores de ejemplo
            'ma_14d': [65],
            'ma_30d': [68],
            'lag_3d': [72],
            'lag_7d': [69],
            'volatilidad_7d': [10],
            'tendencia': [100],
            'tendencia_norm': [0.8],
            'desvio_ma7': [0.2],
            'es_outlier': [0],
            'mes': [fecha['mes']],
            'dia_mes': [15],  # Día medio del mes
            'dia_semana': [2],  # Miércoles
            'dia_año': [fecha['mes'] * 30],
            'mes_sen': [np.sin(2 * np.pi * fecha['mes'] / 12)],
            'mes_cos': [np.cos(2 * np.pi * fecha['mes'] / 12)]
        })
        
        # Predecir demanda
        demanda_predicha = max(0, round(modelo.predict(X_pred)[0]))
        rango_inf = max(0, round(demanda_predicha * 0.9))
        rango_sup = round(demanda_predicha * 1.1)
        
        # Crear predicción mensual
        prediccion_mensual = {
            'año': fecha['año'],
            'mes': fecha['mes'],
            'mes_nombre': fecha['mes_nombre'],
            'demanda_predicha': demanda_predicha,
            'rango_inferior': rango_inf,
            'rango_superior': rango_sup
        }
        
        # Añadir distribución por productos
        if incluir_productos:
            productos_mes = []
            
            # Proporción de productos (simplificado)
            total_productos = len(productos_categoria)
            proporcion_base = 1 / total_productos if total_productos > 0 else 0
            
            # Aplicar distribución con un poco de variación
            for i, producto in enumerate(productos_categoria[:7]):  # Top 7 productos
                proporcion = proporcion_base * (0.8 + 0.4 * (total_productos - i) / total_productos)
                demanda_producto = max(0, round(demanda_predicha * proporcion))
                
                precio_unitario = producto.get('precio', 100)
                if not precio_unitario or precio_unitario <= 0:
                    precio_unitario = 100  # Valor por defecto
                
                valor_estimado = demanda_producto * precio_unitario
                
                productos_mes.append({
                    'producto': producto.get('nombre', f'Producto {i+1}'),
                    'demanda_predicha': demanda_producto,
                    'valor_estimado': round(valor_estimado, 2)
                })
            
            prediccion_mensual['productos'] = productos_mes
        
        predicciones.append(prediccion_mensual)
    
    # Calcular métricas
    demandas = [p['demanda_predicha'] for p in predicciones]
    
    metricas = {
        'demanda_promedio': round(sum(demandas) / len(demandas), 2),
        'demanda_pico': max(demandas),
        'demanda_total': sum(demandas)
    }
    
    return {
        'predicciones': predicciones,
        'metricas': metricas
    }

def obtener_top_productos(modelos, productos_data, historico_ventas, categoria=None, limit=10):
    """
    Obtiene productos con mayor demanda proyectada
    """
    # Lista para resultados
    resultados = []
    
    # Categorías a evaluar
    categorias_evaluar = [categoria] if categoria else modelos.keys()
    
    for cat in categorias_evaluar:
        if cat not in modelos:
            continue
        
        # Predecir demanda para la categoría
        try:
            prediccion = predecir_demanda(
                categoria=cat,
                num_meses=1,  # Un solo mes
                incluir_productos=True,
                modelos=modelos,
                productos_data=productos_data,
                historico_ventas=historico_ventas
            )
            
            # Extraer productos del primer mes
            if prediccion and len(prediccion['predicciones']) > 0:
                primer_mes = prediccion['predicciones'][0]
                if 'productos' in primer_mes:
                    for producto in primer_mes['productos']:
                        resultados.append({
                            'categoria': cat,
                            'producto': producto['producto'],
                            'demanda_predicha': producto['demanda_predicha'],
                            'valor_estimado': producto['valor_estimado']
                        })
        except:
            continue
    
    # Ordenar por demanda
    resultados.sort(key=lambda x: x['demanda_predicha'], reverse=True)
    
    # Limitar resultados
    return resultados[:limit]