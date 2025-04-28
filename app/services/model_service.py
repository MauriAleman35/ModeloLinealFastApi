import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import base64
import io
from datetime import datetime
import logging

from app.services.data_service import cargar_datos, procesar_datos

logger = logging.getLogger("model_service")

# Directorio para guardar modelos
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Directorio para gráficos temporales
CHART_DIR = "app/static/charts"
os.makedirs(CHART_DIR, exist_ok=True)

def analizar_ventas_categoria(df_unificado):
    """Analiza ventas por categoría"""
    try:
        if df_unificado is None:
            return None
        
        # Determinar columna de categoría
        if 'categoria_nombre' in df_unificado.columns:
            cat_column = 'categoria_nombre'
        else:
            cat_column = 'categoria'
            
        # Agrupar por fecha y categoría
        df_unificado['año_mes'] = df_unificado['fecha'].dt.to_period('M')
        ventas_mes_categoria = df_unificado.groupby([cat_column, 'año_mes']).agg(
            cantidad=('cantidad', 'sum'),
            fecha=('fecha', lambda x: x.iloc[0].replace(day=1))  # Primer día del mes
        ).reset_index()
        
        # Obtener categorías con más ventas
        top_categorias = df_unificado.groupby(cat_column)['cantidad'].sum().sort_values(ascending=False).index.tolist()
        
        logger.info(f"Análisis completo: {len(top_categorias)} categorías identificadas")
        return {
            'ventas_mes_categoria': ventas_mes_categoria,
            'df_unificado': df_unificado,
            'top_categorias': top_categorias
        }
    
    except Exception as e:
        logger.error(f"Error analizando ventas por categoría: {str(e)}")
        return None

def preparar_datos_modelo_lineal(datos_analizados, categoria):
    """Prepara datos para el modelo lineal de una categoría"""
    try:
        if datos_analizados is None:
            return None
            
        ventas_mes_categoria = datos_analizados['ventas_mes_categoria']
        
        # Filtrar por categoría 
        if 'categoria_nombre' in ventas_mes_categoria.columns:
            cat_column = 'categoria_nombre'
        else:
            cat_column = 'categoria'
            
        ventas_categoria = ventas_mes_categoria[ventas_mes_categoria[cat_column] == categoria].copy()
        
        if len(ventas_categoria) == 0:
            logger.error(f"No hay datos para la categoría {categoria}")
            return None
            
        # Ordenar por fecha
        ventas_categoria = ventas_categoria.sort_values('fecha')
        
        # Calcular tendencia (meses desde el primer registro)
        primer_fecha = ventas_categoria['fecha'].min()
        ventas_categoria['tendencia'] = ventas_categoria['fecha'].apply(
            lambda x: (x.year - primer_fecha.year) * 12 + x.month - primer_fecha.month
        )
        
        # Características estacionales
        ventas_categoria['mes'] = ventas_categoria['fecha'].dt.month
        ventas_categoria['mes_sin'] = np.sin(2 * np.pi * ventas_categoria['mes']/12)
        ventas_categoria['mes_cos'] = np.cos(2 * np.pi * ventas_categoria['mes']/12)
        
        # Festividades (meses con eventos en Bolivia)
        festividades_mes = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 8: 1, 12: 1}
        ventas_categoria['es_mes_festivo'] = ventas_categoria['mes'].map(lambda m: festividades_mes.get(m, 0))
        
        # Variables dummy para meses específicos importantes
        for mes in [5, 8, 12]:  # Mayo (Día de la Madre), Agosto (Independencia), Diciembre (Navidad)
            ventas_categoria[f'mes_{mes}'] = (ventas_categoria['mes'] == mes).astype(int)
        
        # Características para usar en el modelo
        caracteristicas = ['tendencia', 'mes_sin', 'mes_cos', 'es_mes_festivo', 'mes_5', 'mes_8', 'mes_12']
        
        logger.info(f"Datos preparados para {categoria}: {len(ventas_categoria)} registros")
        return {
            'ventas_categoria': ventas_categoria,
            'caracteristicas': caracteristicas,
            'ultima_fecha': ventas_categoria['fecha'].max(),
            'primer_fecha': primer_fecha,
            'info': {
                'primer_fecha': primer_fecha,
                'n_registros': len(ventas_categoria)
            }
        }
    
    except Exception as e:
        logger.error(f"Error preparando datos para modelo: {str(e)}")
        return None

def entrenar_modelo_lineal(datos_preparados, categoria):
    """Entrena modelo lineal para una categoría"""
    try:
        if datos_preparados is None:
            logger.error("No hay datos preparados para entrenar modelo")
            return None
            
        ventas_categoria = datos_preparados['ventas_categoria']
        caracteristicas = datos_preparados['caracteristicas']
        
        X = ventas_categoria[caracteristicas]
        y = ventas_categoria['cantidad']
        
        logger.info(f"=== ENTRENANDO MODELO LINEAL: {categoria} ===")
        logger.info(f"Registros para entrenamiento: {len(X)}")
        logger.info(f"Variables: {caracteristicas}")
        
        # Comparar modelos: regular vs ridge
        modelo_regular = LinearRegression()
        modelo_regular.fit(X, y)
        pred_regular = modelo_regular.predict(X)
        rmse_regular = np.sqrt(mean_squared_error(y, pred_regular))
        r2_regular = r2_score(y, pred_regular)
        
        modelo_ridge = Ridge(alpha=1.0)
        modelo_ridge.fit(X, y)
        pred_ridge = modelo_ridge.predict(X)
        rmse_ridge = np.sqrt(mean_squared_error(y, pred_ridge))
        r2_ridge = r2_score(y, pred_ridge)
        
        logger.info("=== COMPARACIÓN DE MODELOS ===")
        logger.info(f"Modelo Regular - RMSE: {rmse_regular:.2f}, R²: {r2_regular:.2f}")
        logger.info(f"Modelo Ridge - RMSE: {rmse_ridge:.2f}, R²: {r2_ridge:.2f}")
        
        # Seleccionar mejor modelo
        if r2_ridge > r2_regular:
            modelo = modelo_ridge
            rmse = rmse_ridge
            r2 = r2_ridge
            logger.info("✅ Se utilizará Ridge Regression (más estable)")
        else:
            modelo = modelo_regular
            rmse = rmse_regular
            r2 = r2_regular
            logger.info("✅ Se utilizará Regresión Lineal estándar")
        
        # R² ajustado
        n = len(X)
        p = len(caracteristicas)
        if n > p + 1:
            r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            logger.info(f"R² Ajustado: {r2_adj:.2f}")
        else:
            logger.info(f"⚠️ R² Ajustado no calculable (n ≤ p + 1)")
            r2_adj = None
        
        # Mostrar coeficientes
        logger.info("\nCoeficientes del modelo:")
        logger.info(f"Intercepto: {modelo.intercept_:.4f}")
        coef_df = pd.DataFrame({'caracteristica': caracteristicas, 'coeficiente': modelo.coef_})
        
        # Guardar modelo
        model_path = os.path.join(MODEL_DIR, f"lineal_{categoria.replace(' ', '_')}.joblib")
        joblib.dump(modelo, model_path)
        logger.info(f"✅ Modelo guardado como: {model_path}")
        
        return {
            'modelo': modelo,
            'metricas': {
                'rmse': rmse,
                'r2': r2,
                'r2_adj': r2_adj
            },
            'coeficientes': coef_df,
            'modelo_path': model_path
        }
    except Exception as e:
        logger.error(f"Error entrenando modelo: {str(e)}")
        return None

def predecir_ventas(categoria, meses_futuros=3):
    """Predice ventas para los próximos meses, incluyendo desglose por productos"""
    try:
        # Cargar datos para preparación
        ventas, detalles, productos, categorias = cargar_datos()
        df_unificado = procesar_datos(ventas, detalles, productos, categorias)
        
        if df_unificado is None:
            logger.error("No se pudieron cargar datos para predicción")
            return None
        
        # Analizar ventas por categoría
        datos_analizados = analizar_ventas_categoria(df_unificado)
        
        if categoria not in datos_analizados['top_categorias']:
            logger.error(f"Categoría {categoria} no encontrada")
            return None
            
        # Preparar datos para modelo lineal
        datos_preparados = preparar_datos_modelo_lineal(datos_analizados, categoria)
        
        if datos_preparados is None:
            logger.error(f"No se pudieron preparar datos para {categoria}")
            return None
            
        # Cargar modelo entrenado
        model_path = os.path.join(MODEL_DIR, f"lineal_{categoria.replace(' ', '_')}.joblib")
        if not os.path.exists(model_path):
            logger.info(f"Modelo no encontrado, entrenando uno nuevo para {categoria}")
            resultado_modelo = entrenar_modelo_lineal(datos_preparados, categoria)
            if resultado_modelo is None:
                logger.error(f"No se pudo entrenar modelo para {categoria}")
                return None
            modelo = resultado_modelo['modelo']
            rmse = resultado_modelo['metricas']['rmse']
        else:
            logger.info(f"Cargando modelo existente para {categoria}")
            modelo = joblib.load(model_path)
            # Estimar RMSE del modelo cargado
            X = datos_preparados['ventas_categoria'][datos_preparados['caracteristicas']]
            y = datos_preparados['ventas_categoria']['cantidad']
            rmse = np.sqrt(mean_squared_error(y, modelo.predict(X)))
        
        ultima_fecha = datos_preparados['ultima_fecha']
        caracteristicas = datos_preparados['caracteristicas']
        ventas_categoria = datos_preparados['ventas_categoria']
        info = datos_preparados['info']
        
        logger.info(f"\n=== PREDICCIÓN DE VENTAS: {categoria} ===")
        logger.info(f"Último dato disponible: {ultima_fecha.strftime('%Y-%m-%d')}")
        logger.info(f"Predicción para próximos {meses_futuros} meses")
        
        # Generar fechas futuras (mes a mes)
        fechas_prediccion = []
        fecha_actual = ultima_fecha
        for i in range(1, meses_futuros+1):
            # Avanzar al siguiente mes
            año = fecha_actual.year + ((fecha_actual.month + i - 1) // 12)
            mes = (fecha_actual.month + i - 1) % 12 + 1
            nueva_fecha = pd.Timestamp(año, mes, 1)
            fechas_prediccion.append(nueva_fecha)
        
        # Crear DataFrame para predicción
        pred_df = pd.DataFrame({
            'fecha': fechas_prediccion,
            'mes': [f.month for f in fechas_prediccion],
            'año': [f.year for f in fechas_prediccion],
        })
        
        # Generar las mismas características usadas en entrenamiento
        pred_df['mes_sin'] = np.sin(2 * np.pi * pred_df['mes']/12)
        pred_df['mes_cos'] = np.cos(2 * np.pi * pred_df['mes']/12)
        
        # Calcular tendencia (continuación de los datos anteriores)
        primer_fecha = info['primer_fecha']
        pred_df['tendencia'] = pred_df['fecha'].apply(
            lambda x: (x.year - primer_fecha.year) * 12 + x.month - primer_fecha.month
        )
        
        # Festividades para predicción
        festividades_mes = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 8: 1, 12: 1}
        pred_df['es_mes_festivo'] = pred_df['mes'].map(lambda m: festividades_mes.get(m, 0))
        
        # Variables dummy para meses específicos
        for mes in [5, 8, 12]:
            pred_df[f'mes_{mes}'] = (pred_df['mes'] == mes).astype(int)
        
        # Asegurar que tenemos las mismas columnas que usamos en entrenamiento
        X_pred = pred_df[caracteristicas]
        
        # Realizar predicción
        pred_df['demanda_predicha'] = modelo.predict(X_pred)
        
        # Ajustes: valores no negativos y enteros
        pred_df['demanda_predicha'] = np.maximum(0, pred_df['demanda_predicha'])
        pred_df['demanda_predicha'] = np.round(pred_df['demanda_predicha']).astype(int)
        
        # Calcular rangos de confianza
        pred_df['rango_inferior'] = np.maximum(0, pred_df['demanda_predicha'] - rmse)
        pred_df['rango_superior'] = pred_df['demanda_predicha'] + rmse
        
        pred_df['rango_inferior'] = np.round(pred_df['rango_inferior']).astype(int)
        pred_df['rango_superior'] = np.round(pred_df['rango_superior']).astype(int)
        
        # Nombres de meses
        meses_nombre = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        pred_df['mes_nombre'] = pred_df['mes'].map(meses_nombre)
        
        # Predicción de productos (top 5)
        prediccion_productos = []
        
        # Determinar columna de producto
        if 'producto_nombre' in df_unificado.columns:
            prod_column = 'producto_nombre'
        else:
            prod_column = 'producto'
        
        # Determinar columna de categoría
        if 'categoria_nombre' in df_unificado.columns:
            cat_column = 'categoria_nombre'
        else:
            cat_column = 'categoria'
        
        # Filtrar por la categoría seleccionada
        df_cat = df_unificado[df_unificado[cat_column] == categoria].copy()
        
        if len(df_cat) > 0:
            # Encontrar los 5 productos más vendidos de esta categoría
            top_productos = df_cat.groupby(prod_column)['cantidad'].sum().sort_values(ascending=False).head(5)
            total_ventas = top_productos.sum()
            
            # Calcular la proporción de cada producto en la categoría
            proporciones = top_productos / total_ventas
            
            # Para cada mes, crear predicción por producto
            for i, row in pred_df.iterrows():
                total_predicho = row['demanda_predicha']
                
                productos_mes = []
                for producto, proporcion in proporciones.items():
                    unidades = int(round(proporcion * total_predicho))
                    productos_mes.append({
                        'producto': producto,
                        'proporcion': float(proporcion),
                        'unidades': unidades
                    })
                
                prediccion_productos.append({
                    'mes': int(row['mes']),
                    'año': int(row['año']),
                    'productos': productos_mes
                })
        
        # Generar gráfico
        chart_path, chart_base64 = generar_grafico_prediccion(ventas_categoria, pred_df, categoria)
        
        # Resultados
        resultado = {
            'categoria': categoria,
            'ultima_fecha': ultima_fecha,
            'predicciones': pred_df.to_dict('records'),
            'prediccion_productos': prediccion_productos,
            'grafico_path': chart_path,
            'grafico_base64': chart_base64,
            'metricas': {
                'rmse': float(rmse)
            }
        }
        
        return resultado
    
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generar_grafico_prediccion(ventas_categoria, pred_df, categoria):
    """Genera gráfico para la predicción"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Configuración de estilo
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Datos históricos
        plt.plot(ventas_categoria['fecha'], ventas_categoria['cantidad'], 
                'b-', marker='o', linewidth=2, label='Ventas históricas')
        
        # Predicciones
        plt.plot(pred_df['fecha'], pred_df['demanda_predicha'], 
                'r-', marker='s', linewidth=2, label='Predicción')
        plt.fill_between(pred_df['fecha'], 
                        pred_df['rango_inferior'], 
                        pred_df['rango_superior'], 
                        color='r', alpha=0.2, label='Intervalo de confianza')
        
        # Marcadores para meses festivos importantes
        for fecha in pd.concat([pd.Series(ventas_categoria['fecha']), pd.Series(pred_df['fecha'])]):
            mes = fecha.month
            if mes in [2, 3]:  # Carnaval o Semana Santa
                plt.axvline(x=fecha, color='gray', linestyle='--', alpha=0.3)
            elif mes == 5:  # Día de la Madre
                plt.axvline(x=fecha, color='pink', linestyle='--', alpha=0.5)
                plt.text(fecha, plt.ylim()[1]*0.95, 'Día Madre', 
                       rotation=90, verticalalignment='top', fontsize=9, alpha=0.7)
            elif mes == 8:  # Independencia
                plt.axvline(x=fecha, color='green', linestyle='--', alpha=0.5)
                plt.text(fecha, plt.ylim()[1]*0.95, 'Indep.', 
                       rotation=90, verticalalignment='top', fontsize=9, alpha=0.7)
            elif mes == 12:  # Navidad
                plt.axvline(x=fecha, color='red', linestyle='--', alpha=0.5)
                plt.text(fecha, plt.ylim()[1]*0.95, 'Navidad', 
                       rotation=90, verticalalignment='top', fontsize=9, alpha=0.7)
        
        # Separador entre histórico y predicción
        ultima_fecha = ventas_categoria['fecha'].max()
        plt.axvline(x=ultima_fecha, color='black', linestyle='--')
        plt.text(ultima_fecha, plt.ylim()[1]*0.5, 'Hoy', 
               rotation=90, verticalalignment='top', fontsize=10)
        
        plt.title(f'Predicción de Ventas: {categoria}', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Cantidad (unidades/mes)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Guardar gráfico en archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediccion_{categoria.replace(" ", "_")}_{timestamp}.png'
        chart_path = os.path.join(CHART_DIR, filename)
        plt.savefig(chart_path)
        
        # Convertir a base64 para API
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        plt.close()
        
        return chart_path, chart_base64
    
    except Exception as e:
        logger.error(f"Error generando gráfico: {str(e)}")
        return None, None