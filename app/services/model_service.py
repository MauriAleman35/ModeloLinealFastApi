import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import base64
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from bson import ObjectId
from app.services.data_service import cargar_datos, procesar_datos
from app.database.mongodb import get_sync_client

# Configuración
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger("model_service")

# Asegurarnos que existen los directorios necesarios
os.makedirs('models', exist_ok=True)
os.makedirs('app/static/charts', exist_ok=True)

def analizar_ventas_categoria(df_unificado):
    """Analiza ventas agrupadas por categoría y mes"""
    try:
        if df_unificado is None:
            logger.error("DataFrame es None")
            return None
        
        # Verificar columnas necesarias
        columnas_requeridas = ['fecha', 'cantidad', 'subtotal']
        columnas_categoria = ['categoria_nombre', 'categoria', 'tipo']
        
        # Verificar columna fecha
        if 'fecha' not in df_unificado.columns:
            logger.error("Columna 'fecha' no encontrada")
            return None
        
        # Verificar columna cantidad
        if 'cantidad' not in df_unificado.columns:
            logger.warning("Columna 'cantidad' no encontrada. Usando valor por defecto 1")
            df_unificado['cantidad'] = 1
        
        # Verificar columna subtotal/precio
        if 'subtotal' not in df_unificado.columns:
            if 'precio_unitario' in df_unificado.columns:
                df_unificado['subtotal'] = df_unificado['precio_unitario'] * df_unificado['cantidad']
            else:
                logger.warning("Columna 'subtotal' no encontrada. Generando valores aleatorios")
                df_unificado['subtotal'] = np.random.randint(10, 100, size=len(df_unificado))
        
        # Comprobar columna de categoría
        col_categoria = None
        for col in columnas_categoria:
            if col in df_unificado.columns:
                col_categoria = col
                break
                
        if col_categoria is None:
            logger.error("No se encontró columna de categoría")
            return None
            
        # Extraer año y mes de la fecha
        df_unificado['anio'] = df_unificado['fecha'].dt.year
        df_unificado['mes'] = df_unificado['fecha'].dt.month
        
        # Agrupar por categoría, año y mes
        ventas_por_mes = df_unificado.groupby([col_categoria, 'anio', 'mes']).agg({
            'cantidad': 'sum',
            'subtotal': 'sum'
        }).reset_index()
        
        # Agrupar por producto dentro de cada categoría y mes
        productos_por_categoria = df_unificado.groupby([col_categoria, 'anio', 'mes', 'producto_id']).agg({
            'cantidad': 'sum'
        }).reset_index()
        
        # Calcular proporción de cada producto por categoría y mes
        for cat in ventas_por_mes[col_categoria].unique():
            # Filtrar por categoría
            cat_ventas = ventas_por_mes[ventas_por_mes[col_categoria] == cat]
            cat_productos = productos_por_categoria[productos_por_categoria[col_categoria] == cat]
            
            # Para cada período (año, mes)
            for anio in cat_ventas['anio'].unique():
                for mes in cat_ventas[cat_ventas['anio'] == anio]['mes'].unique():
                    # Ventas totales de esta categoría en este mes
                    total_ventas = cat_ventas[
                        (cat_ventas['anio'] == anio) & 
                        (cat_ventas['mes'] == mes)
                    ]['cantidad'].values[0]
                    
                    # Calcular proporción para cada producto
                    productos_mes = cat_productos[
                        (cat_productos['anio'] == anio) & 
                        (cat_productos['mes'] == mes)
                    ]
                    
                    # Asignar proporción
                    for idx, prod in productos_mes.iterrows():
                        proporcion = prod['cantidad'] / total_ventas
                        productos_por_categoria.at[idx, 'proporcion'] = proporcion
        
        logger.info(f"Análisis completo: {len(ventas_por_mes[col_categoria].unique())} categorías identificadas")
        
        return {
            'ventas_por_mes': ventas_por_mes,
            'productos_por_categoria': productos_por_categoria,
            'columna_categoria': col_categoria
        }
    except Exception as e:
        logger.error(f"Error en análisis de ventas: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def preparar_datos_modelo_lineal(datos_analizados, categoria_id):
    """Prepara datos para entrenar modelo lineal de una categoría"""
    try:
        if datos_analizados is None:
            logger.error("No hay datos analizados")
            return None
            
        col_categoria = datos_analizados['columna_categoria']
        ventas_por_mes = datos_analizados['ventas_por_mes']
        productos_por_categoria = datos_analizados['productos_por_categoria']
        
        # Resolver categoría basada en ID o nombre
        categoria_encontrada = None
        
        # Intentar primero como ID
        cat_matches = ventas_por_mes[ventas_por_mes[col_categoria].astype(str) == str(categoria_id)]
        if len(cat_matches) > 0:
            categoria_encontrada = str(categoria_id)
            logger.info(f"Categoría encontrada por ID: {categoria_id}")
        else:
            # Intentar obtener mappeo ID -> nombre desde MongoDB
            try:
                db = get_sync_client()
                cat_doc = None
                
                # Intentar buscar por ID
                try:
                    from bson import ObjectId
                    cat_doc = db.categorias.find_one({"_id": ObjectId(str(categoria_id))})
                except:
                    pass
                
                # Si no se encontró, buscar por nombre en diferentes campos
                if not cat_doc:
                    cat_doc = db.categorias.find_one({
                        "$or": [
                            {"titulo": categoria_id},
                            {"nombre": categoria_id},
                            {"name": categoria_id}
                        ]
                    })
                
                # Si encontramos la categoría
                if cat_doc:
                    # Buscar nombre de la categoría
                    nombre_cat = None
                    for field in ["titulo", "nombre", "name"]:
                        if field in cat_doc and cat_doc[field]:
                            nombre_cat = cat_doc[field]
                            break
                    
                    if nombre_cat:
                        # Buscar coincidencia en ventas
                        cat_matches = ventas_por_mes[ventas_por_mes[col_categoria] == nombre_cat]
                        if len(cat_matches) > 0:
                            categoria_encontrada = nombre_cat
                            logger.info(f"Categoría encontrada por nombre en DB: {nombre_cat}")
            except Exception as e:
                logger.warning(f"Error buscando categoría en MongoDB: {str(e)}")
            
            # Si todavía no encontramos, buscar coincidencia parcial
            if not categoria_encontrada:
                for cat in ventas_por_mes[col_categoria].unique():
                    if str(categoria_id).lower() in str(cat).lower():
                        categoria_encontrada = cat
                        logger.info(f"Categoría encontrada por coincidencia parcial: {cat}")
                        break
        
        if not categoria_encontrada:
            # Como último recurso, usar la primera categoría disponible
            if len(ventas_por_mes) > 0:
                categoria_encontrada = ventas_por_mes[col_categoria].iloc[0]
                logger.warning(f"Categoría no encontrada. Usando primera disponible: {categoria_encontrada}")
            else:
                logger.error(f"Categoría {categoria_id} no encontrada")
                return None
        
        # Filtrar datos por categoría encontrada
        cat_ventas = ventas_por_mes[ventas_por_mes[col_categoria] == categoria_encontrada]
        
        if len(cat_ventas) == 0:
            logger.error(f"No hay datos para categoría {categoria_encontrada}")
            return None
        
        # Preparar características (X) y target (y)
        cat_ventas = cat_ventas.sort_values(by=['anio', 'mes'])
        cat_ventas['periodo'] = cat_ventas['anio'] * 12 + cat_ventas['mes']
        cat_ventas['periodo_idx'] = range(len(cat_ventas))
        
        X = cat_ventas[['periodo_idx']].values
        y = cat_ventas['cantidad'].values
        
        # Productos asociados a esta categoría
        cat_productos = productos_por_categoria[
            productos_por_categoria[col_categoria] == categoria_encontrada
        ]
        
        return {
            'categoria': categoria_encontrada,
            'X': X,
            'y': y,
            'ventas': cat_ventas,
            'productos': cat_productos
        }
    except Exception as e:
        logger.error(f"Error preparando datos para modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def entrenar_modelo_lineal(datos_preparados, categoria_id):
    """Entrena un modelo de regresión lineal para predicción de ventas"""
    try:
        if datos_preparados is None:
            logger.error("No hay datos preparados para entrenar modelo")
            return None
            
        categoria = datos_preparados['categoria']
        X = datos_preparados['X']
        y = datos_preparados['y']
        
        # Entrenar modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Evaluar modelo
        y_pred = modelo.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Modelo entrenado para {categoria}: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        
        # Guardar modelo
        model_filename = f"lineal_{str(categoria_id).replace(' ', '_')}.joblib"
        model_path = os.path.join("models", model_filename)
        joblib.dump({
            'modelo': modelo,
            'metricas': {
                'mse': mse,
                'mae': mae,
                'r2': r2
            },
            'categoria': categoria,
            'ultima_fecha': datetime.now().strftime('%Y-%m-%d')
        }, model_path)
        
        logger.info(f"Modelo guardado en {model_path}")
        
        # Generar gráfico de evaluación
        plt.figure(figsize=(10, 6))
        plt.plot(datos_preparados['ventas']['periodo'], y, 'b-', label='Actual')
        plt.plot(datos_preparados['ventas']['periodo'], y_pred, 'r--', label='Predicción')
        plt.title(f'Evaluación Modelo: {categoria}')
        plt.xlabel('Período')
        plt.ylabel('Ventas')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Guardar gráfico
        chart_path = os.path.join("app", "static", "charts", f"eval_{str(categoria_id).replace(' ', '_')}.png")
        plt.savefig(chart_path)
        
        return {
            'categoria': categoria,
            'metricas': {
                'mse': mse,
                'mae': mae,
                'r2': r2
            },
            'model_path': model_path,
            'chart_path': chart_path
        }
    except Exception as e:
        logger.error(f"Error entrenando modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predecir_ventas(categoria_id, meses=3):
    """Predice ventas para una categoría en los próximos n meses"""
    try:
        # Importar dependencias necesarias
        import numpy as np
        from bson import ObjectId
        
        # Cargar datos para generar predicciones de productos
        ventas, detalles, productos, categorias_data = cargar_datos()
        if ventas is None or detalles is None:
            logger.error("No se pudieron cargar datos para predicción")
            return None
            
        df_unificado = procesar_datos(ventas, detalles, productos, categorias_data)
        if df_unificado is None:
            logger.error("Error procesando datos para predicción")
            return None
            
        datos_analizados = analizar_ventas_categoria(df_unificado)
        if datos_analizados is None:
            logger.error("Error analizando datos para predicción")
            return None
            
        datos_preparados = preparar_datos_modelo_lineal(datos_analizados, categoria_id)
        if datos_preparados is None:
            logger.error("Error preparando datos para predicción")
            return None
            
        categoria = datos_preparados['categoria']
        
        # Asegurarse que categoria_id sea string para el nombre de archivo
        categoria_id_str = str(categoria_id).replace(' ', '_')
        
        # Buscar modelo entrenado
        model_filename = f"lineal_{categoria_id_str}.joblib"
        model_path = os.path.join("models", model_filename)
        
        if not os.path.exists(model_path):
            logger.warning(f"No existe modelo para {categoria}. Entrenando nuevo modelo...")
            resultado_entrenamiento = entrenar_modelo_lineal(datos_preparados, categoria_id)
            if resultado_entrenamiento is None:
                logger.error(f"Error entrenando nuevo modelo para {categoria}")
                return None
        
        # Cargar modelo
        try:
            modelo_data = joblib.load(model_path)
            modelo = modelo_data['modelo']
            metricas = modelo_data.get('metricas', {})
            ultima_fecha = modelo_data.get('ultima_fecha', datetime.now().strftime('%Y-%m-%d'))
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return None
            
        # Preparar datos para predicción
        X = datos_preparados['X']
        y = datos_preparados['y']
        ventas_historicas = datos_preparados['ventas']
        
        # Obtener último período
        ultimo_periodo_idx = X[-1][0]
        
        # Preparar períodos futuros
        periodos_futuros_idx = np.array([[i] for i in range(ultimo_periodo_idx + 1, ultimo_periodo_idx + meses + 1)])
        
        # Realizar predicción
        prediccion_futura = modelo.predict(periodos_futuros_idx)
        
        # Establecer límites para que no haya predicciones negativas
        prediccion_futura = np.maximum(prediccion_futura, 0)
        
        # Calcular rango de confianza (+-20% por defecto)
        margen_error = metricas.get('mae', prediccion_futura.mean() * 0.2)
        rango_inferior = np.maximum(prediccion_futura - margen_error, 0)
        rango_superior = prediccion_futura + margen_error
        
        # Preparar fechas de predicción
        ultimo_periodo = ventas_historicas.iloc[-1]['periodo']
        ultimo_anio = ventas_historicas.iloc[-1]['anio']
        ultimo_mes = ventas_historicas.iloc[-1]['mes']
        
        # Generar lista de períodos futuros
        periodos_futuros = []
        for i in range(1, meses + 1):
            mes_futuro = ultimo_mes + i
            anio_futuro = ultimo_anio
            
            # Ajustar cuando pasamos de diciembre
            if mes_futuro > 12:
                mes_futuro = mes_futuro - 12
                anio_futuro = anio_futuro + 1
                
            # Nombre del mes
            nombre_mes = {
                1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
            }.get(mes_futuro, '')
            
            periodos_futuros.append({
                'anio': anio_futuro,
                'mes': mes_futuro,
                'periodo': anio_futuro * 12 + mes_futuro,
                'nombre_mes': nombre_mes
            })
        
        # Preparar datos de predicción
        predicciones = []
        for i in range(meses):
            predicciones.append({
                'año': int(periodos_futuros[i]['anio']),
                'mes': int(periodos_futuros[i]['mes']),
                'mes_nombre': periodos_futuros[i]['nombre_mes'],
                'demanda_predicha': int(round(prediccion_futura[i])),
                'rango_inferior': int(round(rango_inferior[i])),
                'rango_superior': int(round(rango_superior[i]))
            })
        
        # Generar predicciones por producto
        productos_por_mes = []
        productos_por_categoria = datos_preparados['productos']
        
        # Obtener proporciones promedio por producto
        productos_unicos = productos_por_categoria['producto_id'].unique()
        proporciones_promedio = {}
        
        for prod_id in productos_unicos:
            # Convertir a string si es un ObjectId
            prod_id_str = str(prod_id)
            
            prod_data = productos_por_categoria[productos_por_categoria['producto_id'] == prod_id]
            if 'proporcion' in prod_data.columns:
                prop_media = prod_data['proporcion'].mean()
                proporciones_promedio[prod_id_str] = float(prop_media)
            else:
                # Si no hay proporciones, distribuir uniformemente
                proporciones_promedio[prod_id_str] = float(1.0 / len(productos_unicos))
        
        # Asegurar que las proporciones suman 1
        sum_proporciones = sum(proporciones_promedio.values())
        if sum_proporciones > 0:
            for prod_id in proporciones_promedio:
                proporciones_promedio[prod_id] = float(proporciones_promedio[prod_id] / sum_proporciones)
        
        # Generar predicción por producto para cada mes
        for i in range(meses):
            demanda_mes = int(round(prediccion_futura[i]))
            productos_mes = []
            
            # Distribuir la demanda entre productos según proporciones
            for prod_id, proporcion in proporciones_promedio.items():
                unidades_producto = int(round(demanda_mes * proporcion))
                if unidades_producto > 0:  # Solo incluir productos con demanda
                    # Intentar obtener nombre del producto
                    nombre_producto = f"Producto {prod_id[-6:]}"
                    try:
                        if isinstance(productos, pd.DataFrame) and 'producto_id' in productos.columns:
                            prod_info = productos[productos['producto_id'] == prod_id]
                            if len(prod_info) > 0:
                                for col in ['nombre', 'name', 'titulo', 'descripcion']:
                                    if col in prod_info.columns and not pd.isna(prod_info[col].iloc[0]):
                                        nombre_producto = str(prod_info[col].iloc[0])
                                        break
                    except:
                        pass
                    
                    productos_mes.append({
                        'producto': nombre_producto,
                        'producto_id': str(prod_id),  # Asegurar que es string
                        'proporcion': float(round(proporcion * 100, 2)),
                        'unidades': int(unidades_producto)
                    })
            
            # Ordenar productos por unidades (mayor a menor)
            productos_mes.sort(key=lambda x: x['unidades'], reverse=True)
            
            productos_por_mes.append({
                'año': int(periodos_futuros[i]['anio']),
                'mes': int(periodos_futuros[i]['mes']),
                'mes_nombre': periodos_futuros[i]['nombre_mes'],
                'demanda_total': int(demanda_mes),
                'productos': productos_mes
            })
        
        # Generar gráfico
        plt.figure(figsize=(12, 7))
        
        # Datos históricos
        periodos_hist = ventas_historicas['periodo'].values
        ventas_hist = ventas_historicas['cantidad'].values
        
        plt.plot(periodos_hist, ventas_hist, 'b-', label='Ventas históricas')
        
        # Datos de predicción
        periodos_pred = [p['periodo'] for p in periodos_futuros]
        
        plt.plot(periodos_pred, prediccion_futura, 'r--', label='Predicción')
        plt.fill_between(periodos_pred, rango_inferior, rango_superior, 
                        color='r', alpha=0.2, label='Intervalo de confianza')
        
        # Añadir etiquetas
        plt.title(f'Predicción de Ventas: {categoria}')
        plt.xlabel('Período')
        plt.ylabel('Unidades')
        plt.grid(True)
        
        # Mejorar visualización de fechas en el eje X
        all_periods = list(periodos_hist) + periodos_pred
        all_labels = []
        
        for p in all_periods:
            anio = p // 12
            mes = p % 12
            if mes == 0:
                mes = 12
            label = f"{mes}/{anio}"
            all_labels.append(label)
        
        # Mostrar menos etiquetas para evitar superposición
        step = max(1, len(all_periods) // 10)
        plt.xticks(all_periods[::step], all_labels[::step], rotation=45)
        
        plt.legend()
        plt.tight_layout()
        
        # Guardar gráfico
        chart_filename = f"pred_{categoria_id_str}.png"
        chart_path = os.path.join("app", "static", "charts", chart_filename)
        plt.savefig(chart_path)
        
        # Convertir gráfico a base64 para API
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        grafico_base64 = base64.b64encode(img_data.getvalue()).decode()
        
        plt.close()
        
        # Asegurarse que la categoría sea string (no ObjectId)
        if isinstance(categoria, ObjectId):
            categoria = str(categoria)
            
        # Convertir métricas (posibles valores NumPy)
        metricas_convertidas = {}
        for key, val in metricas.items():
            if isinstance(val, (np.integer, np.int32, np.int64)):
                metricas_convertidas[key] = int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                metricas_convertidas[key] = float(val)
            else:
                metricas_convertidas[key] = val
            
        # Construir respuesta con tipos serializables
        resultado = {
            'categoria': str(categoria),  # Asegurar que es string
            'categoria_id': str(categoria_id),  # Añadir el ID original
            'ultima_fecha': datetime.strptime(ultima_fecha, '%Y-%m-%d'),
            'predicciones': predicciones,
            'prediccion_productos': productos_por_mes,
            'grafico_base64': grafico_base64,
            'metricas': metricas_convertidas
        }
        
        return resultado
        
    except Exception as e:
        logger.error(f"Error generando predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return None