import os
import pickle
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from threading import Thread, Lock

logger = logging.getLogger("cache_service")

# Ruta para archivos de caché
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Bloqueo para operaciones de caché
cache_lock = Lock()

class DataCache:
    """Servicio para cachear datos y reducir consultas a MongoDB"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCache, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.cache = {}
            self.last_update = {}
            self.update_interval = {
                'ventas': 3600,  # 1 hora
                'detalles': 3600,
                'productos': 3600 * 6,  # 6 horas
                'categorias': 3600 * 24,  # 24 horas
                'modelos': 3600 * 24 * 7,  # 7 días
            }
            self.initialized = True
            self._start_background_updates()
    
    def get(self, key, loader_func=None, force_refresh=False):
        """Obtiene datos de caché o los carga si es necesario"""
        with cache_lock:
            current_time = time.time()
            
            # Si no hay datos o necesitan actualizarse
            if (key not in self.cache or 
                force_refresh or 
                current_time - self.last_update.get(key, 0) > self.update_interval.get(key, 3600)):
                
                # Si se proporciona función de carga, usarla
                if loader_func:
                    try:
                        logger.info(f"Cargando datos frescos para '{key}'")
                        data = loader_func()
                        if data is not None:
                            self.cache[key] = data
                            self.last_update[key] = current_time
                            # Guardar en disco
                            self._save_to_disk(key, data)
                    except Exception as e:
                        logger.error(f"Error cargando datos para '{key}': {str(e)}")
                        # Intentar cargar versión guardada
                        data = self._load_from_disk(key)
                        if data is not None:
                            self.cache[key] = data
                            return data
                        return None
                else:
                    # Cargar desde disco si no hay función de carga
                    logger.info(f"Intentando cargar '{key}' desde disco")
                    data = self._load_from_disk(key)
                    if data is not None:
                        self.cache[key] = data
                        self.last_update[key] = current_time
            
            return self.cache.get(key)
    
    def _save_to_disk(self, key, data):
        """Guarda datos en disco"""
        try:
            file_path = os.path.join(CACHE_DIR, f"{key}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Datos guardados en disco: {file_path}")
        except Exception as e:
            logger.error(f"Error guardando caché en disco: {str(e)}")
    
    def _load_from_disk(self, key):
        """Carga datos desde disco"""
        try:
            file_path = os.path.join(CACHE_DIR, f"{key}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Datos cargados desde disco: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Error cargando caché desde disco: {str(e)}")
        return None
    
    def _start_background_updates(self):
        """Inicia hilo de actualizaciones periódicas"""
        Thread(target=self._background_updates, daemon=True).start()
    
    def _background_updates(self):
        """Ejecuta actualizaciones periódicas en segundo plano"""
        from app.services.data_service import cargar_datos
        
        while True:
            try:
                # Actualizar datos principales cada cierto tiempo
                with cache_lock:
                    current_time = time.time()
                    
                    # Verificar cada tipo de datos
                    for key in self.update_interval:
                        if (key not in self.last_update or 
                            current_time - self.last_update.get(key, 0) > self.update_interval[key]):
                            
                            if key in ['ventas', 'detalles', 'productos', 'categorias']:
                                logger.info(f"Actualizando caché de {key} en segundo plano")
                                ventas, detalles, productos, categorias = cargar_datos()
                                
                                if ventas is not None and 'ventas' == key:
                                    self.cache['ventas'] = ventas
                                    self.last_update['ventas'] = current_time
                                    self._save_to_disk('ventas', ventas)
                                
                                if detalles is not None and 'detalles' == key:
                                    self.cache['detalles'] = detalles
                                    self.last_update['detalles'] = current_time
                                    self._save_to_disk('detalles', detalles)
                                    
                                if productos is not None and 'productos' == key:
                                    self.cache['productos'] = productos
                                    self.last_update['productos'] = current_time
                                    self._save_to_disk('productos', productos)
                                    
                                if categorias is not None and 'categorias' == key:
                                    self.cache['categorias'] = categorias
                                    self.last_update['categorias'] = current_time
                                    self._save_to_disk('categorias', categorias)
            except Exception as e:
                logger.error(f"Error en actualizaciones de fondo: {str(e)}")
            
            # Esperar antes de la siguiente actualización
            time.sleep(60)  # Verificar cada minuto

# Instancia global
cache = DataCache()