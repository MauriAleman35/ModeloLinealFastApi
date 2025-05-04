# gunicorn_config.py
import os
import multiprocessing
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del bind
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5001')}"

# CAMBIO AQUÍ: Reducir workers a 2 o 1 en Render
if os.getenv("ENVIRONMENT") == "production":
    workers = 1  # Usar solo 1 worker en producción
else:
    workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))

worker_class = "uvicorn.workers.UvicornWorker"

# Limitar memoria por worker
worker_tmp_dir = "/dev/shm"  # Usar memoria compartida
max_requests = 1000
max_requests_jitter = 50

# Configuración de logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"

# Timeouts
timeout = 120
keepalive = 5

# Configuración adicional
reload = os.getenv("DEBUG", "False").lower() == "true"