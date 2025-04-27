# gunicorn_config.py
import os
import multiprocessing
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del bind
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5001')}"

# Configuración de workers
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"

# Configuración de logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"  # stdout
errorlog = "-"   # stderr

# Timeouts
timeout = 120
keepalive = 5

# Configuración adicional
reload = os.getenv("DEBUG", "False").lower() == "true"