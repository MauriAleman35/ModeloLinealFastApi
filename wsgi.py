# wsgi.py
from app import app

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Configuración del servidor
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))
    
    # Iniciar servidor
    uvicorn.run("wsgi:app", host=host, port=port)