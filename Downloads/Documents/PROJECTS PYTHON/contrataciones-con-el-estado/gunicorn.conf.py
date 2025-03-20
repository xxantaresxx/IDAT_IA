"""
Configuración de Gunicorn para Azure App Services
"""
import multiprocessing

# Configuración del servidor
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180  # Aumentado para permitir la carga inicial de documentos
keepalive = 75
errorlog = "-"  # Enviar a stderr
accesslog = "-"  # Enviar a stdout
loglevel = "info"

# Hooks
def on_starting(server):
    """Log server start"""
    server.log.info("Starting ContratacionesPeru application") 