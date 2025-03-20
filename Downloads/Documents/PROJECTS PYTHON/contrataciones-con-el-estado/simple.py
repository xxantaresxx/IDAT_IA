"""
Versión extremadamente simple de la aplicación para diagnóstico en Azure App Service.
No depende de ninguna biblioteca externa excepto FastAPI.
"""
import os
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simple_app")

# Registrar información del entorno al inicio
logger.info("=== Iniciando aplicación simple para diagnóstico ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Directorio actual: {os.getcwd()}")
logger.info(f"Contenido del directorio: {os.listdir('.')}")
logger.info(f"Variables de entorno: {list(os.environ.keys())}")

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    logger.info("FastAPI importado correctamente")
except ImportError as e:
    logger.error(f"Error importando FastAPI: {str(e)}")
    sys.exit(1)

# Crear una aplicación FastAPI extremadamente simple
app = FastAPI(title="ContratacionesPeru - Diagnóstico", docs_url="/")

@app.get("/health")
async def health():
    """Endpoint de verificación de salud básico"""
    return {"status": "ok"}

@app.get("/info", response_class=JSONResponse)
async def info():
    """Devuelve información del sistema"""
    try:
        dir_content = os.listdir(".")
    except Exception as e:
        dir_content = f"Error: {str(e)}"
        
    try:
        # Intentar obtener la lista de paquetes instalados
        import pkg_resources
        packages = sorted([f"{dist.project_name}=={dist.version}" 
                       for dist in pkg_resources.working_set])
    except Exception as e:
        packages = [f"Error: {str(e)}"]
    
    return {
        "app": "ContratacionesPeru - Ultra Simple",
        "version": "0.1.0",
        "status": "running",
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "directory_content": dir_content,
        "environment_variables": list(os.environ.keys()),
        "installed_packages": packages[:20]  # Limitar a 20 para evitar respuestas muy grandes
    }

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Interfaz HTML simple para diagnóstico"""
    try:
        # Intentar obtener la lista de paquetes instalados
        import pkg_resources
        packages = sorted([f"{dist.project_name}=={dist.version}" 
                       for dist in pkg_resources.working_set])
    except Exception as e:
        packages = [f"Error: {str(e)}"]
        
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ContratacionesPeru - Diagnóstico</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            pre {{ background: #f4f4f4; padding: 20px; border-radius: 5px; overflow: auto; max-height: 300px; }}
            h1 {{ color: #2196F3; }}
            h2 {{ color: #0D47A1; margin-top: 30px; }}
            .box {{ border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px; }}
            .success {{ color: #388e3c; }}
            .warning {{ color: #f57c00; }}
            .error {{ color: #d32f2f; }}
        </style>
    </head>
    <body>
        <h1>ContratacionesPeru - Modo Ultra Simple</h1>
        
        <div class="box">
            <h2>Estado de la aplicación</h2>
            <p class="success">La aplicación está funcionando en modo diagnóstico</p>
            <p>Python version: {sys.version}</p>
            <p>Directorio actual: {os.getcwd()}</p>
        </div>
        
        <div class="box">
            <h2>Paquetes instalados</h2>
            <pre>{chr(10).join(packages[:20])}{"..." if len(packages) > 20 else ""}</pre>
        </div>
        
        <div class="box">
            <h2>Endpoints disponibles</h2>
            <ul>
                <li><a href="/">/</a> - Documentación Swagger</li>
                <li><a href="/health">/health</a> - Comprobación de salud</li>
                <li><a href="/info">/info</a> - Información detallada (JSON)</li>
                <li><a href="/ui">/ui</a> - Esta interfaz</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_content

# Registrar información al momento de crear la aplicación
logger.info("Aplicación FastAPI creada correctamente")
logger.info("=== Inicialización completada ===")

# Al final del archivo, para depuración
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 