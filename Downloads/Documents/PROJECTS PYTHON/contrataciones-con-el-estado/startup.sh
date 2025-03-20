#!/bin/bash
set -e

echo "======================================================"
echo "Iniciando aplicación ContratacionesPeru en Azure..."
echo "======================================================"

# Listar el directorio actual para diagnóstico
echo "Contenido del directorio de la aplicación:"
ls -la
echo "======================================================"

# Asegurar que existan los directorios necesarios
echo "Creando directorios de datos y caché..."
mkdir -p data/reglamentos
mkdir -p cache
chmod -R 755 data
chmod -R 755 cache

# Verificar si el directorio de caché está configurado en variables de entorno
if [ -n "$CACHE_DIR" ]; then
    echo "Usando directorio de caché configurado: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    chmod -R 755 "$CACHE_DIR"
fi

# Imprimir información del entorno
echo "======================================================"
echo "Información del entorno:"
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo "Directorio actual: $(pwd)"

# Verificar instalación de dependencias
echo "======================================================"
echo "Verificando dependencias instaladas..."
if ! pip list | grep -q fastapi; then
    echo "ADVERTENCIA: FastAPI no parece estar instalado. Instalando..."
    pip install fastapi uvicorn gunicorn
fi

# Verificar configuración de Azure OpenAI
if [ -n "$AZURE_OPENAI_API_KEY" ] && [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "Configuración de Azure OpenAI detectada"
else
    echo "ADVERTENCIA: Configuración de Azure OpenAI incompleta o no encontrada"
fi

# Verificar variables de entorno disponibles (sin mostrar valores sensibles)
echo "======================================================"
echo "Variables de entorno disponibles (nombres):"
env | cut -d= -f1
echo "======================================================"

# Comprobar si existe el archivo simple.py
if [ ! -f simple.py ]; then
    echo "ADVERTENCIA: El archivo simple.py no existe en el directorio actual"
    echo "Creando un archivo simple.py minimalista para funcionamiento de emergencia..."
    cat > simple.py << 'EOL'
"""Aplicación minimalista de emergencia."""
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Aplicación de emergencia funcionando. La versión completa no pudo iniciarse."}
EOL
    echo "Archivo simple.py de emergencia creado con éxito"
fi

# Comprobar si debemos usar la aplicación simple
if [ "$USE_SIMPLE_APP" = "true" ]; then
    echo "======================================================"
    echo "Iniciando la aplicación en modo simple (sin dependencias complejas)..."
    echo "Esto se ha configurado mediante la variable USE_SIMPLE_APP=true"
    echo "======================================================"
    exec gunicorn simple:app -b 0.0.0.0:$PORT --timeout 180 --workers 2
else
    echo "======================================================"
    echo "Intentando iniciar la aplicación completa..."
    
    # Intentar importar módulos críticos para verificar instalación
    echo "Verificando si se pueden importar los módulos críticos..."
    python -c "
try:
    import fastapi
    print('✓ FastAPI importado correctamente')
    import app.backend.api.main
    print('✓ Módulo principal importado correctamente')
except ImportError as e:
    print(f'✗ Error al importar módulos: {str(e)}')
    print('Activando modo simple automáticamente...')
    exit(1)
"
    
    # Si falló la importación, activar modo simple automáticamente
    if [ $? -ne 0 ]; then
        echo "======================================================"
        echo "La importación de módulos falló, cambiando a modo simple..."
        echo "======================================================"
        exec gunicorn simple:app -b 0.0.0.0:$PORT --timeout 180 --workers 2
    else
        # Si todo está bien, usar la aplicación completa
        echo "======================================================"
        echo "Iniciando la aplicación completa con Gunicorn..."
        echo "======================================================"
        exec gunicorn application:app -c gunicorn.conf.py 