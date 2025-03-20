#!/bin/bash
set -e

echo "==============================================="
echo "Iniciando instalación de dependencias..."
echo "==============================================="

# Mostrar información del sistema
echo "Información del sistema:"
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo "==============================================="

# Asegurarse de que startup.sh tiene permisos de ejecución
echo "Asegurando permisos de ejecución para scripts..."
chmod +x startup.sh
echo "Permisos configurados."

# Instalación inicial con requirements-azure.txt (más compatible)
echo "Instalando dependencias fundamentales desde requirements-azure.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements-azure.txt --no-cache-dir

# Intentar instalar el resto de dependencias si las anteriores funcionaron
echo "Intentando instalar dependencias adicionales desde requirements.txt..."
python -m pip install -r requirements.txt --no-cache-dir || echo "Algunas dependencias opcionales no pudieron instalarse. La aplicación intentará funcionar sin ellas."

# Verificar instalación de dependencias críticas
echo "Verificando dependencias críticas..."
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"
python -c "import gunicorn; print(f'Gunicorn: {gunicorn.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"

# Intentar importar dependencias problemáticas o fallar silenciosamente
echo "Verificando dependencias opcionales..."
python -c "try: import langchain_core; print(f'Langchain Core: OK'); except ImportError as e: print(f'Langchain Core: No instalado - {str(e)}');"
python -c "try: import langgraph; print(f'Langgraph: OK'); except ImportError as e: print(f'Langgraph: No instalado - {str(e)}');"
python -c "try: import qdrant_client; print(f'Qdrant Client: OK'); except ImportError as e: print(f'Qdrant Client: No instalado - {str(e)}');"

echo "==============================================="
echo "Instalación de dependencias completada."
echo "==============================================="

# Imprimir mensaje de finalización
echo "La aplicación está lista para ejecutarse."
echo "El despliegue continuará normalmente." 