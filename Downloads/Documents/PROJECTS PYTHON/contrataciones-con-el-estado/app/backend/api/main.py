from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path de Python
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from app.backend.utils.document_processor import DocumentProcessor
from app.backend.utils.query_graph import process_query_with_graph

# Cargar variables de entorno
load_dotenv()

# Configurar directorios
REGLAMENTOS_DIR = ROOT_DIR / "data" / "reglamentos"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
CACHE_DIR = os.getenv("CACHE_DIR", str(ROOT_DIR / "cache"))

# Crear directorios necesarios
for directory in [REGLAMENTOS_DIR, STATIC_DIR, Path(CACHE_DIR)]:
    directory.mkdir(parents=True, exist_ok=True)

logger.info(f"\nDirectorios del sistema:")
logger.info(f"- Directorio raíz: {ROOT_DIR}")
logger.info(f"- Directorio de documentos: {REGLAMENTOS_DIR}")
logger.info(f"- Directorio estático: {STATIC_DIR}")
logger.info(f"- Directorio de caché: {CACHE_DIR}\n")

# Variables globales para los procesadores
document_processor = None

async def initialize_processors():
    """Inicializa los procesadores de forma asíncrona."""
    global document_processor
    
    logger.info("Inicializando procesadores...")
    start_time = time.time()
    
    # Verificar documentos PDF
    pdf_files = list(REGLAMENTOS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("\nADVERTENCIA: No se encontraron archivos PDF.")
        logger.warning(f"Por favor, coloque sus documentos PDF en: {REGLAMENTOS_DIR}")
        return
    
    logger.info(f"\nDocumentos encontrados ({len(pdf_files)}):")
    for pdf in pdf_files:
        logger.info(f"- {pdf.name}")
    
    # Inicializar procesadores
    logger.info("\nInicializando Document Processor...")
    document_processor = DocumentProcessor(docs_path=str(REGLAMENTOS_DIR))
    
    logger.info("\nInicializando modelo de lenguaje...")
    use_gpt4 = os.getenv("USE_GPT4", "false").lower() == "true"
    logger.info(f"Usando modelo: {'GPT-4' if use_gpt4 else 'GPT-3.5-turbo'}")
    
    # Procesar documentos (esto incluirá la generación de embeddings)
    await asyncio.get_event_loop().run_in_executor(None, document_processor.process_all_documents)
    logger.info(f"Procesador de documentos inicializado en {time.time() - start_time:.2f} segundos")
    
    logger.info(f"Inicialización completada en {time.time() - start_time:.2f} segundos")
    logger.info(f"Sistema listo para procesar consultas con enfoque de páginas vecinas")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializar al arrancar
    await initialize_processors()
    yield
    # Limpiar al cerrar
    logger.info("\nLimpiando recursos...")

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Análisis de Documentos Legales",
    description="API para análisis de documentos legales usando LLMs, técnicas de NLP y procesamiento basado en grafos con rutas paralelas",
    version="1.2.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Modelos Pydantic
class Query(BaseModel):
    question: str
    context: Optional[str] = None

class Response(BaseModel):
    answer: str
    sources: List[str]

# Rutas
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    """
    Procesa una consulta del usuario y devuelve una respuesta basada en los documentos.
    
    El flujo de procesamiento es:
    1. Reformulación de la consulta para mejorar la búsqueda
    2a. Búsqueda de páginas relevantes basada en similitud semántica
    2b. Búsqueda de resúmenes relevantes basada en similitud semántica (paralelo a 2a)
    3. Adición de páginas vecinas para proporcionar contexto adicional
    4. Generación de respuesta basada en ambos contextos: páginas y resúmenes
    """
    try:
        if not document_processor:
            raise HTTPException(
                status_code=503,
                detail="El sistema aún se está inicializando. Por favor, intente nuevamente en unos momentos."
            )
        
        logger.info(f"Recibida consulta: '{query.question}'")
        
        # Procesar la consulta utilizando el grafo unificado con enfoque de páginas vecinas
        start_time = time.time()
        result = await process_query_with_graph(
            query.question, 
            document_processor
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Consulta procesada en {processing_time:.2f} segundos")
        logger.info(f"Fuentes utilizadas: {', '.join(result['sources'])}")
        
        return Response(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Por favor, intente nuevamente."
        )

def start():
    """Función para iniciar el servidor"""
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.backend.api.main:app",
        host=host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    start() 