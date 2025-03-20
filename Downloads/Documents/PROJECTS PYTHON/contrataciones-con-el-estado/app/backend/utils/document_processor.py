import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import numpy as np
import tiktoken
from tqdm import tqdm
from pathlib import Path
import pickle
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import logging
from langchain_core.documents import Document
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class DocumentProcessor:
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path).resolve()
        self.pages = []  # Lista para almacenar páginas completas
        self.total_pages = 0  # Contador total de páginas
        self.summaries = []  # Nueva lista para almacenar resúmenes
        
        # Determinar si usar Azure OpenAI o OpenAI estándar para embeddings
        azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        if azure_openai_key and os.getenv('AZURE_OPENAI_ENDPOINT'):
            # Inicializar modelo de embeddings usando Azure OpenAI
            from langchain_openai import AzureOpenAIEmbeddings
            
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-ada-002",  # Asegúrate de tener este modelo desplegado en Azure
                api_key=azure_openai_key,
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version="2023-12-01-preview"
            )
            logger.info("Usando Azure OpenAI para embeddings")
        else:
            # Inicializar modelo de embeddings usando OpenAI estándar como fallback
            from langchain_openai import OpenAIEmbeddings
            
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("Usando OpenAI estándar para embeddings")
        
        # Inicializar cliente Qdrant
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            print("Conectando a Qdrant Cloud...")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=300
            )
        else:
            print("Usando Qdrant local...")
            self.qdrant_client = QdrantClient(":memory:")
        
        # Crear colecciones si no existen
        self._create_collections()
        
        # Inicializar QdrantVectorStore para páginas
        self.vector_store_pages = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="document_pages_openai",
            embedding=self.embedding_model
        )
        
        # Nuevo: Inicializar QdrantVectorStore para resúmenes
        self.vector_store_summaries = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="document_summaries_openai",
            embedding=self.embedding_model
        )
    
    def _create_collections(self):
        """Crea las colecciones necesarias en Qdrant."""
        # Colección para el enfoque de QdrantVectorStore
        try:
            self.qdrant_client.get_collection("document_pages_openai")
            logger.info("Colección 'document_pages_openai' ya existe")
        except Exception as e:
            logger.info(f"Creando colección 'document_pages_openai': {str(e)}")
            self.qdrant_client.create_collection(
                collection_name="document_pages_openai",
                vectors_config=models.VectorParams(
                    size=1536,  # Dimensión de los embeddings de OpenAI
                    distance=models.Distance.COSINE
                )
            )
        
        # Colección para el enfoque de QdrantVectorStore con resúmenes
        try:
            self.qdrant_client.get_collection("document_summaries_openai")
            logger.info("Colección 'document_summaries_openai' ya existe")
        except Exception as e:
            logger.info(f"Creando colección 'document_summaries_openai': {str(e)}")
            self.qdrant_client.create_collection(
                collection_name="document_summaries_openai",
                vectors_config=models.VectorParams(
                    size=1536,  # Dimensión de los embeddings de OpenAI
                    distance=models.Distance.COSINE
                )
            )
    
    def _get_cache_key(self) -> str:
        """Genera una clave de caché basada en los archivos PDF en el directorio."""
        pdf_files = list(self.docs_path.glob("*.pdf"))
        if not pdf_files:
            return ""
        
        # Ordenar archivos por nombre para consistencia
        pdf_files.sort()
        
        # Crear hash basado en nombres de archivo y fechas de modificación
        hash_input = ""
        for pdf in pdf_files:
            mtime = os.path.getmtime(pdf)
            hash_input += f"{pdf.name}_{mtime}_"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Obtiene la ruta al archivo de caché."""
        # Usa una ubicación configurable por variable de entorno
        cache_dir_path = os.getenv("CACHE_DIR", "cache")
        cache_dir = Path(cache_dir_path)
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"pages_cache_{cache_key}.pkl"
    
    def _save_to_cache(self, cache_key: str):
        """Guarda los datos procesados en caché."""
        if not cache_key:
            return False
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'pages': self.pages,
                    'total_pages': self.total_pages,
                    'summaries': self.summaries
                }, f)
            print(f"Datos guardados en caché: {cache_path}")
            return True
        except Exception as e:
            print(f"Error al guardar en caché: {str(e)}")
            return False
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Carga los datos procesados desde caché."""
        if not cache_key:
            return False
        
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.pages = data['pages']
                self.total_pages = data['total_pages']
                self.summaries = data.get('summaries', [])
            
            logger.info(f"Datos cargados desde caché: {len(self.pages)} páginas y {len(self.summaries)} resúmenes")
            return True
        except Exception as e:
            print(f"Error al cargar desde caché: {str(e)}")
            return False
    
    def process_all_documents(self):
        """Procesa todos los documentos PDF en el directorio."""
        # Intentar cargar desde caché
        cache_key = self._get_cache_key()
        if self._load_from_cache(cache_key):
            # Si los datos se cargaron correctamente, verificar si ya están en Qdrant
            try:
                # Verificar si hay datos en Qdrant
                pages_count = self.qdrant_client.count(collection_name="document_pages_openai").count
                
                if pages_count > 0:
                    print(f"Datos ya cargados en Qdrant: {pages_count} páginas")
                    
                    # Cargar también en las colecciones de QdrantVectorStore
                    self._load_to_vector_store()
                    return
            except Exception as e:
                print(f"Error al verificar datos en Qdrant: {str(e)}")
        
        # Si no se pudo cargar desde caché o no hay datos en Qdrant, procesar los documentos
        pdf_files = list(self.docs_path.glob("*.pdf"))
        
        if not pdf_files:
            print("No se encontraron archivos PDF.")
            return
        
        # Procesar cada PDF
        for pdf in pdf_files:
            print(f"Procesando {pdf}...")
            self.process_pdf_by_pages(str(pdf))
        
        print(f"Total de páginas procesadas: {len(self.pages)}")
        
        # Crear embeddings para páginas
        self.create_embeddings()
        
        # Generar y procesar resúmenes
        summary_groups = self.generate_page_summaries(group_size=5)
        self.create_summaries(summary_groups)
        self.create_summary_embeddings()
        
        # Guardar en caché
        self._save_to_cache(self._get_cache_key())
        
        # Cargar datos usando QdrantVectorStore
        self._load_to_vector_store()
    
    def process_pdf_by_pages(self, pdf_path: str):
        """Procesa un archivo PDF página por página."""
        try:
            # Extraer nombre del archivo
            filename = os.path.basename(pdf_path)
            
            # Leer el PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            self.total_pages += total_pages
            
            # Procesar cada página individualmente
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                
                if page_text:
                    # Almacenar la página completa con metadatos
                    self.pages.append({
                        "page_content": page_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": total_pages
                        }
                    })
            
            return total_pages
        except Exception as e:
            print(f"Error al procesar PDF por páginas {pdf_path}: {str(e)}")
            return 0
    
    def create_embeddings(self):
        """Crea embeddings para las páginas."""
        if not self.pages:
            print("No hay páginas para crear embeddings.")
            return
        
        print(f"Creando embeddings para {len(self.pages)} páginas...")
        
        # Crear embeddings para páginas
        for page in tqdm(self.pages, desc="Creando embeddings para páginas"):
            # Verificar si ya tiene embedding
            if "embedding" not in page:
                try:
                    embedding = self.embedding_model.embed_query(page["page_content"])
                    page["embedding"] = embedding
                except Exception as e:
                    print(f"Error al crear embedding para página: {str(e)}")
    
    def search_relevant_number_pages(self, query: str, k: int = 10) -> List[Dict]:
        """Busca páginas relevantes para una consulta sin añadir vecinos."""
        # Usar el retriever de QdrantVectorStore
        retriever = self.vector_store_pages.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        
        # Extraer números de página y fuentes
        relevant_pages = []
        sources_seen = set()
        
        for doc in docs:
            page_num = doc.metadata.get("page")
            source = doc.metadata.get("source")
            if page_num is not None and source:
                relevant_pages.append(page_num)
                sources_seen.add(source)
        
        # Si tenemos menos de 3 fuentes diferentes, intentar buscar más resultados
        if len(sources_seen) < 3 and len(self.pages) > 0:
            # Obtener todas las fuentes disponibles
            all_sources = set()
            for page in self.pages:
                all_sources.add(page["metadata"]["source"])
            
            # Buscar fuentes que faltan
            missing_sources = all_sources - sources_seen
            
            if missing_sources:
                logger.info(f"Buscando resultados adicionales de: {', '.join(missing_sources)}")
                
                # Para cada fuente que falta, buscar las mejores páginas
                for source in missing_sources:
                    # Filtrar páginas por fuente
                    source_pages = [p for p in self.pages if p["metadata"]["source"] == source]
                    
                    if source_pages:
                        # Usar el retriever con filtro de metadatos para esta fuente
                        filtered_retriever = self.vector_store_pages.as_retriever(
                            search_kwargs={
                                "k": 2,
                                "filter": {"source": source}
                            }
                        )
                        
                        try:
                            source_docs = filtered_retriever.get_relevant_documents(query)
                            for doc in source_docs:
                                page_num = doc.metadata.get("page")
                                if page_num is not None and page_num not in relevant_pages:
                                    relevant_pages.append(page_num)
                        except Exception as e:
                            logger.warning(f"Error al buscar en fuente {source}: {str(e)}")
        
        return relevant_pages
    
    def get_pages_content(self, page_numbers: List[int]) -> List[Dict]:
        """Obtiene el contenido completo de las páginas especificadas."""
        # Obtener contenido completo de las páginas
        page_contents = []
        
        for page_num in page_numbers:
            for page in self.pages:
                if page["metadata"]["page"] == page_num:
                    # Estimar tokens para esta página
                    page_text = page["page_content"]
                    token_count = self.get_token_count(page_text)
                    
                    page_with_data = {
                        "page_content": page["page_content"],
                        "metadata": page["metadata"],
                        "token_count": token_count
                    }
                    page_contents.append(page_with_data)
                    break
        
        # Ordenar por fuente para agrupar
        page_contents.sort(key=lambda x: x['metadata']['source'])
        
        return page_contents
    
    def add_neighbor_pages(self, page_numbers: List[int], n: int = 2) -> List[int]:
        """Añade páginas vecinas (n antes y n después) a la lista de páginas."""
        # Obtener todas las fuentes y sus páginas
        source_pages = {}
        for page in self.pages:
            source = page["metadata"]["source"]
            page_num = page["metadata"]["page"]
            
            if source not in source_pages:
                source_pages[source] = []
            
            source_pages[source].append(page_num)
        
        # Para cada página, añadir vecinos
        expanded_pages = set(page_numbers)
        
        for page_num in page_numbers:
            # Encontrar la fuente de esta página
            page_source = None
            for page in self.pages:
                if page["metadata"]["page"] == page_num:
                    page_source = page["metadata"]["source"]
                    break
            
            if not page_source or page_source not in source_pages:
                continue
            
            # Obtener páginas de esta fuente
            pages_in_source = sorted(source_pages[page_source])
            
            # Encontrar índice de la página actual
            try:
                current_index = pages_in_source.index(page_num)
            except ValueError:
                continue
            
            # Añadir n páginas antes
            for i in range(max(0, current_index - n), current_index):
                expanded_pages.add(pages_in_source[i])
            
            # Añadir n páginas después
            for i in range(current_index + 1, min(current_index + n + 1, len(pages_in_source))):
                expanded_pages.add(pages_in_source[i])
        
        return list(expanded_pages)
    
    def get_token_count(self, text: str) -> int:
        """Cuenta el número de tokens en un texto usando tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error al contar tokens: {str(e)}")
            # Estimación aproximada: 1 token ≈ 4 caracteres en español
            return len(text) // 4 
    
    def _load_to_vector_store(self):
        """Carga los documentos en las colecciones de QdrantVectorStore."""
        print("Cargando documentos en QdrantVectorStore...")
        
        # Cargar páginas
        page_documents = []
        for page_data in self.pages:
            doc = Document(
                page_content=page_data["page_content"],
                metadata=page_data["metadata"]
            )
            page_documents.append(doc)
        
        if page_documents:
            print(f"Cargando {len(page_documents)} páginas en QdrantVectorStore...")
            self.vector_store_pages.add_documents(page_documents)
        
        # Cargar resúmenes
        summary_documents = []
        for summary_data in self.summaries:
            doc = Document(
                page_content=summary_data["page_content"],
                metadata=summary_data["metadata"]
            )
            summary_documents.append(doc)
        
        if summary_documents:
            print(f"Cargando {len(summary_documents)} resúmenes en QdrantVectorStore...")
            self.vector_store_summaries.add_documents(summary_documents)
    
    def generate_page_summaries(self, group_size: int = 5):
        """Genera grupos de páginas para resumir."""
        if not self.pages:
            logger.warning("No hay páginas para resumir.")
            return []
        
        logger.info(f"Generando grupos de {group_size} páginas para resumir...")
        
        # Agrupar páginas por fuente
        pages_by_source = {}
        for page in self.pages:
            source = page["metadata"]["source"]
            if source not in pages_by_source:
                pages_by_source[source] = []
            pages_by_source[source].append(page)
        
        # Ordenar páginas por número de página dentro de cada fuente
        for source in pages_by_source:
            pages_by_source[source].sort(key=lambda x: x["metadata"]["page"])
        
        # Generar grupos de páginas para resumir
        summary_groups = []
        for source, pages in pages_by_source.items():
            for i in range(0, len(pages), group_size):
                group = pages[i:i+group_size]
                if group:
                    # Extraer rango de páginas
                    start_page = group[0]["metadata"]["page"]
                    end_page = group[-1]["metadata"]["page"]
                    
                    # Combinar contenido
                    combined_content = "\n\n".join([
                        f"Página {p['metadata']['page']}:\n{p['page_content']}"
                        for p in group
                    ])
                    
                    summary_groups.append({
                        "content": combined_content,
                        "metadata": {
                            "source": source,
                            "page_range": f"{start_page}-{end_page}",
                            "total_pages": len(group)
                        }
                    })
        
        logger.info(f"Se crearon {len(summary_groups)} grupos para resumir")
        return summary_groups

    def create_summaries(self, summary_groups: List[Dict]):
        """Crea resúmenes para los grupos de páginas usando el LLM."""
        import os
        
        if not summary_groups:
            logger.warning("No hay grupos para resumir.")
            return []
        
        logger.info(f"Creando resúmenes para {len(summary_groups)} grupos...")
        
        # Determinar si usar Azure OpenAI o OpenAI estándar
        azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        standard_openai_key = os.getenv('OPENAI_API_KEY')
        
        if azure_openai_key:
            # Usar Azure OpenAI
            from openai import AzureOpenAI
            
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
            
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT no encontrada en variables de entorno")
            
            client = AzureOpenAI(
                api_key=azure_openai_key,
                api_version="2023-12-01-preview",
                azure_endpoint=azure_endpoint
            )
            
            model = deployment_name  # Usar el nombre de despliegue como modelo
        else:
            # Usar OpenAI estándar como fallback
            from openai import OpenAI
            
            if not standard_openai_key:
                raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")
            
            client = OpenAI(api_key=standard_openai_key)
            
            # Determinar el modelo a usar
            use_gpt4 = os.getenv("USE_GPT4", "false").lower() == "true"
            model = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
        
        summaries = []
        for group in tqdm(summary_groups, desc="Generando resúmenes"):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": """
                        Eres un especialista en resumir documentos legales sobre contrataciones con el Estado peruano.
                        Tu tarea es generar un resumen conciso pero informativo del contenido proporcionado.
                        Incluye los puntos clave, conceptos legales importantes y cualquier información relevante.
                        """},
                        {"role": "user", "content": f"Genera un resumen del siguiente contenido:\n\n{group['content']}"}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                summary_text = response.choices[0].message.content
                
                # Crear documento de resumen
                summary = {
                    "page_content": summary_text,
                    "metadata": group["metadata"],
                    "embedding": None  # Se añadirá después
                }
                
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error al generar resumen: {str(e)}")
        
        logger.info(f"Se generaron {len(summaries)} resúmenes")
        self.summaries = summaries
        return summaries

    def create_summary_embeddings(self):
        """Crea embeddings para los resúmenes."""
        if not self.summaries:
            logger.warning("No hay resúmenes para crear embeddings.")
            return
        
        logger.info(f"Creando embeddings para {len(self.summaries)} resúmenes...")
        
        for summary in tqdm(self.summaries, desc="Creando embeddings para resúmenes"):
            try:
                embedding = self.embedding_model.embed_query(summary["page_content"])
                summary["embedding"] = embedding
            except Exception as e:
                logger.error(f"Error al crear embedding para resumen: {str(e)}") 