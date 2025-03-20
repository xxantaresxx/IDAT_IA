import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Union, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
import graphviz
import uuid
import tempfile
import time
import logging
import operator
from functools import reduce
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar nuestras utilidades existentes
from app.backend.utils.document_processor import DocumentProcessor

# Cargar variables de entorno
load_dotenv()

# Función para combinar listas sin duplicados
def combine_unique_lists(list1, list2):
    """Combina dos listas eliminando duplicados."""
    if not list1:
        return list2 if list2 else []
    if not list2:
        return list1
    return list(set(list1 + list2))

# Función para combinar diccionarios de métricas
def combine_metrics(dict1, dict2):
    """Combina dos diccionarios de métricas."""
    if not dict1:
        return dict2 if dict2 else {}
    if not dict2:
        return dict1
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            # Si ambos son números, sumarlos
            if isinstance(value, (int, float)) and isinstance(result[key], (int, float)):
                result[key] += value
            # Si son listas, combinarlas
            elif isinstance(value, list) and isinstance(result[key], list):
                result[key].extend(value)
            # Si son diccionarios, combinarlos recursivamente
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = combine_metrics(result[key], value)
            # De lo contrario, mantener el valor original
            else:
                pass
        else:
            result[key] = value
    return result

# Función para resolver conflictos en next_step
def resolve_next_step(step1, step2):
    """Resuelve conflictos entre dos valores de next_step.
    
    Estrategia:
    1. Si ambos son iguales, devuelve cualquiera de ellos
    2. Si uno es "__end__", prioriza ese
    3. Si uno es "generate_combined_response", prioriza ese
    4. De lo contrario, prioriza el segundo valor (arbitrariamente)
    """
    if step1 == step2:
        return step1
    if step1 == "__end__" or step2 == "__end__":
        return "__end__"
    if step1 == "generate_combined_response" or step2 == "generate_combined_response":
        return "generate_combined_response"
    # Priorizar arbitrariamente el segundo valor
    return step2

# Definir el estado del grafo con anotaciones para manejo paralelo
class GraphState(TypedDict):
    """Estado del grafo para el procesamiento de consultas."""
    query: str  # Consulta original del usuario
    rephrased_query: str  # Consulta reformulada
    
    # Para la ruta de páginas individuales
    initial_relevant_pages: Optional[List[int]]  # Numero de Páginas relevantes iniciales
    relevant_pages: Optional[List[Dict[str, Any]]]  # Páginas relevantes con contenido, Incluye tanto las páginas originalmente relevantes como sus vecinas (para contexto)
    context_pages: Optional[str]  # Contexto de páginas
    
    # Para la ruta de resúmenes
    relevant_summaries: Optional[List[Dict[str, Any]]]  # Resúmenes relevantes
    context_summaries: Optional[str]  # Contexto de resúmenes
    
    final_response: Optional[str]  # Respuesta final
    sources: Annotated[List[str], combine_unique_lists]  # Fuentes utilizadas con reductor personalizado
    next_step: Annotated[str, resolve_next_step]  # Siguiente paso en el grafo con reductor personalizado
    metrics: Annotated[Dict[str, Any], combine_metrics]  # Métricas para análisis con reductor personalizado

# Inicializar el modelo de lenguaje
def get_openai_client():
    """Obtiene el cliente de OpenAI configurado."""
    # Determinar si usar Azure OpenAI o OpenAI estándar
    azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
    standard_openai_key = os.getenv('OPENAI_API_KEY')
    
    if azure_openai_key:
        # Usar Azure OpenAI
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT no encontrada en variables de entorno")
            
        return AzureOpenAI(
            api_key=azure_openai_key,
            api_version="2023-12-01-preview",
            azure_endpoint=azure_endpoint
        )
    else:
        # Usar OpenAI estándar como fallback
        if not standard_openai_key:
            raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")
            
        return OpenAI(api_key=standard_openai_key)

def get_model_name():
    """Obtiene el nombre del modelo configurado."""
    if os.getenv('AZURE_OPENAI_API_KEY'):
        # Usar el nombre de despliegue de Azure OpenAI
        return os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
    else:
        # Usar OpenAI estándar
        use_gpt4 = os.getenv("USE_GPT4", "false").lower() == "true"
        return "gpt-4" if use_gpt4 else "gpt-3.5-turbo"

# Nodo para reformular la consulta
def rephrase_query(state: GraphState) -> GraphState:
    """Reformula la consulta para mejorar la búsqueda."""
    client = get_openai_client()
    model = get_model_name()
    
    logger.info(f"Reformulando consulta: {state['query']}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """
            Eres un asistente especializado en reformular consultas legales para mejorar la búsqueda de información.
            Tu tarea es reformular la consulta del usuario para hacerla más específica y orientada a la búsqueda en documentos legales.
            No respondas la pregunta, solo reformúlala para mejorar la búsqueda.
            - Puedes corregir los errores gramaticales
            - Puedes mejorar la semántica y orden léxico de la palabras para un mejor entendimiento
            """},
            {"role": "user", "content": f"Reformula esta consulta para mejorar la búsqueda en documentos legales: {state['query']}"}
        ],
        temperature=0.1
    )
    
    rephrased_query = response.choices[0].message.content
    
    logger.info(f"Consulta reformulada: {rephrased_query}")
    
    return {
        "rephrased_query": rephrased_query,
        "metrics": {
            "rephrase": {
                "original_query_length": len(state['query']),
                "rephrased_query_length": len(rephrased_query)
            }
        }
    }

# Nodo para buscar páginas relevantes sin vecinos
def search_relevant_pages(state: GraphState, document_processor: DocumentProcessor) -> GraphState:
    """Busca páginas relevantes para la consulta sin añadir vecinos."""
    query = state.get("rephrased_query")
    
    logger.info(f"Buscando páginas relevantes para: {query}")
    
    # Obtener páginas relevantes sin añadir vecinos
    initial_relevant_pages = document_processor.search_relevant_number_pages(query, k=10)
    
    logger.info(f"Se encontraron {len(initial_relevant_pages)} páginas relevantes iniciales")
    
    # Actualizar el estado con las páginas relevantes iniciales
    return {
        "initial_relevant_pages": initial_relevant_pages,
        "next_step": "add_context_neighbors"
    }

# Nodo para añadir páginas vecinas
def add_context_neighbors(state: GraphState, document_processor: DocumentProcessor) -> GraphState:
    """Añade páginas vecinas a las páginas relevantes para proporcionar contexto adicional."""
    initial_relevant_pages = state.get("initial_relevant_pages", [])
    
    if not initial_relevant_pages:
        logger.warning("No se encontraron páginas relevantes iniciales")
        return {
            "relevant_pages": [],
            "context_pages": "",
            "sources": [],
            "metrics": {
                "pages_path": {
                    "pages_found": 0,
                    "context_added": False
                }
            },
            "next_step": "generate_combined_response"  # Ir directamente al nodo final
        }
    
    # Configuración del número de vecinos (podría ser configurable en el futuro)
    neighbor_count = 2
    
    logger.info(f"Añadiendo hasta {neighbor_count} páginas vecinas a cada página relevante")
    
    # Añadir páginas vecinas
    start_time = time.time()
    expanded_pages = document_processor.add_neighbor_pages(initial_relevant_pages, n=neighbor_count)
    
    # Obtener contenido completo de las páginas expandidas
    relevant_pages = document_processor.get_pages_content(expanded_pages)
    
    # Calcular métricas
    added_pages = len(expanded_pages) - len(initial_relevant_pages)
    processing_time = time.time() - start_time
    
    logger.info(f"Se añadieron {added_pages} páginas vecinas en {processing_time:.2f} segundos")
    logger.info(f"Total de páginas con contexto: {len(expanded_pages)}")
    
    # Extraer fuentes
    sources = list(set([page['metadata']['source'] for page in relevant_pages]))
    
    # Combinar el contenido de todas las páginas para formar el contexto
    context_pages = "\n\n".join([
        f"Página {page['metadata']['page']} de {page['metadata']['source']}:\n{page['page_content']}"
        for page in relevant_pages
    ]) if relevant_pages else ""
    
    # Crear métricas para este camino
    metrics = {
        "pages_path": {
            "initial_pages_count": len(initial_relevant_pages),
            "expanded_pages_count": len(expanded_pages),
            "added_neighbors_count": added_pages,
            "neighbor_processing_time": processing_time,
            "sources_count": len(sources)
        }
    }
    
    return {
        "relevant_pages": relevant_pages,
        "context_pages": context_pages,
        "sources": sources,
        "metrics": metrics,
        "next_step": "generate_combined_response"  # Ir directamente al nodo final
    }

def search_relevant_summaries(state: GraphState, document_processor: DocumentProcessor) -> GraphState:
    """Busca resúmenes relevantes para la consulta."""
    query = state.get("rephrased_query")
    
    logger.info(f"Buscando resúmenes relevantes para: {query}")
    
    # Verificar si tenemos un vector store de resúmenes configurado, es decir si existe una base de datos vectorial de resúmenes
    if not hasattr(document_processor, 'vector_store_summaries') or document_processor.vector_store_summaries is None:
        logger.warning("No se encontró un vector store de resúmenes configurado")
        return {
            "relevant_summaries": [],
            "context_summaries": "",
            "sources": [],
            "metrics": {
                "summaries_path": {
                    "summaries_found": 0,
                    "vector_store_available": False
                }
            },
            "next_step": "generate_combined_response"  # Ir directamente al nodo final
        }
    
    # Usar el retriever de resúmenes
    try:
        # Obtener el retriever
        retriever = document_processor.vector_store_summaries.as_retriever(search_kwargs={"k": 3})
        
        # Buscar documentos relevantes
        docs = retriever.get_relevant_documents(query)
        
        # Procesar los resúmenes encontrados
        relevant_summaries = []
        for doc in docs:
            summary_with_data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "token_count": document_processor.get_token_count(doc.page_content)
            }
            relevant_summaries.append(summary_with_data)
        
        logger.info(f"Se encontraron {len(relevant_summaries)} resúmenes relevantes")
        
        # Combinar el contenido de los resúmenes para formar el contexto
        context_summaries = "\n\n".join([
            f"Resumen {i+1} (Páginas {summary['metadata'].get('page_range', 'N/A')} de {summary['metadata'].get('source', 'Desconocido')}):\n{summary['page_content']}"
            for i, summary in enumerate(relevant_summaries)
        ]) if relevant_summaries else ""
        
        # Extraer fuentes
        summary_sources = list(set([summary['metadata'].get('source', '') for summary in relevant_summaries]))
        
        # Crear métricas para este camino
        metrics = {
            "summaries_path": {
                "summaries_found": len(relevant_summaries),
                "vector_store_available": True,
                "sources_count": len(summary_sources)
            }
        }
        
        return {
            "relevant_summaries": relevant_summaries,
            "context_summaries": context_summaries,
            "sources": summary_sources,
            "metrics": metrics,
            "next_step": "generate_combined_response"  # Ir directamente al nodo final
        }
    except Exception as e:
        logger.error(f"Error al buscar resúmenes: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "relevant_summaries": [],
            "context_summaries": "",
            "sources": [],
            "metrics": {
                "summaries_path": {
                    "error": str(e),
                    "summaries_found": 0,
                    "vector_store_available": True
                }
            },
            "next_step": "generate_combined_response"  # Ir directamente al nodo final
        }

def generate_combined_response(state: GraphState) -> GraphState:
    """Genera una respuesta basada en ambos contextos: páginas y resúmenes."""
    client = get_openai_client()
    model = get_model_name()
    
    # Obtener contextos
    context_pages = state.get("context_pages", "")  # Contexto de páginas
    context_summaries = state.get("context_summaries", "")  # Contexto de resúmenes
    
    # Verificar si tenemos al menos un contexto
    if not context_pages and not context_summaries:
        return {
            "final_response": "No se encontró información relevante para responder a tu consulta.",
            "next_step": "__end__"
        }
    
    # Añadir instrucciones específicas sobre citación
    system_prompt = """
    Eres un asistente legal especializado en contrataciones con el Estado peruano.
    Tu tarea es responder consultas basándote en la información proporcionada en los contextos.
    
    INSTRUCCIONES OBLIGATORIAS SOBRE CITACIÓN:
    1. SIEMPRE cita el nombre completo del documento para cada información que proporciones
    2. Usa el formato "[Nombre del documento], página X" para CADA cita (ejemplo: "Ley General de Contrataciones con el Estado.pdf, página 5")
    3. Cuando cites resúmenes, SIEMPRE incluye el nombre del documento original, no solo "Resumen X"
    4. Si mencionas "Resumen X", DEBES añadir inmediatamente el nombre del documento: "Resumen X de [Nombre completo del documento]"
    5. NO inventes referencias ni números de página
    6. Si no estás seguro de una referencia exacta, cita solo el nombre completo del documento
    7. NUNCA omitas el nombre del documento en ninguna cita

    REGLAS ESTRICTAS:
    1. SOLO puedes responder con información que esté EXPLÍCITAMENTE en los documentos proporcionados
    2. Si la información solicitada NO está en los documentos, di CLARAMENTE: "No encuentro información específica sobre [tema] en los documentos proporcionados."
    3. NUNCA inventes información basada en tu conocimiento general
    4. NUNCA inventes citas o referencias a artículos
    5. Si citas un artículo, DEBES citar el texto exacto del artículo

    FORMATO DE RESPUESTA:
    1. Estructura tu respuesta de manera clara y organizada
    2. Usa viñetas o numeración para listar conceptos cuando sea apropiado
    3. Para cada punto importante, incluye la cita completa con nombre del documento y página
    4. Al final de tu respuesta, incluye una sección "Fuentes:" con la lista completa de documentos consultados

    RECUERDA: Es mejor decir "No lo sé" que proporcionar información incorrecta o inventada.
    """
    
    # Generar respuesta con ambos contextos
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                Consulta: {state['query']}
                
                Contexto de páginas específicas:
                {context_pages}
                
                Contexto de resúmenes:
                {context_summaries}
                
                Por favor, responde la consulta basándote en la información proporcionada en ambos contextos.
                IMPORTANTE: Cita SIEMPRE el nombre completo del documento y la página para cada parte de tu respuesta.
                Cuando menciones información de un resumen, especifica claramente de qué documento proviene ese resumen.
                
                Ejemplo de citación correcta:
                "Según la Ley General de Contrataciones con el Estado.pdf, página 10, los principios fundamentales son..."
                "De acuerdo con el Resumen 1 del documento DS009-2023-EF-Reglamento-ley-de-contrataciones.publicas.pdf, páginas 51-55, se establece que..."
                
                Al final de tu respuesta, incluye una sección "Fuentes:" con la lista completa de documentos consultados.
                """}
            ],
            temperature=0
        )
        
        final_response = response.choices[0].message.content
        processing_time = time.time() - start_time
        
        # Métricas para la respuesta final
        response_metrics = {
            "final_response": {
                "response_length": len(final_response),
                "processing_time": processing_time,
                "model_used": model
            }
        }
        
    except Exception as e:
        logger.error(f"Error al generar respuesta combinada: {str(e)}")
        
        # Reintentar con un contexto más pequeño si es necesario
        if len(state.get("relevant_pages", [])) > 2 or len(state.get("relevant_summaries", [])) > 2:
            logger.info("Reintentando con contexto reducido")
            
            # Reducir contexto de páginas
            reduced_pages = state.get("relevant_pages", [])[:2]
            reduced_context_pages = "\n\n".join([
                f"Página {page['metadata']['page']} de {page['metadata']['source']}:\n{page['page_content']}"
                for page in reduced_pages
            ]) if reduced_pages else ""
            
            # Reducir contexto de resúmenes
            reduced_summaries = state.get("relevant_summaries", [])[:2]
            reduced_context_summaries = "\n\n".join([
                f"Resumen {i+1} (Páginas {summary['metadata'].get('page_range', 'N/A')} de {summary['metadata'].get('source', 'Desconocido')}):\n{summary['page_content']}"
                for i, summary in enumerate(reduced_summaries)
            ]) if reduced_summaries else ""
            
            try:
                start_time = time.time()
                retry_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
                        Consulta: {state['query']}
                        
                        Contexto de páginas específicas (reducido):
                        {reduced_context_pages}
                        
                        Contexto de resúmenes (reducido):
                        {reduced_context_summaries}
                        
                        Por favor, responde la consulta basándote en la información proporcionada en ambos contextos.
                        IMPORTANTE: Cita SIEMPRE el nombre completo del documento y la página para cada parte de tu respuesta.
                        Cuando menciones información de un resumen, especifica claramente de qué documento proviene ese resumen.
                        
                        Ejemplo de citación correcta:
                        "Según la Ley General de Contrataciones con el Estado.pdf, página 10, los principios fundamentales son..."
                        "De acuerdo con el Resumen 1 del documento DS009-2023-EF-Reglamento-ley-de-contrataciones.publicas.pdf, páginas 51-55, se establece que..."
                        
                        Al final de tu respuesta, incluye una sección "Fuentes:" con la lista completa de documentos consultados.
                        """}
                    ],
                    temperature=0
                )
                
                final_response = retry_response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # Métricas para la respuesta final con reintento
                response_metrics = {
                    "final_response": {
                        "response_length": len(final_response),
                        "processing_time": processing_time,
                        "model_used": model,
                        "retry_required": True
                    }
                }
                
            except Exception as retry_error:
                logger.error(f"Error en segundo intento: {str(retry_error)}")
                final_response = f"Lo siento, no pude generar una respuesta debido a un error: {str(e)}"
                
                # Métricas para error en reintento
                response_metrics = {
                    "final_response": {
                        "error": str(retry_error),
                        "retry_failed": True
                    }
                }
        else:
            final_response = f"Lo siento, no pude generar una respuesta debido a un error: {str(e)}"
            
            # Métricas para error sin reintento
            response_metrics = {
                "final_response": {
                    "error": str(e),
                    "retry_not_possible": True
                }
            }
    
    # Actualizar el estado con la respuesta final
    return {
        "final_response": final_response,
        "metrics": response_metrics,
        "next_step": "__end__"
    }

# Crear el grafo de procesamiento
def create_legal_document_graph(document_processor: DocumentProcessor) -> StateGraph:
    """Crea el grafo de procesamiento para documentos legales con rutas paralelas."""
    logger.info("\nCreando grafo de procesamiento para documentos legales...")
    
    # Definir el grafo
    builder = StateGraph(GraphState)
    
    # Añadir nodos
    builder.add_node("rephrase_query", rephrase_query)
    
    # Ruta de páginas individuales
    builder.add_node("search_relevant_pages", lambda state: search_relevant_pages(state, document_processor))
    builder.add_node("add_context_neighbors", lambda state: add_context_neighbors(state, document_processor))
    
    # Ruta de resúmenes
    builder.add_node("search_relevant_summaries", lambda state: search_relevant_summaries(state, document_processor))
    
    # Nodo final para combinar resultados
    builder.add_node("generate_combined_response", generate_combined_response)
    
    # Definir el flujo con rutas paralelas
    builder.add_edge(START, "rephrase_query")
    
    # Después de reformular, bifurcar a ambas rutas en paralelo
    builder.add_edge("rephrase_query", "search_relevant_pages")
    builder.add_edge("rephrase_query", "search_relevant_summaries")
    
    # Continuar la ruta de páginas
    builder.add_edge("search_relevant_pages", "add_context_neighbors")
    
    # Ambas rutas convergen en el nodo final
    builder.add_edge("add_context_neighbors", "generate_combined_response")
    builder.add_edge("search_relevant_summaries", "generate_combined_response")
    
    builder.add_edge("generate_combined_response", END)
    
    # Compilar el grafo
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    """
    # Visualizar el grafo usando Mermaid
    try:
        # Usar ruta absoluta para el directorio de gráficos
        # Obtener la ruta base del proyecto
        base_dir = Path(os.path.abspath(__file__)).parent.parent  # Subir dos niveles desde utils
        static_dir = base_dir / "static" / "graphs"
        
        # Crear directorio si no existe
        static_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo único
        timestamp = int(time.time())
        filename = f"legal_document_graph_{timestamp}.png"
        filepath = static_dir / filename
        
        # Imprimir la ruta completa para verificación
        abs_filepath = os.path.abspath(filepath)
        logger.info(f"Guardando grafo en ruta absoluta: {abs_filepath}")
        
        # Generar la imagen usando draw_mermaid_png
        try:
            mermaid_png = graph.get_graph().draw_mermaid_png()
            
            # Verificar que mermaid_png no sea None y tenga contenido
            if mermaid_png:
                # Guardar la imagen en un archivo
                with open(filepath, 'wb') as f:
                    f.write(mermaid_png)
                
                # Verificar que el archivo se haya creado
                if os.path.exists(filepath):
                    logger.info(f"✅ Archivo creado correctamente en: {filepath}")
                    # Obtener tamaño del archivo para verificación
                    file_size = os.path.getsize(filepath)
                    logger.info(f"Tamaño del archivo: {file_size} bytes")
                    
                    # Registrar la URL relativa para acceder a la imagen
                    graph_url = f"/static/graphs/{filename}"
                    logger.info(f"URL del grafo: {graph_url}")
                else:
                    logger.warning(f"❌ No se pudo encontrar el archivo creado en: {filepath}")
            else:
                logger.error("La generación de la imagen del grafo devolvió un resultado vacío o nulo")
        except Exception as e:
            logger.error(f"Error al generar la imagen del grafo: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error al visualizar el grafo: {str(e)}")
        # Imprimir el traceback completo para depuración
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    """
    return graph

# Función principal para procesar consultas legales
async def process_query_with_graph(query: str, document_processor: DocumentProcessor) -> Dict[str, Any]:
    """Procesa una consulta legal utilizando el enfoque de páginas vecinas y resúmenes.
    
    El flujo de procesamiento es:
    1. Reformulación de la consulta para mejorar la búsqueda
    2. Búsqueda de páginas relevantes basada en similitud semántica (paralelo)
    3. Adición de páginas vecinas para proporcionar contexto adicional
    4. Búsqueda de resúmenes relevantes basada en similitud semántica (paralelo)
    5. Generación de respuesta basada en ambos contextos: páginas y resúmenes
    """
    try:
        # Crear el grafo
        graph = create_legal_document_graph(document_processor)
        
        # Generar un ID único para esta ejecución
        _id = str(uuid.uuid4())
        
        # Ejecutar el grafo con el estado inicial
        initial_state = {
            "query": query,
            "rephrased_query": "",  # Se inicializa vacío, será actualizado por rephrase_query
            "initial_relevant_pages": None,
            "relevant_pages": None,
            "context_pages": None,
            "relevant_summaries": None,
            "context_summaries": None,
            "final_response": None,
            "sources": [],  # Inicializar como lista vacía para las anotaciones
            "metrics": {},  # Inicializar como diccionario vacío para las anotaciones
            "next_step": "rephrase_query"
        }
        
        logger.info(f"Procesando consulta legal: '{query}'")
        
        # Invocar el grafo con configuración de thread_id
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": _id}})
        
        # Incluir métricas en la respuesta para análisis (opcional)
        metrics = result.get("metrics", {})
        if metrics:
            logger.info(f"Métricas de procesamiento: {metrics}")
        
        # Devolver la respuesta final
        return {
            "answer": result.get("final_response", "No se pudo generar una respuesta."),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Error al procesar consulta legal: {str(e)}")
        # Imprimir el traceback completo para depuración
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "answer": f"Error al procesar la consulta: {str(e)}", 
            "sources": []
        } 