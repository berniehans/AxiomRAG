import os
import sys
import gc
import logging
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.ingestion import MultimodalParser, DocumentChunker, MetadataExtractor, EmbeddingManager
from src.llm_factory import get_llm

def cleanup_memory(*objects):
    """
    Elimina referencias explícitamente y fuerza al ciclo de garbage collection.
    Limpia la cache interna de PyTorch si CUDA está siendo utilizado,
    previniendo los Memory Leaks tras usar modelos gruesos como M3.
    """
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Memoria liberada correctamente (Lifecycle Managment).")

def run_ingestion(pdf_path: str, categoria: str = None, existing_vector_store=None):
    load_dotenv()
    
    if not os.path.exists(pdf_path):
        logger.error(f"El archivo {pdf_path} no existe. Por favor coloca un documento de prueba en esa ruta y vuelve a ejecutar.")
        return

    logger.info(f"--- Iniciando Procesamiento Completo de Ingesta: {os.path.basename(pdf_path)} ---")

    # 1. Extracción con Unstructured (Multimodal)
    logger.info("Fase 1: Extrayendo contenido del documento PDF...")
    parser = MultimodalParser()
    docs = parser.parse_pdf(pdf_path)
    
    if categoria:
        for idx, d in enumerate(docs):
            docs[idx].metadata["categoria"] = categoria
            docs[idx].metadata["origen"] = os.path.basename(pdf_path)
            
    logger.info(f"Se extrajeron {len(docs)} objetos fuente del documento.")

    # 2. Semantic Chunking (Carga a Memoria el Modelo BGE-M3)
    logger.info("Fase 2: Cargando HuggingFace Embeddings y realizando Semantic Chunking...")
    embed_manager = EmbeddingManager(model_name="BAAI/bge-m3")
    embeddings = embed_manager.get_embeddings()
    
    chunker = DocumentChunker(embeddings_model=embeddings)
    chunks = chunker.split_documents(docs)
    
    logger.info(f"Chunking completado exitosamente: Se generaron {len(chunks)} fragmentos semánticos.")

    # 3. Liberación Explícita de Hardware
    logger.info("Fase de Limpieza: Descargando BGE-M3 de memoria para hacer espacio al LLM de Extracción...")
    cleanup_memory(embed_manager, chunker, embeddings)

    # 4. Extracción de Metadatos (Global - Sin Límite de Chunks)
    logger.info("Fase 3: Evaluando contenido y estructurando Metadatos Globales de Negocio (Pydantic)...")
    llm_provider = os.getenv("LLM_PROVIDER", "openrouter")
    chat_llm = get_llm(provider=llm_provider)
    extractor = MetadataExtractor(llm=chat_llm)
    
    full_text = "\n".join([d.page_content for d in docs])
    content_para_llm = full_text[:4000] # Limite prudente para el Prompt Base
    
    try:
        metadata = extractor.extract(content_para_llm)
        logger.info("Resultado de Predicción Pydantic (Visión Documento Total):")
        logger.info(f"  >> Origen: {metadata.origen}")
        logger.info(f"  >> Fecha: {metadata.fecha_emision}")
        logger.info(f"  >> Categoría: {metadata.categoria}")
        logger.info(f"  >> Resumen: {metadata.resumen}")
        
        for d in docs:
            d.metadata.update({"origen": metadata.origen, "fecha": metadata.fecha_emision, "categoria": metadata.categoria, "resumen": metadata.resumen})
        for c in chunks:
            c.metadata.update({"origen": metadata.origen, "fecha": metadata.fecha_emision, "categoria": metadata.categoria, "resumen": metadata.resumen})
    except Exception as e:
        logger.error(f"Fallo en la inferencia Global de Metadatos: {e}")

    logger.info("\n--- Pipeline Fase 1 finalizada. Limpiando contexto LLM ---")
    cleanup_memory(extractor, chat_llm)

    # 5. Ingesta a Bases de Datos (Fase 2)
    logger.info("Fase 4: Volcando resultados estructurados a Qdrant y Storage Híbrido BM25...")
    from src.config import settings
    os.makedirs(settings.QDRANT_PATH, exist_ok=True)
    os.makedirs(settings.LOCAL_STORE_PATH, exist_ok=True)
    
    from src.retrieval.advanced_retrieval import AdvancedRetriever
    
    if existing_vector_store is not None:
        logger.info("Usando vector store proveído por estado global. Evitando redundancia de Embeddings.")
        v_store = existing_vector_store
        retriever = AdvancedRetriever(vector_store=v_store, chunker=None)
    else:
        from src.retrieval.vector_store import VectorStoreManager
        logger.info("Recargando temporalmente los embebimientos BGE-M3 liberados en Fase 1...")
        embed_manager = EmbeddingManager(model_name="BAAI/bge-m3")
        embeddings = embed_manager.get_embeddings()
        chunker = DocumentChunker(embeddings_model=embeddings)
        vmanager = VectorStoreManager(embeddings_model=embeddings)
        v_store = vmanager.get_store()
        retriever = AdvancedRetriever(vector_store=v_store, chunker=chunker)
    
    retriever.build_and_index(docs, semantic_chunks=chunks) # Ingesta padre e hijos directamente
    logger.info("--- Fase 2 Completada: Pipeline de Ingesta y Recuperación MLOps operativo al 100% ---")

if __name__ == "__main__":
    # Creamos un bloque base resolviendo un target default en la carpeta data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    import glob
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No se encontraron archivos PDF en {data_dir}. Añade documentos y vuelve a ejecutar.")
        
    for pdf in pdf_files:
        logger.info(f"🚀 Iniciando Full Ingestion Pipeline para archivo: {pdf}")
        run_ingestion(pdf)
