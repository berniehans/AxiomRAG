import sys
import os
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.embeddings import EmbeddingManager
from src.ingestion.chunking import DocumentChunker
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.advanced_retrieval import AdvancedRetriever
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("===== INICIANDO HITO 2: VALIDACIÓN RETRIEVAL AVANZADO =====")
    
    # 1. Instanciar Modelos Locales
    embeddings_manager = EmbeddingManager()
    embeddings = embeddings_manager.get_embeddings()
    
    chunker = DocumentChunker(embeddings_model=embeddings)
    vstore_manager = VectorStoreManager(embeddings_model=embeddings)
    
    # 2. Orquestar Retriever Avanzado
    retriever = AdvancedRetriever(vector_store=vstore_manager.get_store(), chunker=chunker)
    
    # 3. Macro Documentos de Prueba Sintéticos
    docs_padre = [
        Document(
            page_content="Reporte Anual de Finanzas 2024. El flujo de caja en octubre tuvo un incremento "
                         "significativo debido a la venta de la nave industrial CX-490 en el distrito norte. "
                         "La nave CX-490 presentaba un grado de depreciación del 15% por impacto del salitre. La junta "
                         "dictaminó su liquidación exitosa evitando el colapso del portafolio.",
            metadata={"origen": "Finanzas", "categoria": "Reporte Anual"}
        ),
        Document(
            page_content="Manual Operativo de Mantenimiento. Las naves industriales del consorcio requieren validación "
                         "estructural del techo nivel 5 cada semestre para vigilar fisuras producidas por factores salitrales. El informe de evaluación debe ser firmado "
                         "por un especialista certificado ISO-9001. La omisión puede resultar en multas graves.",
            metadata={"origen": "Operaciones", "categoria": "Manual Operativo"}
        )
    ]
    
    # 4. Ingesta: Fragmenta semánticamente y mapea a BM25 + Qdrant
    retriever.build_and_index(docs_padre)
    
    # 5. Ejecutar Búsqueda Extremadamente Ambigua
    query = "¿Qué ocurrió con la depreciación de la CX-490 y por qué se dio?"
    resultados = retriever.search(query)
    
    # 6. Mostrar el veredicto del Cross-Encoder
    logger.info("===== RESULTADOS DEL RERANKER (Top Final Contextual) =====")
    for i, doc in enumerate(resultados):
        score = doc.metadata.get('relevance_score', 0)
        logger.info(f"Ranking {i+1} || Score: {score:.4f} || Categoria: {doc.metadata.get('categoria')}")
        logger.info(f"Extracto de Padre Recuperado: {doc.page_content[:150]}...\n")
        
if __name__ == "__main__":
    main()
