import pytest
import torch
from langchain_core.documents import Document
from src.ingestion.embeddings import EmbeddingManager
from src.ingestion.chunking import DocumentChunker
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.advanced_retrieval import AdvancedRetriever

@pytest.fixture(scope="module")
def retriever_setup():
    """Inicialización Singleton MLOps para evitar rebalses de VRAM."""
    embeddings_manager = EmbeddingManager()
    embeddings = embeddings_manager.get_embeddings()
    chunker = DocumentChunker(embeddings_model=embeddings)
    vstore_manager = VectorStoreManager(embeddings_model=embeddings)
    
    retriever = AdvancedRetriever(vector_store=vstore_manager.get_store(), chunker=chunker)
    
    docs_padre = [
        Document(
            page_content="Reporte Anual de Finanzas 2024. Venta de la nave industrial CX-490 con depreciación del 15% por salitre.",
            metadata={"origen": "Finanzas", "categoria": "Reporte"}
        ),
        Document(
            page_content="Manual Operativo Mantenimiento. Las naves requieren validación estructural del techo nivel 5.",
            metadata={"origen": "Operaciones", "categoria": "Manual"}
        )
    ]
    # Mapeo dual Qdrant Vectorial + Diccionario Lexical BM25 (Parent-Child)
    retriever.build_and_index(docs_padre)
    yield retriever
    
    # Guardrail de Higiene VRAM post-Test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.mark.integration
def test_retrieval_ensemble_and_threshold(retriever_setup):
    """
    Asegura que el Bósqueda Híbrida (50/50 BM25 + BGE-M3) detecte el ID 'CX-490'
    y garantice un passing en Reranker de Cross-Encoder superior a Logit > 0.15.
    """
    query = "¿Qué ocurrió con la depreciación de la CX-490?"
    resultados = retriever_setup.search(query)
    
    assert len(resultados) > 0, "¡Zero Match crudo detectado! El retriever híbrido colapsó frente al Query."
    score = resultados[0].metadata.get('relevance_score', 0)
    
    assert score >= 0.15, f"Reranker Guardrail Breach: Score devuelto ({score:.4f}) es sospechoso (< 0.15). Se indujo alucinación."
    assert "CX-490" in resultados[0].page_content, "Parent-Child Falló: El contexto grande no contiene la ID técnica recuperada."
