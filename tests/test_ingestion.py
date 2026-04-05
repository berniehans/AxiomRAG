import pytest
import torch
from langchain_core.documents import Document
from src.ingestion.parsers import MultimodalParser
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embeddings import EmbeddingManager
from src.exceptions import IngestionError

class MockEmbeddings:
    """Mock inyectable para Pruebas Rápidas Unitarias."""
    def embed_documents(self, texts):
        return [[0.2] * 5 for _ in texts]
    def embed_query(self, text):
        return [0.2] * 5

@pytest.mark.fast
def test_multimodal_parser_not_found():
    """Valida el manejo de excepciones cuando se piden documentos fantasma."""
    parser = MultimodalParser()
    with pytest.raises(IngestionError, match="Falla al extraer datos"):
        parser.parse_pdf("falso.pdf")

@pytest.mark.fast
def test_document_chunker_flow():
    """Valida el flujo de particionamiento Semántico usando DI (Mocks)."""
    chunker = DocumentChunker(embeddings_model=MockEmbeddings())
    docs = [Document(page_content="Iniciando. Segmentando textos largos en la pipeline de RAG.")]
    chunks = chunker.split_documents(docs)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)

@pytest.mark.integration
def test_gpu_embedding_loading():
    """
    Hito 1.3: Prueba End-to-End validando que los Tensores viajan en la GPU para ahorrar VRAM,
    o en su defecto si no hay gráfica, caen en CPU sin romper el flujo.
    """
    manager = EmbeddingManager()
    embeddings = manager.get_embeddings()
    
    # Evaluar inferencia real
    encoded = embeddings.embed_query("Optimizando Pipeline")
    assert len(encoded) > 0
    
    # Trazabilidad MLOps: Validar hardware
    if torch.cuda.is_available():
        assert embeddings.model_kwargs.get("device") == "cuda", "El modelo falló en enrutar a CUDA."
        assert embeddings.model_kwargs.get("torch_dtype") != torch.float16, "No debe usar Float16 si se quitó por conflictos de TypeError."
