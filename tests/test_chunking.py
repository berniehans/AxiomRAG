import pytest
from langchain_core.documents import Document
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embeddings import EmbeddingManager

class MockEmbeddings:
    """Mock inyectable para simular el comportamiento numérico de HuggingFaceEmbeddings en CPU limitados."""
    def embed_documents(self, texts):
        return [[0.5] * 10 for _ in texts]
    def embed_query(self, text):
        return [0.5] * 10

@pytest.mark.fast
def test_document_chunker_basic_split():
    # Uso de Dependency Injection para inyectar matriz mockeada
    mock_emb = MockEmbeddings()
    chunker = DocumentChunker(embeddings_model=mock_emb)
    
    docs = [Document(page_content="Este es el primer párrafo de prueba. Aquí inicia el segundo párrafo con nueva información.")]
    chunks = chunker.split_documents(docs)
    
    # Langchain semantic chunker debería al menos devolver validamente un resultado
    assert len(chunks) > 0

@pytest.mark.integration
def test_document_chunker_real_bge_m3_load():
    """Prueba pesada de integración: forzará la inicialización del modelo en torch. NO usar en CI de bajos recursos."""
    manager = EmbeddingManager()
    real_embeddings = manager.get_embeddings()
    
    chunker = DocumentChunker(embeddings_model=real_embeddings)
    docs = [Document(page_content="Verificando compatibilidad de pipeline con el modelo fp16. Test integral en progreso.")]
    
    chunks = chunker.split_documents(docs)
    assert len(chunks) > 0
