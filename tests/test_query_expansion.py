import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.retrieval.query_expansion import AsynchronousQueryExpander
from src.retrieval.advanced_retrieval import AdvancedRetriever

class MockAIMessage:
    def __init__(self, content: str):
        self.content = content

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_success():
    """
    Verifica que AsynchronousQueryExpander retorne exactamente 3 variantes semánticas
    cuando el LLM devuelve un formato JSON válido y esperado.
    """
    # 1. Mockear la respuesta del LLM para retornar JSON estructurado correcto
    mock_response = AIMessage(content='{"variaciones": ["variante uno", "variante dos", "variante tres"]}')
    
    with patch("src.retrieval.query_expansion.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        
        # 2. Instanciar expander y ejecutar
        expander = AsynchronousQueryExpander(provider="test")
        variations = await expander.expand_query("consulta de prueba")
        
        # 3. Validaciones
        assert len(variations) == 3
        assert variations[0] == "variante uno"
        assert variations[1] == "variante dos"
        assert variations[2] == "variante tres"
        mock_llm.ainvoke.assert_called_once()

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_graceful_degradation_invalid_json():
    """
    Verifica la degradación graciosa (Graceful Degradation): ante un JSON no estructurado
    o inválido del LLM, el expander debe capturar la excepción y retornar una lista vacía.
    """
    mock_response = AIMessage(content="Esta es una respuesta inválida que no es JSON.")
    
    with patch("src.retrieval.query_expansion.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        
        expander = AsynchronousQueryExpander(provider="test")
        variations = await expander.expand_query("consulta de prueba")
        
        # Debe retornar lista vacía y no colapsar el hilo de ejecución
        assert variations == []

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_graceful_degradation_exception():
    """
    Verifica que ante una falla de red, timeout o excepción general del LLM,
    el expander capture el error y retorne una lista vacía.
    """
    with patch("src.retrieval.query_expansion.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Fallo crítico de conexión / Timeout"))
        mock_get_llm.return_value = mock_llm
        
        expander = AsynchronousQueryExpander(provider="test")
        variations = await expander.expand_query("consulta de prueba")
        
        # Debe retornar lista vacía asegurando degradación graciosa
        assert variations == []

from langchain_core.vectorstores import VectorStore

class MockVectorStore(VectorStore):
    def add_texts(self, texts, metadatas=None, **kwargs):
        return []
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        return cls()
    def similarity_search(self, query, k=4, **kwargs):
        return []

@pytest.mark.fast
def test_retriever_sync_wrapper_delegation(mocker):
    """
    Verifica que el wrapper síncrono 'search' en AdvancedRetriever invoque
    correctamente 'asearch' asíncrono y devuelva la lista de documentos.
    """
    # Mockear el AdvancedRetriever y su método asearch
    mock_docs = [Document(page_content="Contenido recuperado", metadata={"relevance_score": 0.9})]
    
    # Crear mock de retriever sin inicializar pesos de red ni VRAM
    mock_vector_store = MockVectorStore()
    mock_chunker = MagicMock()
    
    with patch("src.retrieval.advanced_retrieval.AdvancedRetriever._init_compressor"), \
         patch("src.retrieval.advanced_retrieval.AdvancedRetriever.update_bm25_en_caliente"):
         
        retriever = AdvancedRetriever(vector_store=mock_vector_store, chunker=mock_chunker)
        
        # Mockear asearch asíncrono puro
        retriever.asearch = AsyncMock(return_value=mock_docs)
        
        # Llamar a search de forma síncrona ordinaria
        results = retriever.search("pregunta síncrona")
        
        assert results == mock_docs
        retriever.asearch.assert_called_once_with("pregunta síncrona")

@pytest.mark.fast
@pytest.mark.asyncio
async def test_retriever_asearch_flow(mocker):
    """
    Prueba que el flujo asíncrono completo 'asearch' en AdvancedRetriever
    genere consultas, busque concurrentemente, deduplique y rerankee de forma segura.
    """
    mock_vector_store = MockVectorStore()
    mock_chunker = MagicMock()
    
    with patch("src.retrieval.advanced_retrieval.AdvancedRetriever._init_compressor"), \
         patch("src.retrieval.advanced_retrieval.AdvancedRetriever.update_bm25_en_caliente"):
         
        retriever = AdvancedRetriever(vector_store=mock_vector_store, chunker=mock_chunker)
        
        # 1. Mockear el expander
        mock_expander = MagicMock()
        mock_expander.expand_query = AsyncMock(return_value=["variacion uno", "variacion dos"])
        retriever.query_expander = mock_expander
        
        # 2. Mockear vector store asimilarity_search y bm25
        doc_1 = Document(page_content="Contenido de chunk 1", metadata={"doc_id": "parent_a"})
        doc_2 = Document(page_content="Contenido de chunk 2", metadata={"doc_id": "parent_a"})
        doc_3 = Document(page_content="Contenido de chunk 3", metadata={"doc_id": "parent_b"})
        
        # asimilarity_search debe devolver listas de documentos
        mock_vector_store.asimilarity_search = AsyncMock(side_effect=[
            [doc_1], # para query original
            [doc_2], # para variacion uno
            [doc_3], # para variacion dos
        ])
        
        # 3. Mockear bm25_retriever si existe
        mock_bm25 = MagicMock()
        mock_bm25.ainvoke = AsyncMock(return_value=[doc_1])
        retriever.bm25_retriever = mock_bm25
        
        # 4. Mockear el reranker
        mock_reranker_result = [
            Document(page_content="Contenido de chunk 1", metadata={"doc_id": "parent_a", "relevance_score": 0.8}),
            Document(page_content="Contenido de chunk 3", metadata={"doc_id": "parent_b", "relevance_score": 0.6})
        ]
        retriever.reranker = MagicMock()
        retriever.reranker.compress_documents = MagicMock(return_value=mock_reranker_result)
        
        # 5. Mockear docstore mget
        parent_doc_a = Document(page_content="Parent A content", metadata={"origen": "a.pdf", "categoria": "A"})
        parent_doc_b = Document(page_content="Parent B content", metadata={"origen": "b.pdf", "categoria": "B"})
        retriever.docstore = MagicMock()
        retriever.docstore.mget = MagicMock(return_value=[parent_doc_a, parent_doc_b])
        
        # Ejecutar asearch
        results = await retriever.asearch("consulta original")
        
        # Validar resultados
        assert len(results) == 2
        assert results[0].metadata["relevance_score"] == 0.8
        assert results[1].metadata["relevance_score"] == 0.6
        
        # Validar que se llamó al expander con la query correcta
        mock_expander.expand_query.assert_called_once_with("consulta original")
        
        # Validar que se buscó con asimilarity_search de forma concurrente para todas las queries
        assert mock_vector_store.asimilarity_search.call_count == 3
        
        # Validar que el reranker se llamó con la query original
        retriever.reranker.compress_documents.assert_called_once()
        called_docs = retriever.reranker.compress_documents.call_args[0][0]
        
        # Debería haber deduplicado por doc_id:
        # doc_1 y doc_2 tienen doc_id = parent_a. Así que solo un fragmento de parent_a pasa.
        assert len(called_docs) == 2

