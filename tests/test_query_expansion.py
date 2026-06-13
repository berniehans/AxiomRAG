import pytest
import asyncio
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStore

from src.retrieval.query_expansion import expand_query_async
from src.retrieval.advanced_retrieval import AdvancedRetriever

class MockVectorStore(VectorStore):
    def add_texts(self, texts, metadatas=None, **kwargs):
        return []
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        return cls()
    def similarity_search(self, query, k=4, **kwargs):
        return []

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_success():
    """
    Verifica que expand_query_async retorne exactamente 3 variantes semánticas
    cuando el LLM devuelve un formato JSON válido y esperado.
    """
    async def mock_call(prompt_input):
        return AIMessage(content='{"queries": ["variante uno", "variante dos", "variante tres"]}')
        
    mock_llm = RunnableLambda(mock_call)
    variations = await expand_query_async("consulta de prueba", llm=mock_llm)
    
    assert len(variations) == 3
    assert variations[0] == "variante uno"
    assert variations[1] == "variante dos"
    assert variations[2] == "variante tres"

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_graceful_degradation_invalid_json():
    """
    Verifica la degradación graciosa ante un JSON inválido:
    debe capturar la excepción y retornar una lista vacía.
    """
    async def mock_call(prompt_input):
        return AIMessage(content="Esta es una respuesta inválida que no es JSON.")
        
    mock_llm = RunnableLambda(mock_call)
    variations = await expand_query_async("consulta de prueba", llm=mock_llm)
    assert variations == []

@pytest.mark.fast
@pytest.mark.asyncio
async def test_query_expander_graceful_degradation_exception():
    """
    Verifica que ante una excepción del LLM, el expander
    capture el error y retorne una lista vacía.
    """
    async def mock_call(prompt_input):
        raise Exception("Fallo crítico de conexión / Timeout")
        
    mock_llm = RunnableLambda(mock_call)
    variations = await expand_query_async("consulta de prueba", llm=mock_llm)
    assert variations == []

@pytest.mark.fast
def test_retriever_search_flow_mocked():
    """
    Verifica que search en AdvancedRetriever invoque similarity_search,
    rerankee y mapee contra docstore correctamente usando mocks.
    """
    mock_vector_store = MockVectorStore()
    mock_docstore = MagicMock()
    mock_reranker = MagicMock()
    
    # Mockear inicializaciones para no cargar modelos reales
    with patch("src.retrieval.advanced_retrieval.AdvancedRetriever._init_compressor"), \
         patch("src.retrieval.advanced_retrieval.AdvancedRetriever.update_bm25_en_caliente"):
         
        retriever = AdvancedRetriever(vector_store=mock_vector_store)
        retriever.docstore = mock_docstore
        retriever.reranker = mock_reranker
        
        # 1. Configurar mocks de comportamiento
        child_doc = Document(page_content="child text", metadata={"doc_id": "parent_1"})
        mock_vector_store.similarity_search = MagicMock(return_value=[child_doc])
        
        reranked_doc = Document(page_content="child text", metadata={"doc_id": "parent_1", "relevance_score": 0.95})
        mock_reranker.compress_documents = MagicMock(return_value=[reranked_doc])
        
        parent_doc = Document(page_content="Parent text full", metadata={"origen": "doc.pdf"})
        mock_docstore.mget = MagicMock(return_value=[parent_doc])
        
        # 2. Ejecutar
        results = retriever.search("query")
        
        # 3. Validar
        assert len(results) == 1
        assert results[0].page_content == "Parent text full"
        assert results[0].metadata["relevance_score"] == 0.95
        
        mock_vector_store.similarity_search.assert_called_once_with("query", k=20)
        mock_reranker.compress_documents.assert_called_once_with([child_doc], "query")
        mock_docstore.mget.assert_called_once_with(["parent_1"])
