import pytest
import torch
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.ingestion.embeddings import EmbeddingManager
from src.retrieval.advanced_retrieval import AdvancedRetriever
from src.agent.rag_chain import RAGAgent

@pytest.fixture(scope="module")
def agent_mlops_setup():
    """Construye un enclave seguro y volatil (Qdrant en RAM) para testear Agente Generativo."""
    embed_manager = EmbeddingManager()
    embeddings = embed_manager.get_embeddings()
    
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client, collection_name="test_collection", embedding=embeddings
    )
    
    retriever = AdvancedRetriever(vector_store=vector_store, chunker=None)
    mock_doc = Document(
        page_content="[PADRE COMPLETO] El reporte de ingeniería declara que el motor CX-490 explotó por microfisuras.",
        metadata={"origen": "auditoria_2024.pdf", "categoria": "Logística"}
    )
    retriever.build_and_index([mock_doc])
    
    agent = RAGAgent(retriever=retriever)
    yield agent
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.mark.asyncio
async def test_agent_parent_context_injection_and_chain(mocker, agent_mlops_setup):
    """
    Verifica la inyección del contexto PADRE TOTAL y la ejecución del Grafo de Estados.
    """
    # 1. Parcheamos el Retriever para inyectar datos controlados (Hito 2.2)
    mock_docs = [
        Document(
            page_content="[PADRE COMPLETO] El motor CX-490 falló por microfisuras.",
            metadata={"relevance_score": 0.88, "origen": "auditoria.pdf", "categoria": "Ingeniería"}
        )
    ]
    mocker.patch("src.retrieval.advanced_retrieval.AdvancedRetriever.search", return_value=mock_docs)

    # 2. Solución Crítica: Parchamos la ejecución asíncrona del LLM y de las salidas estructuradas
    from src.agent.rag_chain import (
        ExpandedQueries, DocumentRelevance, HallucinationAudit, 
        UtilityEvaluation, RefinedQuery
    )
    from langchain_core.runnables import RunnableLambda

    def mock_with_structured_output(schema, *args, **kwargs):
        async def mock_chain_ainvoke(input, *args, **kwargs):
            if schema == ExpandedQueries:
                return ExpandedQueries(queries=["motor failure", "CX-490 issue"])
            elif schema == DocumentRelevance:
                return DocumentRelevance(relevance="yes")
            elif schema == HallucinationAudit:
                return HallucinationAudit(grounded="yes")
            elif schema == UtilityEvaluation:
                return UtilityEvaluation(useful="yes")
            elif schema == RefinedQuery:
                return RefinedQuery(refined_query="refined query text")
            return MagicMock()
        return RunnableLambda(mock_chain_ainvoke)

    mocker.patch("langchain_openai.ChatOpenAI.with_structured_output", side_effect=mock_with_structured_output)

    async def mock_ainvoke(messages_input, *args, **kwargs):
        return AIMessage(content="El motor CX-490 explotó por microfisuras. [Fuente: auditoria.pdf | Categoría: Ingeniería]")
            
    mocker.patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=mock_ainvoke)

    # Acción QA (asíncrona)
    resultado = await agent_mlops_setup.ask("¿Por qué falló el motor?", session_id="test_qa_final")

    # Assertions
    assert "microfisuras" in resultado["respuesta"]
    assert resultado["fuentes"][0]["score"] == 0.88
    assert resultado["fuentes"][0]["origen"] == "auditoria.pdf"
