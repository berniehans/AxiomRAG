import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from src.agent.rag_chain import RAGAgent, GraphState

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Mocking standard synchronous and asynchronous invoke
    retriever.ainvoke = AsyncMock(return_value=[
        Document(
            page_content="[LOCAL] Información local de prueba.", 
            metadata={"origen": "local.pdf", "categoria": "Ingeniería", "relevance_score": 0.8}
        )
    ])
    retriever.search = MagicMock(return_value=[
        Document(
            page_content="[LOCAL] Información local de prueba.", 
            metadata={"origen": "local.pdf", "categoria": "Ingeniería", "relevance_score": 0.8}
        )
    ])
    return retriever

@pytest.mark.asyncio
async def test_graph_full_flow_success(mocker, mock_retriever):
    """
    Testeo del flujo exitoso del grafo (Retrieve -> Grade: Relevant -> Generate).
    """
    agent = RAGAgent(retriever=mock_retriever)
    
    from src.agent.rag_chain import (
        ExpandedQueries, DocumentRelevance, HallucinationAudit, 
        UtilityEvaluation, RefinedQuery
    )

    from langchain_core.runnables import RunnableLambda

    # Mock for structured outputs
    def mock_with_structured_output(schema, *args, **kwargs):
        async def mock_chain_ainvoke(input, *args, **kwargs):
            if schema == ExpandedQueries:
                return ExpandedQueries(queries=["var1", "var2"])
            elif schema == DocumentRelevance:
                return DocumentRelevance(relevance="yes")
            elif schema == HallucinationAudit:
                return HallucinationAudit(grounded="yes")
            elif schema == UtilityEvaluation:
                return UtilityEvaluation(useful="yes")
            elif schema == RefinedQuery:
                return RefinedQuery(refined_query="refined")
            return MagicMock()
        return RunnableLambda(mock_chain_ainvoke)

    mocker.patch("langchain_openai.ChatOpenAI.with_structured_output", side_effect=mock_with_structured_output)

    # Mock for regular LLM calls
    async def mock_ainvoke(messages, *args, **kwargs):
        return AIMessage(content="Respuesta simulada exitosa. [Fuente: local.pdf | Categoría: Ingeniería]")
            
    mocker.patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=mock_ainvoke)
    
    res = await agent.ask("¿Cuál es la consulta?", session_id="test_success_session")
    
    assert "Respuesta simulada exitosa" in res["respuesta"]
    assert len(res["fuentes"]) == 1
    assert res["fuentes"][0]["origen"] == "local.pdf"


@pytest.mark.asyncio
async def test_graph_flow_fallback_web_search(mocker, mock_retriever):
    """
    Testeo del fallback a búsqueda externa cuando la relevancia local falla (Grade: Irrelevant -> Fallback).
    """
    agent = RAGAgent(retriever=mock_retriever)
    
    from src.agent.rag_chain import (
        ExpandedQueries, DocumentRelevance, HallucinationAudit, 
        UtilityEvaluation, RefinedQuery
    )

    from langchain_core.runnables import RunnableLambda

    # Mock for structured outputs where relevance is no
    def mock_with_structured_output(schema, *args, **kwargs):
        async def mock_chain_ainvoke(input, *args, **kwargs):
            if schema == ExpandedQueries:
                return ExpandedQueries(queries=["var1", "var2"])
            elif schema == DocumentRelevance:
                return DocumentRelevance(relevance="no")
            elif schema == HallucinationAudit:
                return HallucinationAudit(grounded="yes")
            elif schema == UtilityEvaluation:
                return UtilityEvaluation(useful="yes")
            elif schema == RefinedQuery:
                return RefinedQuery(refined_query="refined")
            return MagicMock()
        return RunnableLambda(mock_chain_ainvoke)

    mocker.patch("langchain_openai.ChatOpenAI.with_structured_output", side_effect=mock_with_structured_output)

    # Mock de DDGS
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value.text.return_value = [{"body": "Resultados de búsqueda externa en internet."}]
    mocker.patch("duckduckgo_search.DDGS", return_value=mock_ddgs)

    # Mock for regular LLM calls
    async def mock_ainvoke(messages, *args, **kwargs):
        return AIMessage(content="Respuesta externa. [Fuente: Búsqueda Web (DuckDuckGo) | Categoría: Internet]")
            
    mocker.patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=mock_ainvoke)
    
    res = await agent.ask("¿Cuál es la consulta?", session_id="test_fallback_session")
    
    assert "Respuesta externa" in res["respuesta"]
    assert res["fuentes"][0]["origen"] == "Búsqueda Web (DuckDuckGo)"
