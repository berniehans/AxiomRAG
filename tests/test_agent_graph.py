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
    
    # Mock de las llamadas a ChatOpenAI
    async def mock_ainvoke(messages, *args, **kwargs):
        text = str(messages)
        if "queries" in text:
            return AIMessage(content='{"queries": ["var1", "var2"]}')
        elif "relevance" in text:
            return AIMessage(content='{"relevance": "yes"}')
        elif "grounded" in text:
            return AIMessage(content='{"grounded": "yes"}')
        elif "useful" in text:
            return AIMessage(content='{"useful": "yes"}')
        else:
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
    
    # Mock de ChatOpenAI donde la relevancia local es NO
    async def mock_ainvoke(messages, *args, **kwargs):
        text = str(messages)
        if "queries" in text:
            return AIMessage(content='{"queries": ["var1", "var2"]}')
        elif "relevance" in text:
            return AIMessage(content='{"relevance": "no"}')  # Todos los docs locales son irrelevantes
        elif "grounded" in text:
            return AIMessage(content='{"grounded": "yes"}')
        elif "useful" in text:
            return AIMessage(content='{"useful": "yes"}')
        else:
            return AIMessage(content="Respuesta externa. [Fuente: Búsqueda Web (DuckDuckGo) | Categoría: Internet]")
            
    mocker.patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=mock_ainvoke)
    
    # Mock de DuckDuckGoSearchRun
    mock_ddg = mocker.patch("src.agent.rag_chain.DuckDuckGoSearchRun")
    mock_ddg.return_value.run.return_value = "Resultados de búsqueda externa en internet."
    
    res = await agent.ask("¿Cuál es la consulta?", session_id="test_fallback_session")
    
    assert "Respuesta externa" in res["respuesta"]
    assert res["fuentes"][0]["origen"] == "Búsqueda Web (DuckDuckGo)"
