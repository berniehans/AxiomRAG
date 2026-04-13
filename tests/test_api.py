import pytest
import os
import torch
from fastapi.testclient import TestClient
from src.main import app, app_state
from src.agent.rag_chain import RAGAgent
from src.api.schemas import QueryRequest

client = TestClient(app)

@pytest.fixture(autouse=True)
def teardown_gpu():
    """Libera la Tensor-Flow Cache GPU luego de bombardear los Endpoints HTTP."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.mark.integration
def test_api_ingest_document():
    """Verifica que el API maneje multipart y controle asíncronamente las colas (HTTP 200)."""
    test_pdf = "data/dummy_test_api.pdf"
    os.makedirs("data", exist_ok=True)
    with open(test_pdf, "wb") as f:
        # PDF ficticio pero con Header Mágico válido
        f.write(b"%PDF-1.4\n%EOF")
    
    with open(test_pdf, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("dummy_test_api.pdf", f, "application/pdf")},
            data={"categoria": "Test_CI"}
        )
    
    os.remove(test_pdf)
    assert response.status_code == 200, f"Endpoint /ingest rechazó el archivo: {response.text}"
    assert response.json().get("status") == "processing"

@pytest.mark.integration
def test_api_chat_conversational_endpoint(mocker):
    """
    Mockea la cadena en memoria previniendo consumos VRAM innecesarios, 
    enfocándose 100% en el Request/Response Data Contract de FastAPI.
    """
    # Aislar Agente MLOps del Framework de Servidor
    mock_agent = mocker.MagicMock(spec=RAGAgent)
    mock_agent.ask.return_value = {
        "respuesta": "Salitre detectado en nivel norte.",
        "fuentes": [{"origen":"Doc.pdf","categoria":"General","score":0.99}],
        "tiempo_procesamiento_s": 0.25
    }
    
    app_state["agent"] = mock_agent
    
    payload = {"pregunta": "Identifica las fallas de salitre", "session_id": "test_qa_44"}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "respuesta" in data
    assert data["respuesta"] == "Salitre detectado en nivel norte."
    assert len(data["fuentes"]) > 0

@pytest.mark.fast
def test_api_schema_validations():
    """Prueba el blindaje de payload OOM (max_length=600). Debe retornar 422 Unprocessable."""
    payload = {
        "pregunta": "x" * 601,  # Supera límite
        "session_id": "safe_session_01"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
