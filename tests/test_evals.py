import pytest
import os
import torch
from src.evals.engine import RagasEvaluator
from src.ingestion.embeddings import EmbeddingManager

@pytest.fixture(scope="module")
def evaluator_setup():
    """Inyección de Dependencias del Evaluador MLOps usando Embeddings BGE-M3 reales."""
    manager = EmbeddingManager()
    evaluator = RagasEvaluator(embeddings_model=manager.get_embeddings())
    yield evaluator
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.mark.integration
def test_mlops_faithfulness_baseline(evaluator_setup):
    """
    Invoca el motor RagasEvaluator para cerciorarse de que una métrica 
    de Fidelidad (Faithfulness) artificialmente pobre arroje fallo crítico del test < 0.60.
    """
    if os.getenv("SKIP_RAGAS_EVALS", "false").lower() == "true":
         pytest.skip("Test RAGAS saltado por política local. (Heavy API req)")

    # Data de prueba simulada (El contexto dicta X, el LLM inventa Y = Faithfulness Pésimo)
    questions = ["¿Qué indica el reporte de octubre?"]
    ground_truths = ["El reporte indica depreciación por salitre."]
    generated_answers = ["El reporte indica inversiones en oro y plata."] # Fake hallucination
    retrieved_contexts = [["Reporte Anual. El flujo depreció por salitre marino."]]
    
    try:
        # Aquí Ragas invocará secretamente al LLM de Juicio para dictaminar alucinación
        metrics = evaluator_setup.run_evals(
            questions=questions, 
            ground_truths=ground_truths, 
            generated_answers=generated_answers, 
            retrieved_contexts=retrieved_contexts
        )
        
        assert "faithfulness" in metrics, "Módulo evaluator corrompido, no hay faithfulness."
        
        # En caso de una respuesta inventada totalmente, la fidelidad será muy baja (< 0.60)
        # Aseguramos operativamente que el sistema de MLOps sea capaz de advertir la regresión.
        is_failing_baseline = metrics["faithfulness"] < 0.60
        
        # Invertimos el ASsert porque EN ESTE TEST específico de QA, QUEREMOS confirmar 
        # que nuestro evaluador CAPTURA la alucinación (flaggeando score pobre).
        assert is_failing_baseline, f"El motor MLOps no fue lo suficientemente asertivo midiendo Faithfulness. Devolvió: {metrics['faithfulness']}."

    except Exception as e:
        if "API_KEY" in str(e).upper() or "AUTHENTICATION" in str(e).upper():
            pytest.skip("Test abortado localmente: Faltan llaves de entorno (OpenAI/Groq).")
        raise
