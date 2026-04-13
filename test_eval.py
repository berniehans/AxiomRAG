import pytest
import os
import torch
from src.evals.engine import RagasEvaluator
from src.ingestion.embeddings import EmbeddingManager

if __name__ == "__main__":
    from src.config import settings
    print("OPENROUTER_API_KEY IS:", bool(settings.OPENROUTER_API_KEY))
    
    manager = EmbeddingManager()
    evaluator = RagasEvaluator(embeddings_model=manager.get_embeddings())
    
    questions = ["¿Qué indica el reporte de octubre?"]
    ground_truths = ["El reporte indica depreciación por salitre."]
    generated_answers = ["El reporte indica inversiones en oro y plata."] 
    retrieved_contexts = [["Reporte Anual. El flujo depreció por salitre marino."]]
    
    try:
        metrics = evaluator.run_evals(questions, ground_truths, generated_answers, retrieved_contexts)
        print("METRICS:", metrics)
    except Exception as e:
        print("REAL ERROR:", e)
        import traceback
        traceback.print_exc()
