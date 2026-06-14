import os
import json
import sys
import types
from typing import List, Dict, Any, Union

# Stubs para registrar módulos de VertexAI y evitar dependencias ausentes en la inicialización de Ragas
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_stub = types.ModuleType("vertexai")
    vertexai_stub.ChatVertexAI = type("ChatVertexAI", (object,), {}) # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_stub

if "langchain_community.llms" not in sys.modules:
    llms_stub = types.ModuleType("llms")
    llms_stub.VertexAI = type("VertexAI", (object,), {}) # type: ignore
    sys.modules["langchain_community.llms"] = llms_stub

from datasets import Dataset
from ragas import evaluate
from openai import OpenAI
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from src.utils.logging_config import setup_logger
from src.config import settings

logger = setup_logger(__name__)

class RagasEvaluator:
    """Motor de Evaluación Empírica: Cuantifica Alucinaciones e Índices de Memoria Vectorial."""
    
    def __init__(self, embeddings_model: Any) -> None:
        logger.info("Inicializando RagasEvaluator con Clientes Nativos.")
        
        self.openai_client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL
        )
        
        openai_model = ChatOpenAI(
            model=settings.OPENROUTER_DEFAULT_MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL
        )
        self.evaluator_llm = LangchainLLMWrapper(openai_model)
        
        # Envoltura de embeddings para compatibilidad con Ragas
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
        
    def run_evals(self, questions: List[str], ground_truths: List[str], 
                  generated_answers: List[str], retrieved_contexts: List[List[str]]) -> Dict[str, Union[float, str]]:
        """Dispara un pipeline RAGAS y compila los resultados a JSON."""
        data_packet = {
            "question": questions,
            "answer": generated_answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths
        }
        
        logger.info(f"Construyendo Dataset RAGAS con {len(questions)} puntos de prueba.")
        hf_dataset = Dataset.from_dict(data_packet)
        
        try:
            logger.info("Lanzando evaluación estructurada RAGAS (Faithfulness & Context Precision)...")
            
            metrics = [
                Faithfulness(llm=self.evaluator_llm),
                LLMContextPrecisionWithReference(llm=self.evaluator_llm)
            ]

            run_config = RunConfig(max_workers=1, timeout=60, max_retries=3)
            result = evaluate(
                dataset=hf_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings,
                run_config=run_config
            )
            
            output_file = "ragas_eval_metrics.json"
            import pandas as pd
            import math
            
            # Extracción robusta de promedios
            try:
                raw_means = result.to_pandas().mean(numeric_only=True).to_dict()
            except Exception as e:
                logger.error(f"Fallo al procesar resultados: {e}")
                raw_means = {}

            limpio = {}
            for k, v in raw_means.items():
                key = str(k)
                if key == "llm_context_precision_with_reference":
                    key = "context_precision"
                limpio[key] = float(v) if not (pd.isna(v) or v is None) else 0.0
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(limpio, f, indent=4)
                
            logger.info(f"✨ Métricas depositadas en {output_file}")
            return limpio
            
        except Exception as e:
            logger.error(f"Fallo crítico en Eval RAGAS: {e}")
            raise