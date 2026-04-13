import os
import json
from typing import List, Dict, Any, Union
from datasets import Dataset
from ragas import evaluate
from openai import OpenAI
from ragas.metrics import Faithfulness, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
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
        
        self.evaluator_llm = llm_factory(
            model=settings.OPENROUTER_DEFAULT_MODEL,
            client=self.openai_client
        )
        
        # FIX 3: Inicialización moderna de embeddings sin factory obsoleta
        self.evaluator_embeddings = HuggingFaceEmbeddings(
            model=settings.EMBEDDINGS_MODEL_NAME
        )
        
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
                ContextPrecision(llm=self.evaluator_llm)
            ]

            result = evaluate(
                dataset=hf_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
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

            limpio = {str(k): (float(v) if not (pd.isna(v) or v is None) else 0.0) for k, v in raw_means.items()}
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(limpio, f, indent=4)
                
            logger.info(f"✨ Métricas depositadas en {output_file}")
            return limpio
            
        except Exception as e:
            logger.error(f"Fallo crítico en Eval RAGAS: {e}")
            raise