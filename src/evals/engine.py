import os
import json
from typing import List, Dict, Any, Union
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.llm_factory import get_llm
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class RagasEvaluator:
    """Motor de Evaluación Empírica: Cuantifica Alucinaciones (Faithfulness) e Índices de Memoria Vectorial (Context Precision)."""
    
    def __init__(self, embeddings_model: Any) -> None:
        # Despachamos un LLM robusto libre de JSON mode, ideal para juicios evaluativos RAGAS
        raw_llm = get_llm(max_tokens=2000, require_json=False)
        self.llm_wrapper = LangchainLLMWrapper(raw_llm)
        self.emb_wrapper = LangchainEmbeddingsWrapper(embeddings_model)
        
    def run_evals(self, questions: List[str], ground_truths: List[str], generated_answers: List[str], retrieved_contexts: List[List[str]]) -> Dict[str, Union[float, str]]:
        """Dispara un pipeline RAGAS. Extrae dict y compila a JSON."""
        data_packet = {
            "question": questions,
            "answer": generated_answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths
        }
        
        logger.info(f"Construyendo Dataseta RAGAS con {len(questions)} puntos de prueba observables.")
        hf_dataset = Dataset.from_dict(data_packet)
        
        try:
            logger.info("Lanzando evaluación estructurada RAGAS (Faithfulness & Context Precision). MLOps Backend...")
            result = evaluate(
                dataset=hf_dataset,
                metrics=[faithfulness, context_precision],
                llm=self.llm_wrapper,
                embeddings=self.emb_wrapper
            )
            
            output_file = "ragas_eval_metrics.json"
            import pandas as pd
            import math
            
            # Robust extraction using to_pandas().mean().to_dict()
            try:
                raw_means = result.to_pandas().mean(numeric_only=True).to_dict()
            except Exception as e:
                logger.error(f"Fallo al convertir result a pandas: {e}")
                raw_means = {}

            limpio = {}
            for k, v in raw_means.items():
                if pd.isna(v) or v is None or (isinstance(v, float) and math.isnan(v)):
                    limpio[str(k)] = 0.0
                else:
                    limpio[str(k)] = float(v)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(limpio, f, indent=4)
                
            logger.info(f"✨ Éxito Evals Categórico: Métricas depositadas formalmente en {output_file}")
            return limpio
        except Exception as e:
            logger.error(f"Caída de servidor o token threshold en Eval RAGAS: {e}")
            raise
