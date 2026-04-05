import os
import sys
import json
import gc
from dotenv import load_dotenv

# Resolver paths correctamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.advanced_retrieval import AdvancedRetriever
from src.agent.rag_chain import RAGAgent
from src.evals.engine import RagasEvaluator
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

def main():
    load_dotenv()
    
    print("🚀 Iniciando Motor de Evaluación MLOps con Ragas...")

    # 1. Carga de Componentes
    logger.info("Cargando EmbeddingManager...")
    embed_manager = EmbeddingManager(model_name="BAAI/bge-m3")
    embeddings = embed_manager.get_embeddings()

    logger.info("Conectando a Qdrant vía VectorStoreManager...")
    vmanager = VectorStoreManager(embeddings_model=embeddings)
    v_store = vmanager.get_store()
    
    # Verificación de Índice
    doc_count = vmanager.client.count(collection_name=vmanager.collection_name).count
    logger.info(f"Documentos detectados en Qdrant: [{doc_count}]")

    logger.info("Inicializando AdvancedRetriever y sincronizando pipeline...")
    retriever = AdvancedRetriever(vector_store=v_store, chunker=None)
    retriever.update_bm25_en_caliente()

    logger.info("Inicializando Agente RAG...")
    agent = RAGAgent(retriever=retriever)
    
    # Modo Debug: Ignorar el confidence_threshold temporalmente para forzar que responda y evaluar recuperaciones bajas
    agent.confidence_threshold = 0.0

    # 2. Golden Dataset (Ejemplo enfocado a la teoría Retinex de Edwin Land)
    golden_dataset = [
        {
            "question": "¿En qué consiste la teoría Retinex propuesta por Edwin Land?",
            "ground_truth": "La teoría Retinex de Edwin Land propone que el color percibido de los objetos se determina por la reflectancia de las superficies y es independiente de la iluminación de la escena, logrando constancia de color mediante la computación conjunta de la respuesta de la retina y la corteza cerebral."
        },
        {
            "question": "¿Cuál es el objetivo principal de los algoritmos basados en Retinex en el procesamiento de imágenes?",
            "ground_truth": "Su objetivo principal es separar la imagen en dos componentes: la iluminación (que suele variar suavemente) y la reflectancia (que contiene los detalles de los objetos), para así mejorar imágenes con baja iluminación o corregir desviaciones de color."
        },
        {
            "question": "¿Cómo maneja el algoritmo Single Scale Retinex (SSR) la estimación de la iluminación?",
            "ground_truth": "El algoritmo SSR estima la componente de iluminación aplicando un filtro paso bajo (típicamente un filtro Gaussiano) a la imagen original, y luego obtiene la reflectancia restando esta iluminación estimada en el dominio logarítmico."
        }
    ]

    # Preparar listas nativas para enviarlas a RagasEvaluator
    questions = []
    ground_truths = []
    generated_answers = []
    retrieved_contexts = []

    # 3. Recolección de Evidencia
    print("\n🧐 Iniciando Inferencia contra Golden Dataset...")
    for index, item in enumerate(golden_dataset):
        q = item["question"]
        gt = item["ground_truth"]
        
        logger.info(f"\n[Evaluando Pregunta {index+1}/{len(golden_dataset)}] -> {q}")
        
        # Obtención de contextos crudos recuperados por Retrieval
        docs_recuperados = retriever.search(q)
        contexts = [d.page_content for d in docs_recuperados]
        
        # Llamada al sistema RAG para generar la respuesta contextualizada final
        try:
            resp_payload = agent.ask(question=q, session_id=f"eval_session_{index}")
            ans = resp_payload["respuesta"]
        except Exception as e:
            logger.error(f"Fallo generando inferencia del LLM: {e}")
            ans = "Error al general respuesta LLM."
        
        questions.append(q)
        ground_truths.append(gt)
        retrieved_contexts.append(contexts)
        generated_answers.append(ans)
        
        logger.info(f"Respuesta Lograda: {ans[:200]}...")

    # Forzar recolección de basuras para estabilizar consumo del LLM
    del agent
    del retriever
    gc.collect()

    # 4. Filtro de Tokens y Ajuste de System Prompt para Evaluación RAGAS
    print("\n🔬 Procesando y Filtrando Respuestas (Context Precision & Faithfulness)...")
    
    valid_q, valid_gt, valid_ans, valid_ctx = [], [], [], []
    invalid_count = 0
    
    for q, gt, ans, ctx in zip(questions, ground_truths, generated_answers, retrieved_contexts):
        if "Negativa de seguridad" in ans or "No se encontró información" in ans or "no tengo suficiente información" in ans.lower() or "no se encontró información relevante" in ans.lower():
            invalid_count += 1
            logger.info("⚠️ Aplicando score automático 0.0 a respuesta vacía/bloqueada (Ahorro de Tokens LLM).")
        else:
            valid_q.append(q)
            valid_gt.append(gt)
            valid_ans.append(ans)
            valid_ctx.append(ctx)

    evaluator = RagasEvaluator(embeddings_model=embeddings)
    
    logger.info(f"📊 Desglose de Evaluación: {len(valid_q)} Éxitos de Recuperación vs {invalid_count} Bloqueos de Seguridad.")
    
    if valid_q:
        logger.info("🔍 Verificación Visual de Datos al Juez (Muestra 1):")
        ctx_text = valid_ctx[0][0][:100].replace('\n', ' ') if valid_ctx[0] else 'Vacío'
        ans_text = valid_ans[0][:100].replace('\n', ' ')
        logger.info(f"--- Contexto parcial: {ctx_text}...")
        logger.info(f"--- Respuesta parcial: {ans_text}...")
        logger.info(f"Evaluando empíricamente {len(valid_q)} preguntas válidas con RAGAS...")
        metrics = evaluator.run_evals(
            questions=valid_q,
            ground_truths=valid_gt,
            generated_answers=valid_ans,
            retrieved_contexts=valid_ctx
        )
    else:
        metrics = {"faithfulness": 0.0, "context_precision": 0.0}
        
    # Recalibración Matemática de Promedios con invalid_count para JSON final
    if invalid_count > 0:
        total = len(questions)
        val_f = metrics.get("faithfulness", 0.0)
        val_c = metrics.get("context_precision", 0.0)
        
        metrics["faithfulness"] = float((val_f * len(valid_q) + 0.0 * invalid_count) / total)
        metrics["context_precision"] = float((val_c * len(valid_q) + 0.0 * invalid_count) / total)
        
        with open("ragas_eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    # 5. Reporte en consola
    print("\n")
    print("="*60)
    print(f"{'MÉTRICA DE RAGAS':<30} | {'PROMEDIO OBTENIDO'}")
    print("="*60)
    for kw, val in metrics.items():
        if isinstance(val, (int, float)):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        print(f"\033[96m{kw.upper():<30}\033[0m | \033[92m{val_str}\033[0m")
    print("="*60)
        
    print(f"\n✅ Resultados persistidos oficialmente y de forma exitosa en el disco (ragas_eval_metrics.json)")

if __name__ == "__main__":
    main()
