import os
import sys
import json
import gc
import asyncio
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

async def main():
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
        # Retinex Theory (Lightness and Retinex Theory.pdf)
        {
            "question": "¿En qué consiste la teoría Retinex propuesta por Edwin Land?",
            "ground_truth": "La teoría Retinex de Edwin Land propone que el color percibido de los objetos se determina por la reflectancia de las superficies y es independiente de la iluminación de la escena, logrando constancia de color mediante la computación conjunta de la respuesta de la retina y la corteza cerebral."
        },
        {
            "question": "According to Edwin Land's Retinex theory, how does the visual system achieve color constancy?",
            "ground_truth": "The visual system achieves color constancy by computing lightness or color values based on the ratio of radiation intensities at boundary areas across the whole field. This separates the illumination component from the surface reflectance, allowing visual cortex and retina computation to perceive stable surface colors under varying light conditions."
        },
        # Transformers (Attention Is All You Need.pdf)
        {
            "question": "¿Cuál es el componente de atención principal introducido por el paper 'Attention Is All You Need' y cómo se calculan sus pesos?",
            "ground_truth": "El paper introduce la atención de producto escalar escalado (Scaled Dot-Product Attention) y la atención multi-cabezal (Multi-Head Attention). El cálculo de pesos se realiza proyectando linealmente las consultas (Queries), claves (Keys) y valores (Values), aplicando el producto escalar entre consultas y claves, dividiendo por la raíz cuadrada de la dimensión de la clave (d_k), aplicando una función softmax para obtener las ponderaciones, y multiplicándolas por los valores."
        },
        {
            "question": "What is the scaled factor used in the Scaled Dot-Product Attention of the Transformer model, and why is it applied?",
            "ground_truth": "The scaling factor is 1/sqrt(d_k), where d_k is the dimension of the keys. It is applied because for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients where learning becomes difficult."
        },
        {
            "question": "¿Cuál es la diferencia entre Self-Attention y Encoder-Decoder Attention en la arquitectura Transformer original?",
            "ground_truth": "En Self-Attention, todas las consultas (Queries), claves (Keys) y valores (Values) provienen de la misma entrada (la salida de la capa anterior en el codificador o decodificador). En la atención Encoder-Decoder del decodificador, las consultas provienen de la capa anterior del decodificador, mientras que las claves y valores provienen directamente de la salida final del codificador."
        },
        {
            "question": "What model architecture does GPT-1 adopt for its generative pre-training stage, and how does it compare to the encoder-decoder structure of the original Transformer?",
            "ground_truth": "GPT-1 adopts a multi-layer Transformer decoder architecture. Unlike the original Transformer encoder-decoder which uses cross-attention between encoder and decoder, GPT-1's decoder uses only self-attention layers with masked multi-head attention to prevent the model from attending to future context tokens."
        },
        # BERT (BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf)
        {
            "question": "¿Cuáles son las dos tareas de preentrenamiento no supervisadas que utiliza BERT para aprender representaciones de lenguaje?",
            "ground_truth": "BERT se preentrena usando dos tareas no supervisadas: 1) Masked Language Model (MLM), donde se enmascara aleatoriamente un porcentaje de los tokens de entrada para predecirlos usando el contexto bidireccional; y 2) Next Sentence Prediction (NSP), una tarea de clasificación binaria para predecir si una oración B es la continuación lógica de una oración A."
        },
        {
            "question": "How does BERT resolve the constraint of unidirectionality in standard language model pre-training?",
            "ground_truth": "BERT resolves the constraint of unidirectionality by using a Masked Language Model (MLM) pre-training objective. Instead of predicting tokens left-to-right or right-to-left, MLM randomly masks a percentage of input tokens and predicts them using both left and right context bidirectionally across all layers."
        },
        {
            "question": "¿Qué ventajas tiene el preentrenamiento bidireccional de BERT en comparación con arquitecturas unidireccionales de izquierda a derecha?",
            "ground_truth": "El preentrenamiento bidireccional permite que cada representación de token fusione el contexto tanto de la izquierda como de la derecha simultáneamente en todas las capas del modelo. Esto proporciona una representación mucho más rica y robusta para tareas a nivel de token y a nivel de oración en comparación con modelos unidireccionales (como GPT) que solo miran al pasado."
        },
        # LoRA (LORA LOW-RANK ADAPTATION OF LARGE LAN-GUAGE MODELS.pdf)
        {
            "question": "¿Cómo reduce LoRA el número de parámetros entrenables durante el ajuste fino de modelos de lenguaje grandes sin añadir latencia de inferencia?",
            "ground_truth": "LoRA congela los pesos preentrenados del modelo y representa las actualizaciones de la matriz de peso mediante la descomposición de bajo rango utilizando dos matrices de menor rango A y B (donde el rango r << d). No introduce latencia en la inferencia porque las matrices entrenadas A y B se pueden multiplicar y sumar directamente a la matriz de pesos original al finalizar el entrenamiento."
        },
        {
            "question": "In LoRA (Low-Rank Adaptation), how are the weight updates represented mathematically, and how is the adaptation matrix integrated during inference?",
            "ground_truth": "The weight update is represented mathematically as delta_W = B * A, where B is a d x r matrix and A is a r x k matrix with rank r << min(d, k). During inference, the product delta_W = B * A is scaled and added directly to the frozen pre-trained weight matrix W_0, resulting in zero additional inference latency."
        },
        {
            "question": "How does the rank $r$ affect the computational and parameter efficiency of LoRA (Low-Rank Adaptation) updates?",
            "ground_truth": "The rank $r$ is the bottleneck dimension of the low-rank updates (delta_W = B * A). A smaller $r$ (e.g., 1, 2, or 4) reduces the number of trainable parameters and memory footprint during training, while a larger $r$ increases capacity to capture complex updates but increases parameter size and computational cost."
        },
        # RAG (Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf)
        {
            "question": "¿Cómo combina Retrieval-Augmented Generation (RAG) la memoria paramétrica y la no paramétrica para responder preguntas?",
            "ground_truth": "RAG combina memoria paramétrica (representada por un modelo generativo preentrenado secuencia a secuencia, como BART) con memoria no paramétrica (un índice vectorial denso de pasajes de Wikipedia al que se accede mediante un recuperador denso de pasajes, o DPR)."
        },
        {
            "question": "What is the key difference between the two proposed formulations in the RAG paper: RAG-Sequence and RAG-Token?",
            "ground_truth": "The key difference is that RAG-Sequence uses the same retrieved document to generate the entire output sequence, marginalizing over all retrieved documents, whereas RAG-Token retrieves and can transition between different documents for each individual token in the generated response."
        },
        {
            "question": "In Retrieval-Augmented Generation, what model is typically used as the pre-trained seq2seq generator, and how are the retrieved documents prepended to the input?",
            "ground_truth": "The generator is typically a sequence-to-sequence model like BART. The retrieved document passages are prepended to the user query as a prefix, forming a combined input sequence that is processed by the generator's encoder to produce the final response."
        },
        {
            "question": "En el contexto de RAG, ¿cómo funciona el Dense Passage Retriever (DPR) para obtener los documentos relevantes?",
            "ground_truth": "DPR utiliza un codificador de dos torres basado en BERT: un codificador de preguntas que mapea la consulta a un vector denso y un codificador de pasajes que mapea los pasajes de texto a vectores densos en el mismo espacio. La recuperación se realiza buscando los pasajes cuyos vectores tengan el producto escalar o similitud coseno más alta con el vector de la consulta."
        },
        # GANs (Generative Adversarial Nets.pdf)
        {
            "question": "Describe el juego minimax entre el Generador y el Discriminador según la formulación original de GAN de Ian Goodfellow.",
            "ground_truth": "El generador (G) intenta generar muestras realistas para engañar al discriminador a partir de ruido latente, mientras que el discriminador (D) intenta distinguir entre datos reales de entrenamiento y datos sintéticos de G. Esto se formula como un juego minimax donde D maximiza la probabilidad de asignar la etiqueta correcta a ambos y G minimiza la probabilidad de que D detecte sus fallos."
        },
        {
            "question": "What is the optimal value of the discriminator in the global minimum of the minimax game in the original GAN framework, and what distribution does the generator recover?",
            "ground_truth": "At the global minimum of the minimax game, the generator's distribution (p_g) exactly matches the data generating distribution (p_data). In this state, the discriminator's output is D(x) = 1/2 everywhere, meaning it cannot distinguish between real and generated samples."
        },
        {
            "question": "What is the primary role of the generator $G$ and the discriminator $D$ in the training of Generative Adversarial Networks (GANs)?",
            "ground_truth": "The primary role of the generator $G$ is to map noise vectors from a prior distribution to the data space, producing synthetic samples. The primary role of the discriminator $D$ is to estimate the probability that a sample came from the training data rather than $G$. They are trained simultaneously, with $D$ learning to distinguish real from fake and $G$ learning to generate realistic samples to fool $D$."
        },
        # Vision Transformer (AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.pdf)
        {
            "question": "¿Cómo procesa el Vision Transformer (ViT) las imágenes bidimensionales utilizando una arquitectura de Transformer estándar?",
            "ground_truth": "El Vision Transformer (ViT) divide la imagen 2D en parches planos (patches) de tamaño fijo (ej. 16x16), los proyecta linealmente a una dimensión de vector (embedding de parche), les añade codificaciones posicionales, e introduce un token de clasificación especial '[CLASS]' al principio de la secuencia antes de alimentar todo a un codificador Transformer estándar."
        },
        {
            "question": "In the Vision Transformer (ViT) paper, how are 2D images reshaped and prepared before being fed into the Transformer encoder?",
            "ground_truth": "The 2D image is reshaped into a sequence of flattened 2D patches of size P x P. These patches are then mapped to a vector of size d_model using a trainable linear projection, prepended with a learnable classification token [class], and added to 1D learnable position embeddings."
        },
        {
            "question": "En Vision Transformer, ¿cuál es el propósito del token '[CLASS]' o token de clasificación?",
            "ground_truth": "El token '[CLASS]' es un vector de embedding aprendible que se añade al principio de la secuencia de parches de imagen. El estado de salida correspondiente a este token en la última capa del Transformer se utiliza como la representación agregada de la imagen para alimentar la cabeza de clasificación lineal."
        },
        {
            "question": "Why does the Vision Transformer (ViT) perform worse than CNNs when pre-trained on smaller datasets, and how does this change on larger datasets?",
            "ground_truth": "ViT performs worse on smaller datasets because it lacks the inductive biases inherent to CNNs, such as translation equivariance and locality. However, when pre-trained on larger datasets (like ImageNet-21k or JFT-300M), ViT's capacity to learn global relationships overrides the lack of inductive bias, outperforming CNNs at scale."
        },
        # Stable Diffusion / Latent Diffusion Models (High-Resolution Image Synthesis with Latent Diffusion Models.pdf)
        {
            "question": "¿Por qué los modelos de difusión latente (LDM) realizan el proceso de difusión en un espacio latente en lugar del espacio de píxeles directamente?",
            "ground_truth": "Los LDM realizan el proceso de difusión en un espacio latente de menor dimensión (obtenido mediante un codificador autoencoder entrenado) para reducir drásticamente el costo computacional de entrenamiento e inferencia, manteniendo la calidad visual al separar la fase de compresión perceptiva de la fase de generación semántica."
        },
        {
            "question": "What are the two main phases of Latent Diffusion Models (LDMs), and how do they reduce computational complexity during training?",
            "ground_truth": "The two main phases are: 1) a perceptual compression phase, where an autoencoder maps images to a lower-dimensional latent space; and 2) a semantic generation phase, where diffusion and denoising are performed in this latent space. Complexity is reduced because diffusion operates on a lower-dimensional representation rather than high-resolution pixels."
        },
        {
            "question": "¿Cómo utiliza Latent Diffusion Models (LDM) el mecanismo de atención cruzada (cross-attention) para condicionar la generación de imágenes?",
            "ground_truth": "LDM introduce la atención cruzada en la red de reducción de ruido (U-Net) para integrar diversas modalidades de entrada (como texto, mapas de segmentación o representaciones de imágenes). Las claves y los valores se proyectan a partir de la representación del condicionamiento (ej. embeddings de texto de CLIP), y las consultas se proyectan a partir de los estados intermedios de la U-Net."
        },
        # Auto-Encoding Variational Bayes (Auto-Encoding Variational Bayes.pdf)
        {
            "question": "¿Qué es el truco de reparametrización (reparameterization trick) introducido en Auto-Encoding Variational Bayes y por qué es necesario?",
            "ground_truth": "El truco de reparametrización es una técnica que permite propagar el gradiente a través del paso estocástico (muestreo de las variables latentes z) en un codificador variacional. Funciona expresando la variable aleatoria z como una función determinista de los parámetros de la distribución y un ruido auxiliar epsilon, permitiendo calcular derivadas respecto a los parámetros del codificador mediante retropropagación estándar."
        },
        {
            "question": "In Auto-Encoding Variational Bayes, what is the variational lower bound (ELBO) and how is it optimized?",
            "ground_truth": "The Variational Lower Bound (ELBO) is a lower bound on the marginal log-likelihood of the observed data. It consists of two terms: 1) the reconstruction loss (expected log-likelihood under the approximate posterior), and 2) a regularization term (KL divergence between the approximate posterior and the prior distribution of the latent variables). It is optimized by maximizing the bound with respect to both variational and generative parameters using backpropagation."
        },
        # GPT-1 (Improving Language Understanding by Generative Pre-Training.pdf)
        {
            "question": "¿Cuál es el objetivo de entrenamiento de dos etapas propuesto por GPT-1 para el aprendizaje de transferencia en NLP?",
            "ground_truth": "GPT-1 propone un aprendizaje en dos etapas: 1) Preentrenamiento no supervisado, utilizando un objetivo de modelado de lenguaje estándar para aprender parámetros en un gran corpus de texto; y 2) Ajuste fino (fine-tuning) supervisado, adaptando los parámetros a tareas específicas utilizando datos etiquetados con un objetivo de clasificación y un objetivo auxiliar de modelado de lenguaje."
        },
        # Parameter-Efficient Fine-Tuning (Scaling Down to Scale Up A Guide to Parameter-Efficient Fine-Tuning.pdf)
        {
            "question": "De acuerdo con la guía de ajuste fino eficiente en parámetros (PEFT), ¿cuáles son las tres categorías principales en las que se clasifican los métodos PEFT?",
            "ground_truth": "Los métodos PEFT se clasifican en tres categorías principales: 1) Métodos basados en adición (como adaptadores e inyección de prefijos); 2) Métodos basados en selección (que ajustan solo un subconjunto de los pesos existentes del modelo); y 3) Métodos basados en reparametrización (que aprovechan representaciones de bajo rango como LoRA)."
        },
        {
            "question": "What is parameter-efficient fine-tuning (PEFT), and what are the main trade-offs it aims to address compared to full fine-tuning of large models?",
            "ground_truth": "Parameter-efficient fine-tuning (PEFT) is a collection of techniques to adapt pre-trained models to downstream tasks by tuning only a tiny fraction of parameters (often less than 1%). It aims to address the massive storage costs, computational overhead, and risk of catastrophic forgetting associated with full fine-tuning."
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
            resp_payload = await agent.ask(question=q, session_id=f"eval_session_{index}")
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
    asyncio.run(main())
