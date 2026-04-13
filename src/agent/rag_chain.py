import time
from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.llm_factory import get_llm

from src.utils.logging_config import setup_logger
from src.config import settings
from src.exceptions import LLMGenerationError
import openai

logger = setup_logger(__name__)

# Memoria global en RAM (Dict Store) para el control de sesiones
_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Invoca o crea la línea de tiempo de mensajes para una sesión particular."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
        logger.info(f"Nueva sesión conversacional iniciada con ID: {session_id}")
    return _store[session_id]

class RAGAgent:
    """Agente Conversacional con Memoria Corta y Guardrails para Mitigación de Alucinación."""
    
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.llm = self._init_llm()
        self.chain = self._build_chain()
        # umbral de confianza (logits de bge-reranker suelen ser < 0 para no relacionados)
        self.confidence_threshold = 0.15 

    def _init_llm(self) -> Any:
        """Inicializa LLM delegando a la fábrica unificada en llm_factory."""
        if settings.LANGCHAIN_TRACING_V2.lower() == "true":
            logger.info(f"LangSmith Telemetry Activo sobre proyecto: {settings.LANGCHAIN_PROJECT}")
        # Agente conversacional libre: 1000 tokens, sin forzar JSON mode
        return get_llm(max_tokens=1000, require_json=False)

    def _build_chain(self) -> RunnableWithMessageHistory:
        """Construye y acopla el RAG System Prompt + Historia."""
        sys_prompt = """Eres un asistente corporativo experto. Tienes que fundamentar tus respuestas EXCLUSIVAMENTE en el contexto recuperado proporcionado a continuación.
        PROHIBIDO usar conocimiento general. Si el contexto es insuficiente o irrelevante (como preguntas de geografía en un entorno técnico), responde únicamente con la negativa de seguridad. No alucines ni inventes respuestas bajo ninguna circunstancia.
        Si el contexto contiene fórmulas o pasos técnicos, cítalos textualmente. No parafrasees conceptos científicos si no estás 100% seguro.
        Cuando ofrezcas información, DEBES incluir al final de tu respuesta la fuente y la categoría del documento recuperado usando EXACTAMENTE el formato: [Fuente: <valor_origen> | Categoría: <valor_categoria>].
        Responde directamente en texto claro, detallando y explicando la información técnica recuperada.

        Contexto Recuperado:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        self.qa_chain = prompt | self.llm | StrOutputParser()

        # Acopla el historial dinámico transparente
        return RunnableWithMessageHistory(
            self.qa_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def ask(self, question: str, session_id: str = "default_session") -> Dict[str, any]:
        """Flujo Core: Ingesta Query -> Retieve Híbrido -> Validate Guardrail -> Generate Res."""
        logger.info(f"Agente RAG - Procesando query: '{question}' (Sesión: {session_id})")
        start_time = time.time()
        
        # 1. Recuperación 
        logger.info("Iniciando fase de recuperación...")
        docs = self.retriever.search(question)
        
        # 2. Guardrails (Seguridad y Mitigación de Alucinaciones - Hito 3.3)
        if not docs:
            logger.warning("Guardrail activado [G-01]: La búqueda no regresó documentos. Abortando LLM.")
            return {"respuesta": "No tengo suficiente información almacenada para responder a esta pregunta explícitamente.", "fuentes": [], "tiempo_procesamiento_s": 0.0}
            
        top_score = float(docs[0].metadata.get("relevance_score", -999.0))
        if top_score < self.confidence_threshold:
            logger.warning(f"Guardrail activado [G-02]: Top Document Score {top_score:.4f} está debajo del Reranker Threshold ({self.confidence_threshold}). Abortando LLM.")
            return {"respuesta": "No se encontró información relevante en los documentos para responder a esta consulta.", "fuentes": [], "tiempo_procesamiento_s": 0.0}
            
        # 3. Formateo y Generación
        logger.info(f"Guardrails aprobados (Mejor Score: {top_score:.4f}). Empaquetando Contexto para Generación...")
        context_str = "\n\n".join([f"[valor_origen: {d.metadata.get('origen', 'Desconocido')} | valor_categoria: {d.metadata.get('categoria', 'General')}]\n{d.page_content}" for d in docs])
        
        # Mapeando origen de vuelta a formato para guardrails/responses tempranas
        src_map = [
            {
                "origen": d.metadata.get('origen', 'Desconocido'), 
                "categoria": d.metadata.get('categoria', 'General'),
                "score": float(d.metadata.get('relevance_score', 0.0))
            } 
            for d in docs
        ]
        
        try:
            response = self.chain.invoke(
                {"question": question, "context": context_str},
                config={"configurable": {"session_id": session_id}}
            )
        except (TimeoutError, openai.APITimeoutError) as e:
            logger.error(f"Timeout al generar respuesta en la sesión '{session_id}': {e}")
            end_time = time.time()
            return {
                "respuesta": f"La consulta ha superado el tiempo límite de espera ({settings.LLM_TIMEOUT}s) y ha sido cancelada por seguridad. Por favor, intenta de nuevo en unos momentos.",
                "fuentes": src_map,
                "tiempo_procesamiento_s": round(end_time - start_time, 4)
            }
        except Exception as e:
            logger.error(f"Fallo crítico en la generación desde el LLM en la sesión '{session_id}': {e}")
            raise LLMGenerationError(f"Error generando respuesta del LLM MLOps: {e}") from e
        
        end_time = time.time()
        gen_time = end_time - start_time
        logger.info(f"Tiempo total de Generación LLM (End-to-End Fase 3): {gen_time:.4f} segundos")
        
        return {
            "respuesta": response,
            "fuentes": src_map,
            "tiempo_procesamiento_s": round(gen_time, 4)
        }
