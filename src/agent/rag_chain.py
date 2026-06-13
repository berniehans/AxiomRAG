import time
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.llm_factory import get_llm
from src.retrieval.query_expansion import expand_query_async
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


class GraphState(TypedDict):
    original_query: str
    expanded_queries: List[str]
    retrieved_documents: List[Document]
    relevance_scores: List[float]
    loop_count: int
    final_answer: str
    session_id: str
    sources: List[Dict[str, Any]]


class RAGAgent:
    """Agente Conversacional con Memoria Corta, Grafo de Estados (Self-Corrective RAG) y Guardrails."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.confidence_threshold = 0.15
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construye y compila la topología del StateGraph con LangGraph."""
        
        async def expand_query_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            logger.info(f"[Grafo] Expandiendo consulta original: '{query}'")
            
            llm = ChatOpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
                model=settings.OPENROUTER_FAST_MODEL,
                temperature=0.0,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
                extra_body={"reasoning_effort": "low"}
            )
            
            class ExpandedQueries(BaseModel):
                queries: List[str] = Field(
                    description="Exactly 3 technical search query variations (queries), synonyms, or acronyms of the original query."
                )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Eres un experto en optimización de búsquedas técnicas. Tu tarea es generar variantes de búsqueda (queries) "
                    "que ayuden a encontrar la información relevante en una base de datos vectorial y motor léxico. "
                    "Genera exactamente 3 variaciones técnicas y precisas de la consulta original."
                )),
                ("human", "Consulta original: {query}")
            ])
            
            chain = prompt | llm.with_structured_output(ExpandedQueries)
            try:
                res = await chain.ainvoke({"query": query})
                expanded = res.queries
                logger.info(f"[Grafo] Variantes de queries generadas: {expanded}")
                return {"expanded_queries": expanded}
            except Exception as e:
                logger.error(f"[Grafo] Error al expandir consulta: {e}. Usando fallback vacío.")
                return {"expanded_queries": []}

        async def retrieve_local_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            expanded = state.get("expanded_queries", [])
            all_queries = [query] + expanded
            
            logger.info(f"[Grafo] Recuperando localmente para consultas: {all_queries}")
            
            tasks = [self.retriever.ainvoke(q) for q in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            seen_contents = set()
            unique_docs = []
            for res in results:
                if isinstance(res, list):
                    for doc in res:
                        c = doc.page_content.strip()
                        if c not in seen_contents:
                            seen_contents.add(c)
                            unique_docs.append(doc)
                            
            logger.info(f"[Grafo] Recuperados {len(unique_docs)} documentos únicos (Padres Completos).")
            return {"retrieved_documents": unique_docs}

        async def grade_documents_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            docs = state.get("retrieved_documents", [])
            if not docs:
                return {"retrieved_documents": []}
                
            logger.info(f"[Grafo] Evaluando relevancia de {len(docs)} documentos...")
            llm = get_llm(require_json=True)
            
            async def grade_doc(doc: Document) -> Optional[Document]:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "Eres un evaluador de relevancia. Tu tarea es determinar si un fragmento de documento recuperado "
                        "es relevante para responder a la consulta del usuario. Retorna estrictamente un objeto JSON con la "
                        "clave 'relevance' y valor 'yes' (si es relevante) o 'no' (si no es relevante)."
                    )),
                    ("human", "Consulta: {query}\n\nDocumento:\n{doc_content}")
                ])
                chain = prompt | llm
                try:
                    res = await chain.ainvoke({"query": query, "doc_content": doc.page_content})
                    content = res.content.strip()
                    if content.startswith("```"):
                        if content.startswith("```json"):
                            content = content[7:]
                        else:
                            content = content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()
                    data = json.loads(content)
                    if data.get("relevance", "").lower() == "yes":
                        return doc
                except Exception as e:
                    logger.warning(f"Error evaluando relevancia (documento aceptado por fallback): {e}")
                    return doc
                return None
                
            tasks = [grade_doc(d) for d in docs]
            results = await asyncio.gather(*tasks)
            filtered = [d for d in results if d is not None]
            
            logger.info(f"[Grafo] {len(filtered)} de {len(docs)} documentos pasaron el filtro de relevancia del LLM.")
            return {"retrieved_documents": filtered}

        async def web_search_fallback_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            logger.warning(f"[Grafo/Fallback] Relevancia insuficiente en documentos locales. Iniciando búsqueda externa para: '{query}'")
            
            try:
                search = DuckDuckGoSearchRun()
                search_result = await asyncio.to_thread(search.run, query)
                web_doc = Document(
                    page_content=search_result,
                    metadata={"origen": "Búsqueda Web (DuckDuckGo)", "categoria": "Internet", "relevance_score": 0.5}
                )
                return {"retrieved_documents": [web_doc]}
            except Exception as e:
                logger.error(f"[Grafo/Fallback] Error al ejecutar búsqueda web externa: {e}")
                return {"retrieved_documents": []}

        async def generate_answer_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            docs = state.get("retrieved_documents", [])
            session_id = state.get("session_id", "default_session")
            
            if not docs:
                return {
                    "final_answer": "No tengo suficiente información almacenada ni en la red para responder a esta consulta.",
                    "sources": []
                }
                
            logger.info(f"[Grafo] Generando respuesta final basada en {len(docs)} documentos...")
            
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
            
            history = get_session_history(session_id)
            history_messages = history.messages
            
            context_str = "\n\n".join([f"[valor_origen: {d.metadata.get('origen', 'Desconocido')} | valor_categoria: {d.metadata.get('categoria', 'General')}]\n{d.page_content}" for d in docs])
            llm = ChatOpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
                model=settings.OPENROUTER_DEFAULT_MODEL,
                temperature=0.0,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES,
                max_tokens=1000,
                max_completion_tokens=1000,
                extra_body={"reasoning_effort": "high"}
            )
            chain = prompt | llm | StrOutputParser()
            
            try:
                response = await chain.ainvoke({
                    "question": query,
                    "context": context_str,
                    "history": history_messages
                })
            except (TimeoutError, openai.APITimeoutError) as e:
                logger.error(f"Timeout al generar respuesta en sesión '{session_id}': {e}")
                response = "La consulta ha superado el tiempo límite de espera y ha sido cancelada por seguridad."
            except Exception as e:
                logger.error(f"Error en generación LLM: {e}")
                raise LLMGenerationError(f"Error generando respuesta del LLM MLOps: {e}") from e
                
            history.add_user_message(query)
            history.add_ai_message(response)
            
            src_map = [
                {
                    "origen": d.metadata.get('origen', 'Desconocido'), 
                    "categoria": d.metadata.get('categoria', 'General'),
                    "score": float(d.metadata.get('relevance_score', 0.0))
                } 
                for d in docs
            ]
            
            return {
                "final_answer": response,
                "sources": src_map
            }

        async def refine_query_node(state: GraphState) -> Dict[str, Any]:
            query = state["original_query"]
            loop_count = state.get("loop_count", 0)
            logger.info(f"[Grafo] Auto-Corrección: Refinando consulta original '{query}'...")
            
            llm = get_llm(require_json=True)
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Eres un experto en optimización de consultas RAG. Tu tarea es reescribir la consulta del usuario "
                    "para mejorar la precisión y exhaustividad de la búsqueda vectorial y léxica, considerando que el "
                    "intento anterior falló debido a alucinaciones o falta de utilidad. "
                    "Retorna un objeto JSON con la clave 'refined_query' que contenga la nueva consulta optimizada."
                )),
                ("human", "Consulta original: {query}")
            ])
            chain = prompt | llm
            refined = query
            try:
                res = await chain.ainvoke({"query": query})
                content = res.content.strip()
                if content.startswith("```"):
                    if content.startswith("```json"):
                        content = content[7:]
                    else:
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                data = json.loads(content)
                refined = data.get("refined_query", query)
                logger.info(f"[Grafo] Consulta optimizada (Loop {loop_count + 1}): '{query}' -> '{refined}'")
            except Exception as e:
                logger.error(f"[Grafo] Error al refinar consulta: {e}")
                
            return {
                "original_query": refined,
                "loop_count": loop_count + 1
            }

        workflow = StateGraph(GraphState)
        
        workflow.add_node("expand_query", expand_query_node)
        workflow.add_node("retrieve_local", retrieve_local_node)
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("web_search_fallback", web_search_fallback_node)
        workflow.add_node("generate_answer", generate_answer_node)
        workflow.add_node("refine_query", refine_query_node)
        
        workflow.add_edge(START, "expand_query")
        workflow.add_edge("expand_query", "retrieve_local")
        workflow.add_edge("retrieve_local", "grade_documents")
        
        def route_after_grading(state: GraphState) -> str:
            docs = state.get("retrieved_documents", [])
            if not docs:
                return "web_search_fallback"
            return "generate_answer"
            
        workflow.add_conditional_edges(
            "grade_documents",
            route_after_grading,
            {
                "web_search_fallback": "web_search_fallback",
                "generate_answer": "generate_answer"
            }
        )
        
        workflow.add_edge("web_search_fallback", "generate_answer")
        
        async def check_groundedness_and_utility(state: GraphState) -> str:
            docs = state.get("retrieved_documents", [])
            answer = state.get("final_answer", "")
            query = state["original_query"]
            loop_count = state.get("loop_count", 0)
            
            if not docs or not answer or "No tengo suficiente información" in answer:
                return "end"
                
            if loop_count >= 3:
                logger.warning(f"[Grafo/Auto-Corrección] Límite de re-intentos alcanzado ({loop_count}). Saltando bucle.")
                return "end"
                
            llm = get_llm(require_json=True)
            context_str = "\n\n".join([d.page_content for d in docs])
            
            prompt_grounded = ChatPromptTemplate.from_messages([
                ("system", (
                    "Eres un auditor de alucinaciones. Tu tarea es verificar si la respuesta generada está completamente "
                    "fundamentada y respaldada por los documentos de contexto proporcionados. Retorna estrictamente un objeto JSON "
                    "con la clave 'grounded' y valor 'yes' (si está fundamentada sin inventar nada) o 'no' (si contiene alucinaciones)."
                )),
                ("human", "Contexto:\n{context}\n\nRespuesta:\n{answer}")
            ])
            chain_grounded = prompt_grounded | llm
            try:
                res_g = await chain_grounded.ainvoke({"context": context_str, "answer": answer})
                content = res_g.content.strip()
                if content.startswith("```"):
                    if content.startswith("```json"):
                        content = content[7:]
                    else:
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                data_g = json.loads(content)
                grounded = data_g.get("grounded", "").lower() == "yes"
            except Exception as e:
                logger.warning(f"Error evaluando groundedness: {e}")
                grounded = True
                
            if not grounded:
                logger.warning("[Grafo/Auto-Corrección] Groundedness check fallido. Activando re-enrutamiento.")
                return "refine"
                
            prompt_utility = ChatPromptTemplate.from_messages([
                ("system", (
                    "Eres un evaluador de utilidad. Tu tarea es determinar si la respuesta generada realmente responde a "
                    "la pregunta del usuario de manera útil. Retorna estrictamente un objeto JSON con la clave 'useful' "
                    "y valor 'yes' (si responde de manera útil) o 'no' (si no responde adecuadamente)."
                )),
                ("human", "Pregunta:\n{question}\n\nRespuesta:\n{answer}")
            ])
            chain_utility = prompt_utility | llm
            try:
                res_u = await chain_utility.ainvoke({"question": query, "answer": answer})
                content = res_u.content.strip()
                if content.startswith("```"):
                    if content.startswith("```json"):
                        content = content[7:]
                    else:
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                data_u = json.loads(content)
                useful = data_u.get("useful", "").lower() == "yes"
            except Exception as e:
                logger.warning(f"Error evaluando utilidad: {e}")
                useful = True
                
            if not useful:
                logger.warning("[Grafo/Auto-Corrección] Utility check fallido. Activando re-enrutamiento.")
                return "refine"
                
            return "end"

        async def route_after_generation(state: GraphState) -> str:
            decision = await check_groundedness_and_utility(state)
            if decision == "refine":
                return "refine_query"
            return END
            
        workflow.add_conditional_edges(
            "generate_answer",
            route_after_generation,
            {
                "refine_query": "refine_query",
                "__end__": END
            }
        )
        
        workflow.add_edge("refine_query", "expand_query")
        
        logger.info("Compilando StateGraph de LangGraph...")
        return workflow.compile(checkpointer=MemorySaver())

    async def ask(self, question: str, session_id: str = "default_session") -> Dict[str, Any]:
        """Flujo Core: Ejecuta de forma asíncrona el StateGraph compilado."""
        logger.info(f"Agente RAG - Procesando query con LangGraph: '{question}' (Sesión: {session_id})")
        start_time = time.time()
        
        initial_state: GraphState = {
            "original_query": question,
            "expanded_queries": [],
            "retrieved_documents": [],
            "relevance_scores": [],
            "loop_count": 0,
            "final_answer": "",
            "session_id": session_id,
            "sources": []
        }
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            result_state = await self.graph.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error(f"Fallo crítico en ejecución del Grafo: {e}")
            raise LLMGenerationError(f"Fallo crítico en ejecución de StateGraph: {e}") from e
            
        end_time = time.time()
        gen_time = end_time - start_time
        logger.info(f"Tiempo total de orquestación en Grafo: {gen_time:.4f} segundos")
        
        return {
            "respuesta": result_state.get("final_answer", ""),
            "fuentes": result_state.get("sources", []),
            "tiempo_procesamiento_s": round(gen_time, 4)
        }
