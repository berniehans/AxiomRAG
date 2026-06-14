import time
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict, cast, Type, Annotated
import operator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.llm_factory import get_llm
from src.utils.logging_config import setup_logger
from src.config import settings
from src.exceptions import LLMGenerationError
import openai

logger = setup_logger(__name__)

# ==========================================
# 1. ESQUEMAS DE CONTROL (STRUCTURED OUTPUT)
# ==========================================

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(
        description="Exactly 3 technical search query variations, synonyms, or acronyms of the original query."
    )

class DocumentRelevance(BaseModel):
    relevance: str = Field(
        description="Determine if the document is relevant to the user query. Must be 'yes' or 'no'."
    )

class HallucinationAudit(BaseModel):
    grounded: str = Field(
        description="Determine if the response is completely grounded in the provided context. Must be 'yes' or 'no'."
    )

class UtilityEvaluation(BaseModel):
    useful: str = Field(
        description="Determine if the response directly and usefully answers the question. Must be 'yes' or 'no'."
    )

class RefinedQuery(BaseModel):
    refined_query: str = Field(
        description="Optimized and rewritten user query to maximize vector store matching."
    )

# ==========================================
# 2. DEFINICIÓN DEL ESTADO CON REDUCERS
# ==========================================

class OverwriteList(list):
    """Subclase de list para indicar al reducer que debe sobrescribir el estado en lugar de fusionar."""
    pass

def merge_documents(old_docs: List[Document], new_docs: List[Document]) -> List[Document]:
    """Reducer senior para evitar duplicados en el estado manteniendo el orden."""
    if isinstance(new_docs, OverwriteList):
        return list(new_docs)
    seen = set(d.page_content.strip() for d in old_docs)
    merged = list(old_docs)
    for d in new_docs:
        content = d.page_content.strip()
        if content not in seen:
            seen.add(content)
            merged.append(d)
    return merged

class GraphState(TypedDict):
    original_query: str
    expanded_queries: List[str]
    # Usamos reducers para no machacar el estado en ciclos de auto-corrección
    retrieved_documents: Annotated[List[Document], merge_documents]
    loop_count: Annotated[int, operator.add] 
    final_answer: str
    session_id: str
    sources: List[Dict[str, Any]]
    # Integramos la memoria conversacional dentro del ciclo de vida nativo de LangGraph
    messages: Annotated[List[BaseMessage], operator.add]

# ==========================================
# 3. NODOS AISLADOS (TESTEABLES / DECOUPLED)
# ==========================================

async def expand_query_node(state: GraphState) -> Dict[str, Any]:
    query = state["original_query"]
    logger.info(f"[Grafo] Expandiendo consulta original: '{query}'")
    
    llm = ChatOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
        model=settings.OPENROUTER_FAST_MODEL,
        temperature=0.0,
        extra_body={"reasoning_effort": "low"}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en optimización de búsquedas técnicas. Genera exactamente 3 variaciones precisas."),
        ("human", "Consulta original: {query}")
    ])
    
    chain = prompt | llm.with_structured_output(ExpandedQueries)
    try:
        res = await chain.ainvoke({"query": query})
        return {"expanded_queries": res.queries}
    except Exception as e:
        logger.error(f"[Grafo] Error al expandir consulta: {e}")
        return {"expanded_queries": []}


async def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    query = state["original_query"]
    docs = state.get("retrieved_documents", [])
    if not docs:
        return {"retrieved_documents": []}
        
    logger.info(f"[Grafo] Evaluando relevancia de {len(docs)} documentos en paralelo...")
    
    # Reutilizamos la misma instancia del LLM configurada para JSON/Structured Output
    llm = ChatOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
        model=settings.OPENROUTER_FAST_MODEL,
        temperature=0.0
    ).with_structured_output(DocumentRelevance)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Determina si el documento es relevante para responder la consulta. Responde JSON con 'relevance': 'yes' o 'no'."),
        ("human", "Consulta: {query}\n\nDocumento:\n{doc_content}")
    ])
    chain = prompt | llm

    async def grade_doc(doc: Document) -> Optional[Document]:
        try:
            res = await chain.ainvoke({"query": query, "doc_content": doc.page_content})
            if res.relevance.lower() == "yes":
                return doc
        except Exception as e:
            logger.warning(f"Error evaluando relevancia (fallback acepta doc): {e}")
            return doc
        return None
        
    # Concurrencia real estructurada
    tasks = [grade_doc(d) for d in docs]
    results = await asyncio.gather(*tasks)
    filtered = [d for d in results if d is not None]
    
    # Hack Senior: Limpiamos la lista vieja forzando un estado limpio mediante el return
    return {"retrieved_documents": OverwriteList(filtered)}


async def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    query = state["original_query"]
    docs = state.get("retrieved_documents", [])
    
    if not docs:
        return {"final_answer": "No tengo suficiente información para responder.", "sources": []}
        
    logger.info(f"[Grafo] Generando respuesta final con LLM estructurado de razonamiento alto...")
    
    sys_prompt = """Eres un asistente corporativo experto. Fundamenta tus respuestas EXCLUSIVAMENTE en el contexto recuperado.
    PROHIBIDO alucinar. Incluye al final el formato estricto: [Fuente: <valor_origen> | Categoría: <valor_categoria>].
    
    Contexto:
    {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}")
    ])
    
    context_str = "\n\n".join([f"[origen: {d.metadata.get('origen', 'Local')} | categoria: {d.metadata.get('categoria', 'General')}]\n{d.page_content}" for d in docs])
    
    llm = ChatOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
        model=settings.OPENROUTER_DEFAULT_MODEL,
        temperature=0.0,
        extra_body={"reasoning_effort": "high"}
    )
    chain = prompt | llm
    
    try:
        response_msg = await chain.ainvoke({
            "question": query,
            "context": context_str,
            "messages": state["messages"]
        })
        answer_text = response_msg.content
    except Exception as e:
        logger.error(f"Error en generación LLM: {e}")
        raise LLMGenerationError(f"Fallo en nodo de generación: {e}")
        
    src_map = [
        {
            "origen": d.metadata.get('origen', 'Local'),
            "categoria": d.metadata.get('categoria', 'General'),
            "score": float(d.metadata.get('relevance_score', 0.0))
        }
        for d in docs
    ]
    
    return {
        "final_answer": answer_text,
        "sources": src_map,
        # Guardamos la interacción de forma nativa en la línea de tiempo de LangGraph
        "messages": [HumanMessage(content=query), AIMessage(content=answer_text)]
    }


async def refine_query_node(state: GraphState) -> Dict[str, Any]:
    query = state["original_query"]
    logger.info(f"[Grafo] Auto-Corrección: Refinando consulta original para romper el bucle...")
    
    llm = ChatOpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
        model=settings.OPENROUTER_FAST_MODEL,
        temperature=0.0
    ).with_structured_output(RefinedQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Reescribe la consulta para maximizar el hit-rate del motor RAG. Devuelve un JSON con 'refined_query'."),
        ("human", "Consulta fallida anterior: {query}")
    ])
    
    try:
        res = await (prompt | llm).ainvoke({"query": query})
        return {"original_query": res.refined_query, "loop_count": 1} # El reducer suma +1 automáticamente
    except Exception as e:
        logger.error(f"Error al refinar: {e}")
        return {"loop_count": 1}

# ==========================================
# 4. ORQUESTADOR Y COMPILACIÓN
# ==========================================

class RAGAgent:
    """Agente RAG Empresarial Asíncrono con Ciclo de Vida y Memoria Centralizada."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(GraphState)
        
        # Inyección de nodos desacoplados
        workflow.add_node("expand_query", expand_query_node)
        workflow.add_node("retrieve_local", self._retrieve_local_node) # Requiere acceso a self.retriever
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("web_search_fallback", self._web_search_fallback_node)
        workflow.add_node("generate_answer", generate_answer_node)
        workflow.add_node("refine_query", refine_query_node)
        
        # Construcción de bordes fijos
        workflow.add_edge(START, "expand_query")
        workflow.add_edge("expand_query", "retrieve_local")
        workflow.add_edge("retrieve_local", "grade_documents")
        
        # Enrutamiento condicional post-evaluación
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {"web_search_fallback": "web_search_fallback", "generate_answer": "generate_answer"}
        )
        workflow.add_edge("web_search_fallback", "generate_answer")
        
        # Enrutamiento condicional post-generación (Self-Correction Loop)
        workflow.add_conditional_edges(
            "generate_answer",
            self._route_after_generation,
            {"refine_query": "refine_query", "__end__": END}
        )
        workflow.add_edge("refine_query", "expand_query")
        
        return workflow.compile(checkpointer=MemorySaver())

    async def _retrieve_local_node(self, state: GraphState) -> Dict[str, Any]:
        all_queries = [state["original_query"]] + state.get("expanded_queries", [])
        tasks = [self.retriever.ainvoke(q) for q in all_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        flat_docs = []
        for res in results:
            if isinstance(res, list):
                flat_docs.extend(res)
        return {"retrieved_documents": flat_docs}

    async def _web_search_fallback_node(self, state: GraphState) -> Dict[str, Any]:
        from duckduckgo_search import DDGS
        query = state["original_query"]
        try:
            # Forzamos ejecución sin bloqueo de hilos
            loop = asyncio.get_running_loop()
            def ddg_sync():
                with DDGS() as ddgs:
                    return " ".join([r.get("body", "") for r in list(ddgs.text(query, max_results=4))])
            
            search_result = await loop.run_in_executor(None, ddg_sync)
            return {"retrieved_documents": [Document(page_content=search_result, metadata={"origen": "Búsqueda Web (DuckDuckGo)", "categoria": "Internet", "relevance_score": 0.5})]}
        except Exception as e:
            return {"retrieved_documents": []}

    def _route_after_grading(self, state: GraphState) -> str:
        return "web_search_fallback" if not state.get("retrieved_documents") else "generate_answer"

    async def _route_after_generation(self, state: GraphState) -> str:
        docs = state.get("retrieved_documents", [])
        answer = state.get("final_answer", "")
        if not docs or not answer or state.get("loop_count", 0) >= 2:
            return END

        # PARALELIZACIÓN DE AUDITORÍA (Groundedness + Utility)
        llm = ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY",
            model=settings.OPENROUTER_FAST_MODEL,
            temperature=0.0
        )
        
        c_grounded = ChatPromptTemplate.from_messages([
            ("system", "Analiza alucinaciones. Responde JSON con 'grounded': 'yes' o 'no'."),
            ("human", "Contexto:\n{context}\n\nRespuesta:\n{answer}")
        ]) | llm.with_structured_output(HallucinationAudit)

        c_utility = ChatPromptTemplate.from_messages([
            ("system", "Evalúa utilidad de la respuesta. Responde JSON con 'useful': 'yes' o 'no'."),
            ("human", "Pregunta:\n{question}\n\nRespuesta:\n{answer}")
        ]) | llm.with_structured_output(UtilityEvaluation)

        context_str = "\n\n".join([d.page_content for d in docs])
        
        # Disparamos ambas evaluaciones en un solo paso asíncrono concurrente
        try:
            g_res, u_res = await asyncio.gather(
                c_grounded.ainvoke({"context": context_str, "answer": answer}),
                c_utility.ainvoke({"question": state["original_query"], "answer": answer}),
                return_exceptions=True
            )
            if (not isinstance(g_res, Exception) and g_res.grounded.lower() == "no") or \
               (not isinstance(u_res, Exception) and u_res.useful.lower() == "no"):
                return "refine_query"
        except Exception as e:
            logger.error(f"Error crítico en validaciones: {e}")
            
        return END

    async def ask(self, question: str, session_id: str = "default_session") -> Dict[str, Any]:
        start_time = time.time()
        config = {"configurable": {"thread_id": session_id}}
        
        initial_state: GraphState = {
            "original_query": question,
            "expanded_queries": [],
            "retrieved_documents": [],
            "loop_count": 0,
            "final_answer": "",
            "session_id": session_id,
            "sources": [],
            "messages": []
        }
        
        result_state = await self.graph.ainvoke(initial_state, config=config)
        gen_time = time.time() - start_time
        
        return {
            "respuesta": result_state.get("final_answer", ""),
            "fuentes": result_state.get("sources", []),
            "tiempo_procesamiento_s": round(gen_time, 4)
        }