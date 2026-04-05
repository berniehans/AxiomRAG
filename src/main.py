import os
import shutil
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager

from src.config import settings
from src.utils.logging_config import setup_logger
from src.ingestion.embeddings import EmbeddingManager
from src.retrieval.advanced_retrieval import AdvancedRetriever
from src.agent.rag_chain import RAGAgent
from src.api.schemas import QueryRequest, ChatResponse
from src.exceptions import IngestionError
from fastapi.responses import JSONResponse

logger = setup_logger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan de Hardware: Carga memoria VRAM una única vez a nivel de red."""
    logger.info("⚡ [Lifespan FastAPI] Despertando MLOps Backend y precargando Tensor-Flow en GPU...")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ [HARDWARE] Limpieza inicial de VRAM (torch.cuda.empty_cache()) superada.")
        
    try:
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import Distance, VectorParams
        
        embed_manager = EmbeddingManager()
        embeddings = embed_manager.get_embeddings()
        
        # Para el ciclo real usar VectorStoreManager o conectar a localhost:6333
        # Con fines locales preparamos un client robusto a data temporal/local si falla:
        try:
             client = QdrantClient(path=settings.QDRANT_PATH)
        except Exception:
             client = QdrantClient(":memory:")
             
        # Si la colección no existe, inicialízala (BGE-M3 = 1024 vector size)
        if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
             client.create_collection(
                 collection_name=settings.QDRANT_COLLECTION_NAME,
                 vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
             )

        store = QdrantVectorStore(
             client=client,
             collection_name=settings.QDRANT_COLLECTION_NAME,
             embedding=embeddings,
        )
        
        retriever = AdvancedRetriever(vector_store=store, chunker=None)
        agent = RAGAgent(retriever=retriever)
        
        app_state["agent"] = agent
        
        # HW Logging for Verification
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[HARDWARE] {gpu_name} detectada. VRAM total disponible: {vram_gb:.2f} GB asignada para BGE-M3 y Reranker.")
        
        logger.info("✅ VRAM & Retriever Híbrido Listos. Server escuchando.")
    except Exception as e:
        logger.error(f"Fallo crítico en alocación de tensores en inicio: {e}")
        raise
        
    yield  # La API recibe llamadas aquí
    
    logger.info("🛑 [Lifespan API] Cierre de Sesión. Vaciando Tensor Cache...")
    app_state.clear()
    import gc
    gc.collect()
    try:
       import torch
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           logger.info("✅ [HARDWARE] torch.cuda.empty_cache() invocado: Fragmentación prevenida en la VRAM de la RTX.")
    except Exception:
       pass

app = FastAPI(
    title="🏢 Enterprise RAG MLOps",
    description="Backend oficial con Búsqueda Híbrida y Reranking en VRAM.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest) -> ChatResponse:
    """Interfaz Conversacional: Soporta Memoria a través de `session_id`."""
    agent = app_state.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Modelo deshabilitado / Falla de VRAM.")
    
    try:
        resultado = agent.ask(question=request.pregunta, session_id=request.session_id)
        
        return ChatResponse(
            respuesta=resultado["respuesta"],
            fuentes=resultado.get("fuentes", []),
            tiempo_procesamiento_s=resultado.get("tiempo_procesamiento_s", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error HTTP 500 en endpoint conversacional (/chat): {e}")
        raise HTTPException(status_code=500, detail="Fallo catastrófico en Pipeline LLM.")

@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    categoria: str = Form(None)
) -> dict:
    """Inyección en segundo plano de PDF/XLSX pasando por Data Wrangling."""
    if not file.filename.endswith((".pdf", ".xlsx", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF, XLSX and DOCX supported.")
        
    os.makedirs(settings.DATA_PATH, exist_ok=True)
    temp_path = os.path.join(settings.DATA_PATH, file.filename)
    
    with open(temp_path, "wb+") as f:
        shutil.copyfileobj(file.file, f)
        
    logger.info(f"📋 Documento recibido en endpoint: {temp_path}")
    try:
        from scripts.run_ingestion import run_ingestion
        
        agent = app_state.get("agent")
        vector_store = None
        if agent and hasattr(agent, "retriever") and hasattr(agent.retriever, "vector_store"):
            vector_store = agent.retriever.vector_store
            
        def process_in_background():
            try:
                run_ingestion(temp_path, categoria, existing_vector_store=vector_store)
                
                # Gestión de Sesiones: Actualiza en caliente el pipeline (BM25 + Ensemble) en VRAM
                if agent and hasattr(agent.retriever, "update_bm25_en_caliente"):
                    logger.info("⚡ [Hot-Reload] Actualizando Retriever en memoria para nueva sesión de preguntas...")
                    agent.retriever.update_bm25_en_caliente()
            except Exception as e:
                logger.error(f"Fallo en ingesta asíncrona: {e}")

        # Añadida la tarea en background
        background_tasks.add_task(process_in_background)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"status": "processing", "message": "El documento se está vectorizando en segundo plano."}
