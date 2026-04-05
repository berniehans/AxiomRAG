from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    pregunta: str = Field(..., description="Pregunta del usuario a resolver usando datos de la organización.")
    session_id: str = Field(..., description="Identificador único del cliente/sesión para memoria (History).")

class Fuente(BaseModel):
    origen: str = Field(..., description="Nombre del archivo o subsistema emisor.")
    categoria: str = Field(..., description="Tipo de documento determinado en Ingesta.")
    score: float = Field(0.0, description="Score de relevancia retornado por Reranker.")

class ChatResponse(BaseModel):
    respuesta: str = Field(..., description="Respuesta final validada y referenciada producida por el llm.")
    fuentes: List[Fuente] = Field(..., description="Documentos del top-3 aprobados post Reranker y utilizados como contexto.")
    tiempo_procesamiento_s: float = Field(..., description="Monitorización MLOps: Segundos dedicados en Fase RAG y Generación LLM.")
