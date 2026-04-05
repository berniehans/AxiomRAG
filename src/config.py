from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App General
    LLM_PROVIDER: str = "openrouter"
    
    # Paths
    DATA_PATH: str = "data"
    LOGS_PATH: str = "logs"
    QDRANT_PATH: str = "qdrant_db"
    QDRANT_COLLECTION_NAME: str = "rag_documents"
    LOCAL_STORE_PATH: str = "local_doc_store"

    # API Keys
    OPENROUTER_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    
    # Modelo Config
    EMBEDDINGS_MODEL_NAME: str = "BAAI/bge-m3"
    OPENROUTER_DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    
    # Resiliencia LLM
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 1
    
    # Observabilidad MLOps (LangSmith / OpenTelemetry)
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "RAG_Enterprise_Ops"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="allow"
    )

settings = Settings()

import os
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

if settings.LANGCHAIN_TRACING_V2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
