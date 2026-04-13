from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.logging_config import setup_logger
from src.config import settings

logger = setup_logger(__name__)

def get_llm(provider: Optional[str] = None, max_tokens: int = 1000, require_json: bool = False) -> BaseChatModel:
    """
    Fábrica centralizada para instanciar el LLM utilizando patrón factory.
    Evita la creación múltiple del motor y bloqueos de circular import.
    """
    provider = provider or settings.LLM_PROVIDER
    logger.info(f"Instanciando LLM a través de provider (Factory): {provider} | max_tokens={max_tokens} | require_json={require_json}")
    
    model_kwargs: Dict[str, Any] = {}
    if require_json:
        model_kwargs["response_format"] = {"type": "json_object"}
        
    stop_seq: Optional[list[str]] = ["}"] if require_json else None
    
    if provider.lower() == "groq" and settings.GROQ_API_KEY:
        return ChatGroq(
            api_key=settings.GROQ_API_KEY, # type: ignore
            model_name="llama3-8b-8192", 
            temperature=0.0,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
            max_tokens=max_tokens,
            stop=stop_seq,
            model_kwargs=model_kwargs
        )
    else:
        # Por defecto OpenRouter / OpenAI JSON Mode
        return ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY or "DUMMY_KEY", # type: ignore
            model=settings.OPENROUTER_DEFAULT_MODEL,
            temperature=0.0,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
            max_tokens=max_tokens,
            max_completion_tokens=max_tokens,
            stop=stop_seq,
            model_kwargs=model_kwargs
        )
