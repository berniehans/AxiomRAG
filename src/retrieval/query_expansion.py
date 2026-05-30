import asyncio
import json
from typing import List
from pydantic import BaseModel, Field

from src.llm_factory import get_llm
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class ExpandedQueries(BaseModel):
    """
    Modelo Pydantic para el tipado estricto del output del LLM.
    Cumple con el mantra de AxiomRAG (AGENTS.md) de usar tipado fuerte en metadata.
    """
    variaciones: List[str] = Field(
        description="Lista de exactamente 3 variaciones semánticas, sinónimos o acrónimos alternativos de la consulta original."
    )

class AsynchronousQueryExpander:
    """
    Componente asíncrono para la expansión y reescritura de consultas.
    Utiliza LLMs locales o externos mediante llm_factory de forma resiliente.
    """
    def __init__(self, provider: str = None) -> None:
        # Instanciamos el LLM con JSON mode habilitado para asegurar respuesta estructurada limpia
        self.llm = get_llm(provider=provider, max_tokens=150, require_json=True)
        
        self.system_prompt = (
            "Eres un motor experto en optimización de Search Recall y procesamiento de lenguaje natural.\n"
            "Tu tarea es interceptar la consulta original del usuario y generar EXACTAMENTE 3 variaciones semánticas alternativas,\n"
            "extrayendo acrónimos técnicos, terminología formal o sinónimos corporativos pertinentes.\n"
            "Instrucción Estricta: Debes responder ÚNICAMENTE con un objeto JSON que siga el esquema requerido:\n"
            "{\n"
            "  \"variaciones\": [\"consulta alternativa 1\", \"consulta alternativa 2\", \"consulta alternativa 3\"]\n"
            "}\n"
            "No incluyas explicaciones, introducciones ni rodeos."
        )

    async def expand_query(self, query: str) -> List[str]:
        """
        Genera asíncronamente 3 variantes semánticas alternativas utilizando prompts de ingeniería contextuales.
        
        Implementa Graceful Degradation: si el LLM falla, da timeout, o retorna un formato corrompido,
        captura la excepción limpiamente y devuelve una lista vacía, permitiendo que el pipeline principal
        continúe operando únicamente con la consulta original del usuario.
        """
        logger.info(f"Iniciando Query Expansion asíncrona para: '{query}'")
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Consulta original a expandir:\n'{query}'")
            ]
            
            # Invocación asíncrona (ainvoke)
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            if not content or not isinstance(content, str):
                logger.warning("Query Expansion: LLM retornó un contenido vacío o no válido. Aplicando fallback.")
                return []
                
            content = content.strip()
            
            # Parsear el JSON
            data = json.loads(content)
            variations = data.get("variaciones", [])
            
            if isinstance(variations, list):
                # Limpiar y filtrar vacíos
                cleaned = [str(v).strip() for v in variations if str(v).strip()]
                logger.info(f"Query Expansion completada exitosamente. Variantes generadas: {cleaned}")
                return cleaned[:3]
                
            logger.warning("Query Expansion: El formato del JSON recibido no es el esperado. Aplicando fallback.")
            return []
            
        except asyncio.TimeoutError:
            logger.error("Query Expansion: Tiempo de espera (timeout) agotado para el LLM. Aplicando fallback de degradación graciosa.")
            return []
        except Exception as e:
            logger.error(f"Query Expansion: Error inesperado al invocar el LLM ({e}). Aplicando fallback de degradación graciosa.")
            return []
