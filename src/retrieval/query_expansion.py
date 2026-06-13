import json
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

async def expand_query_async(query: str, llm: Optional[BaseChatModel] = None) -> List[str]:
    """
    Genera de forma asíncrona variantes optimizadas para la búsqueda de la consulta del usuario.
    Retorna una lista con las variantes de queries expandidas (máximo 3).
    """
    logger.info(f"Query Expansion - Expandiendo consulta: '{query}'")
    if llm is None:
        # Generamos un modelo que retorne JSON estructurado
        llm = get_llm(require_json=True)
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Eres un experto en optimización de búsquedas técnicas. Tu tarea es generar variantes de búsqueda "
            "que ayuden a encontrar la información relevante en una base de datos vectorial y motor léxico. "
            "Retorna estrictamente un objeto JSON con una lista de strings bajo la clave 'queries' conteniendo exactamente "
            "3 variaciones técnicas y precisas de la consulta original. No incluyas explicaciones."
        )),
        ("human", "Consulta original: {query}")
    ])
    
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({"query": query})
        content = response.content.strip()
        # Fallback si el content viene rodeado de markdown json blocks
        if content.startswith("```"):
            if content.startswith("```json"):
                content = content[7:]
            else:
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
        data = json.loads(content)
        queries = data.get("queries", [])
        logger.info(f"Query Expansion - Variantes generadas exitosamente: {queries}")
        return [str(q) for q in queries]
    except Exception as e:
        logger.error(f"Error en Query Expansion (retornando lista vacía): {e}")
        return []
