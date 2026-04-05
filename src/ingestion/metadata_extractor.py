from typing import Optional, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class DocumentMetadata(BaseModel):
    """Esquema para enriquecimiento de metadatos de documentos administrativos."""
    origen: Optional[str] = Field(default="Desconocido", description="Departamento o sector que emite el documento, ej: RRHH, Finanzas, Legal, Dirección")
    fecha_emision: Optional[str] = Field(default="Desconocida", description="Fecha en la que fue emitido el documento (Formato YYYY-MM-DD o aproximado)")
    categoria: str = Field(default="General", description="Tipo de documento, ej: Contrato, Manual, Factura, Recibo, Acta")
    resumen: str = Field(default="Sin resumen", description="Resumen breve, de 1 o 2 oraciones, sobre de qué trata el fragmento de texto.")

class MetadataExtractor:
    """
    Analiza un fragmento (chunk) de texto y extrae metadatos estructurados 
    utilizando un LLM para etiquetar y asegurar trazabilidad.
    """
    
    def __init__(self, llm: BaseChatModel):
        if llm is None:
            raise ValueError("Se debe proveer un modelo de lenguaje (LLM).")
            
        # Inyectar propiedades MLOps locales (sobrescribiendo los defaults si existieran)
        if hasattr(llm, "max_tokens"):
            llm.max_tokens = 400
        if hasattr(llm, "max_completion_tokens"):
            llm.max_completion_tokens = 400
            
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Responde ÚNICAMENTE con el objeto json. Sin explicaciones, sin preámbulos, sin bloques de código markdown. "
                       "Sé extremadamente breve en el campo 'resumen'. Máximo 20 palabras. Formato estricto: json."),
            ("human", "Extrae metadatos del siguiente texto:\n\n{texto}")
        ])
        
    def extract(self, text: str) -> DocumentMetadata:
        """Extrae la estructura inyectando las instrucciones garantizando JSON y fallando rápido."""
        txt_lower = text.strip().lower()
        
        # Filtro de ruido
        if len(txt_lower) < 100:
            logger.info("Chunk ignorado por filtro de ruido (< 100 caracteres). Evitando llamada al LLM.")
            return DocumentMetadata()
            
        # Filtro legal (evitar gastar tokens en licencias o copyrights inútiles para RAG)
        legal_terms = ["copyright", "permissions", "all rights reserved"]
        if any(term in txt_lower for term in legal_terms):
            logger.info("Chunk descartado por Filtro Legal (Detectado copyright u otros términos limitantes).")
            return DocumentMetadata()
            
        try:
            # with_structured_output delega la restricción al LLM (ahora Pro, en formato JSON Mode explícito)
            chain = self.prompt | self.llm.with_structured_output(DocumentMetadata)
            resultado = chain.invoke({"texto": text})
            
            if resultado is None:
                raise ValueError("La cadena devolvió None de forma sigilosa. JSON inválido.")
                
            if not resultado.categoria or str(resultado.categoria).lower() == "none":
                resultado.categoria = "General"
                
            return resultado
        except Exception as e:
            logger.error(f"Fallo irrecuperable de parsing al generar JSON en gpt-4o-mini. Emitiendo Fallback vacío. Error: {e}")
            return DocumentMetadata()
