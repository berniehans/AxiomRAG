from typing import List, Any
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class DocumentChunker:
    """
    Aplica Semantic Chunking para dividir el documento preservando el significado.
    """
    
    def __init__(self, embeddings_model: Any):
        """
        Requiere la inyección del modelo de embeddings
        """
        if embeddings_model is None:
            raise ValueError("Se debe proveer un modelo de embeddings para particionar.")
            
        self.text_splitter = SemanticChunker(
            embeddings_model, 
            breakpoint_threshold_type="gradient",
            buffer_size=1
        )
        
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Divide documentos enteros en chunks semánticos."""
        logger.info(f"Dividiendo {len(docs)} documentos usando Semantic Chunking...")
        try:
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Se generaron {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error en chunking: {str(e)}")
            raise e
