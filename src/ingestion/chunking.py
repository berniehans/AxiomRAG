import re
import numpy as np
from typing import List, Any
from langchain_core.documents import Document
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class CustomSemanticChunker:
    """
    Particionador semántico de alto rendimiento para dividir texto basándose 
    en similitud y cambios de gradiente semántico.
    """
    def __init__(self, embeddings_model: Any, breakpoint_threshold_type: str = "gradient", breakpoint_threshold_amount: float = 95.0):
        self.embeddings_model = embeddings_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def split_text(self, text: str) -> List[str]:
        # 1. Separación de oraciones mediante regex simple
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # 2. Generación de embeddings
        embeddings = self.embeddings_model.embed_documents(sentences)
        
        # 3. Cálculo de distancias cosenoidales consecutivas
        distances = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i+1])
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            distances.append(1.0 - sim)

        if not distances:
            return [" ".join(sentences)]

        # 4. Cálculo de umbrales según tipo
        if self.breakpoint_threshold_type == "gradient":
            if len(distances) > 1:
                gradient = np.abs(np.gradient(distances))
            else:
                gradient = np.array([abs(distances[0])])
            threshold = np.percentile(gradient, self.breakpoint_threshold_amount)
            metric_to_compare = gradient
        elif self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(distances, self.breakpoint_threshold_amount)
            metric_to_compare = np.array(distances)
        else:
            threshold = np.mean(distances) + np.std(distances)
            metric_to_compare = np.array(distances)

        # 5. Agrupamiento de oraciones basadas en el umbral
        chunks = []
        current_chunk_sentences = [sentences[0]]
        for i in range(len(distances)):
            if metric_to_compare[i] > threshold:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])
        chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                chunks_docs.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()
                ))
        return chunks_docs

class DocumentChunker:
    """
    Aplica Semantic Chunking para dividir el documento preservando el significado.
    """
    
    def __init__(self, embeddings_model: Any):
        """
        Requiere la inyección del modelo de embeddings.
        """
        if embeddings_model is None:
            raise ValueError("Se debe proveer un modelo de embeddings para particionar.")
            
        self.text_splitter = CustomSemanticChunker(
            embeddings_model, 
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=95.0
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
