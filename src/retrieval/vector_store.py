import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import settings
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class VectorStoreManager:
    """Implementación local de Base de Datos Vectorial sin servidor."""
    def __init__(self, embeddings_model, collection_name: str = None):
        self.embeddings = embeddings_model
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.qdrant_path = settings.QDRANT_PATH
        
        os.makedirs(self.qdrant_path, exist_ok=True)
        # Inicialización de Qdrant en Local Disk Mode
        self.client = QdrantClient(path=self.qdrant_path)
        
        from qdrant_client.models import VectorParams, Distance
        if not self.client.collection_exists(collection_name=self.collection_name):
            # BGE-M3 produce vectores estandarizados de 1024 dimensiones
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

        # Conector para LangChain
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        logger.info(f"Conexión estable a Vector Store Qdrant Local ({self.qdrant_path})")
        
    def get_store(self) -> QdrantVectorStore:
        return self.vector_store
