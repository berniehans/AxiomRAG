import torch
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logging_config import setup_logger
from src.exceptions import ModelLoadError
from src.config import settings

logger = setup_logger(__name__)

class EmbeddingManager:
    """
    Gestiona la carga de modelos de embeddings locales (Sentences Transformers).
    Patrón Singleton para eficiencia de memoria usando BGE-M3.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = None) -> None:
        if self._initialized:
            return
        self.model_name = model_name or settings.EMBEDDINGS_MODEL_NAME
        self._embeddings = None
        self._initialized = True

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Carga e inicializa el modelo BGE-M3 solo cuando se requiera (Lazy Loading)."""
        if self._embeddings is not None:
            return self._embeddings

        logger.info(f"Cargando modelo de embeddings local: {self.model_name}...")
        
        # Detectar el mejor dispositivo disponible
        if not torch.cuda.is_available():
            logger.warning("[ALERTA CRÍTICA] CUDA no está disponible. El pipeline MLOps se degradará gravemente usando CPU. Revisa los drivers NVIDIA.")
            device = "cpu"
        else:
            logger.info("[GPU INFO] Utilizando RTX 3060 con CUDA 12.4. Vinculando Embeddings internamente a CUDA.")
            device = "cuda"
            
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    "device": device,
                    # Evitar errores si se requieren pesos remotos extras o custom architectures
                    "trust_remote_code": True 
                },
                encode_kwargs={
                    "normalize_embeddings": True # Importante para búsqueda híbrida / cosenoidal con BGE
                }
            )
            logger.info("Modelo de embeddings cargado correctamente.")
            return self._embeddings
        except Exception as e:
            logger.error(f"Fallo crítico al cargar modelo de embeddings {self.model_name}: {e}")
            raise ModelLoadError(f"Error inicializando modelo local: {e}") from e
