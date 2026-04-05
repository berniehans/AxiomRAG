class RAGBaseError(Exception):
    """Excepción base para el proyecto RAG."""
    pass

class IngestionError(RAGBaseError):
    """Excepción lanzada cuando ocurre un error procesando o cargando documentos."""
    pass

class ModelLoadError(RAGBaseError):
    """Excepción lanzada cuando hay un fallo cargando un modelo (Local o API)."""
    pass

class ConfigurationError(RAGBaseError):
    """Excepción lanzada cuando faltan variables de configuración claves."""
    pass

class LLMGenerationError(RAGBaseError):
    """Excepción lanzada cuando hay un fallo en la generación con el modelo de lenguaje."""
    pass
