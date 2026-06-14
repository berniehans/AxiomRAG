import os
import pypdf
from typing import List
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from src.utils.logging_config import setup_logger
from src.exceptions import IngestionError

logger = setup_logger(__name__)

class MultimodalParser:
    """
    Clase para procesar distintos tipos de archivos administrativos y extraer texto y estructura.
    """
    
    def __init__(self):
        pass

    def parse_pdf(self, file_path: str) -> List[Document]:
        """Extrae texto de un PDF utilizando pypdf nativo de Python para garantizar portabilidad sin Poppler."""
        logger.info(f"Parsing PDF: {file_path}")
        try:
            reader = pypdf.PdfReader(file_path)
            docs = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                ))
            return docs
        except Exception as e:
            logger.error(f"Error parseando PDF {file_path}: {str(e)}")
            raise IngestionError(f"Falla al extraer datos del PDF (Pure Python): {str(e)}") from e

    def parse_excel(self, file_path: str) -> List[Document]:
        """Extrae texto de Excel usando unstructured para preservar filas completas."""
        logger.info(f"Parsing Excel: {file_path}")
        try:
            # Usamos UnstructuredLoader de la nueva integración oficial langchain-unstructured
            loader = UnstructuredLoader(file_path, mode="single", languages=["spa", "eng"])
            docs = loader.load()
            return docs
        except Exception as e:
            logger.error(f"Error parseando Excel {file_path}: {str(e)}")
            raise IngestionError(f"Falla al extraer datos del Excel: {str(e)}") from e
