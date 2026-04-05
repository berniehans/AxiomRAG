import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
import time
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import LocalFileStore
from langchain_core.stores import BaseStore

from src.config import settings
from src.utils.logging_config import setup_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)

# Cache global para Singleton del Reranker
_SHARED_RERANKER_MODEL = None

class TimedCrossEncoderReranker(CrossEncoderReranker):
    """Compresor custom que propaga relevance_score estricto y mide tiempo exacto."""
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        
        if not documents:
            return []
            
        # Garantía de puente CPU a GPU para inferencia rápida
        if hasattr(self.model, "model") and torch.cuda.is_available():
            self.model.model.to("cuda")
            
        texts = [[query, doc.page_content] for doc in documents]
        
        # Procesamiento secuencial explícito (batch_size=1) para proteger la VRAM
        scores = []
        for txt in texts:
            scores.append(self.model.score([txt])[0])
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        docs_with_scores = list(zip(documents, scores))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = docs_with_scores[:self.top_n]
        
        final_docs = []
        for doc, score in top_docs:
            new_metadata = doc.metadata.copy()
            # Forzamos conversión a float puro (para evitar anomalías 0.000 generadas por tensores o floats de numpy)
            new_metadata["relevance_score"] = float(score)
            final_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
            
        end = time.time()
        logger.info(f"Tiempo de fase de Reranking exacto: {end - start:.4f} segundos")
        return final_docs

class DocumentJsonFileStore(BaseStore):
    """File store con serialización personalizada compatible con objetos Document de LangChain."""
    
    def __init__(self, root_path: str, **kwargs):
        os.makedirs(root_path, exist_ok=True)
        self.store = LocalFileStore(root_path)

    def mset(self, key_value_pairs):
        encoded_pairs = []
        for key, doc in key_value_pairs:
            dict_doc = {"page_content": doc.page_content, "metadata": doc.metadata}
            encoded_doc = json.dumps(dict_doc).encode("utf-8")
            encoded_pairs.append((key, encoded_doc))
        self.store.mset(encoded_pairs)

    def mget(self, keys):
        encoded_docs = self.store.mget(keys)
        docs = []
        for val in encoded_docs:
            if val is None:
                docs.append(None)
            else:
                data = json.loads(val.decode("utf-8"))
                docs.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
        return docs

    def mdelete(self, keys):
        self.store.mdelete(keys)
        
    def yield_keys(self, *args, **kwargs):
        return iter([])

class AdvancedRetriever:
    """Ingeniería de Recuperación (BM25 + Semantic + Reranker + ParentDocs)."""
    
    def __init__(self, vector_store, chunker):
        self.vector_store = vector_store
        
        # 1. Parent-Document Storage (Hito 2.3)
        os.makedirs(settings.LOCAL_STORE_PATH, exist_ok=True)
        self.docstore = DocumentJsonFileStore(settings.LOCAL_STORE_PATH)

        logger.info(f"Inicializando jerarquía Parent/Child usando caché local ({settings.LOCAL_STORE_PATH})")
        # ParentDocumentRetriever requiere un TextSplitter nativo (SemanticChunker es experimental)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=child_splitter, # Los fragmentos volarán a Qdrant
        )
        
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.pipeline_final = None
        self._init_compressor()

    def _init_compressor(self):
        """Preconstruye la capa de compresión BGE Reranker (Hito 2.2) con Singleton."""
        global _SHARED_RERANKER_MODEL
        import torch
        logger.info(f"Cargando Reranker Cross-Encoder ({settings.RERANKER_MODEL_NAME}) en VRAM...")
        try:
            if _SHARED_RERANKER_MODEL is None:
                if not torch.cuda.is_available():
                    logger.warning("[ALERTA CRÍTICA] Reranker inicializando en CPU por falla de acelerador NVIDIA.")
                    device = "cpu"
                else:
                    device = "cuda"
                    
                _kwargs = {"device": device}
                _SHARED_RERANKER_MODEL = HuggingFaceCrossEncoder(
                    model_name=settings.RERANKER_MODEL_NAME, 
                    model_kwargs=_kwargs
                )
                logger.info("Reranker instanciado y cacheado globalmente (Singleton).")
            else:
                logger.info("Usando instancia Singleton cacheada del Reranker.")
                
            self.cross_encoder = _SHARED_RERANKER_MODEL
            # Reranker evaluará los matches y retornará estrictamente el Top 3 propagando scores
            self.reranker = TimedCrossEncoderReranker(model=self.cross_encoder, top_n=3)
        except Exception as e:
            logger.error(f"Error cargando Reranker: {e}")
            raise e

    def build_and_index(self, docs_padre: List[Document], semantic_chunks: List[Document] = None):
        """Procesa e ingiere nuevos flujos completos."""
        logger.info(f"Fase de Indexación [Iniciada]: Validando volcados para {len(docs_padre)} Macro-Documentos Padre...")
        os.makedirs(settings.LOCAL_STORE_PATH, exist_ok=True)
        
        if semantic_chunks is not None:
             import uuid
             id_key = self.parent_retriever.id_key if hasattr(self.parent_retriever, "id_key") else "doc_id"
             
             macro_content = "\n".join([d.page_content for d in docs_padre])
             macro_parent = Document(page_content=macro_content, metadata=docs_padre[0].metadata.copy() if docs_padre else {})
             p_id = str(uuid.uuid4())
             
             enriched_chunks = []
             for chunk in semantic_chunks:
                 new_metadata = chunk.metadata.copy()
                 new_metadata[id_key] = p_id
                 for k, v in macro_parent.metadata.items():
                     if k not in new_metadata:
                         new_metadata[k] = v
                 enriched_chunks.append(Document(page_content=chunk.page_content, metadata=new_metadata))
                 
             self.docstore.mset([(p_id, macro_parent)])
             self.vector_store.add_documents(enriched_chunks)
             logger.info(f"Indexación Semántica: {len(enriched_chunks)} Semantic Chunks guardados apuntando al Parent-Doc ({p_id}).")
        else:
             self.parent_retriever.add_documents(docs_padre)
             logger.info(f"Fase de Indexación: {len(docs_padre)} padres procesados vía Splitter estándar.")
        
        # Construir índice BM25 Léxico puro con TODOS los padres disponibles
        self.update_bm25_en_caliente()

    def update_bm25_en_caliente(self):
        """Actualiza el índice BM25 leyendo todos los documentos del docstore local y reconstruye el pipeline."""
        if not os.path.exists(settings.LOCAL_STORE_PATH):
            return
            
        keys = os.listdir(settings.LOCAL_STORE_PATH)
        if keys:
            all_docs = self.docstore.mget(keys)
            valid_docs = [d for d in all_docs if d is not None]
            if valid_docs:
                self.bm25_retriever = BM25Retriever.from_documents(valid_docs)
                self.bm25_retriever.k = 8
                logger.info(f"Índice Léxico BM25 actualizado internamente con {len(valid_docs)} documentos totales.")
                self._sync_pipeline()

    def _sync_pipeline(self):
        """Acopla las tuberías en un ensamble unificado."""
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.parent_retriever, self.bm25_retriever],
                weights=[0.5, 0.5] # 50% Semántica vectorial, 50% Léxico Exacto
            )
        else:
            self.ensemble_retriever = self.parent_retriever
            
        self.pipeline_final = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.ensemble_retriever
        )
        logger.info("Pipeline de Búsqueda Avanzada Creado y Lista.")

    def search(self, query: str) -> List[Document]:
        """Ejecuta Retrieval con mitigación de ruido cruzando atención del Reranker."""
        start = time.time()
        logger.info(f"Ejecutando Búsqueda Optimizada (Reranking sobre Hijos) para: '{query}'")
        
        # 1. Recuperar Chunks Hijos desde Vector Store (k=20)
        # Esto reduce drásticamente los datos leídos de disco/VRAM
        child_docs = self.vector_store.similarity_search(query, k=20)
        
        # 2. Reranking estricto sobre los Chunks Hijos
        self.reranker.top_n = 5
        reranked_children = self.reranker.compress_documents(child_docs, query)
        
        # 3. Ensamblaje y Mapeo: Trazamos al padre de los mejores sub-docs
        id_key = self.parent_retriever.id_key if hasattr(self.parent_retriever, "id_key") else "doc_id"
        parent_ids = []
        best_scores = {}
        for doc in reranked_children:
            p_id = doc.metadata.get(id_key)
            if p_id:
                if p_id not in parent_ids:
                    parent_ids.append(p_id)
                if p_id not in best_scores:
                    best_scores[p_id] = doc.metadata.get("relevance_score", 0.0)
                
        final_docs = []
        if parent_ids:
            fetched = self.docstore.mget(parent_ids)
            for p_id, doc in zip(parent_ids, fetched):
                if doc is not None:
                    # Propagamos el score del hijo ganador al documento padre original e intacto
                    doc.metadata["relevance_score"] = best_scores.get(p_id, 0.0)
                    final_docs.append(doc)
                    
        # Optional: Añadir resultados de BM25 de forma aditiva si se desea ensemble léxico.
        logger.info(f"Búsqueda finalizada en {time.time()-start:.2f}s. {len(final_docs)} documentos indexados listos.")
        return final_docs
