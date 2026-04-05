import os
import sys

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.ingestion.embeddings import EmbeddingManager
from src.retrieval.advanced_retrieval import AdvancedRetriever
from src.agent.rag_chain import RAGAgent
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

def prefill_test_data(retriever: AdvancedRetriever) -> None:
    """Inyecta un documento padre sobre CX-490 en la memoria local para la simulación."""
    logger.info("Inyectando documento de prueba (CX-490) en el VectorStore...")
    mock_doc = Document(
        page_content="Reporte de Activos 2024: La máquina industrial modelo CX-490 ha sufrido una depreciación acelerada del 15% este trimestre. La razón principal de la venta y liquidación de esta maquinaria fue autorizada por el comité directivo debido a los altos costos de mantenimiento del rotor principal, incurriendo en un desfase de presupuesto.",
        metadata={
            "origen": "Reporte_2024.pdf",
            "categoria": "Finanzas",
            "fecha_emision": "2024-03-01",
            "resumen": "Reporte de depreciación y venta de máquina CX-490"
        }
    )
    # Lanza indexación parent-document al qdrant en memoria
    retriever.build_and_index([mock_doc])

def run_simulation() -> None:
    # 1. Init Dependencias compartidas y Singleton local
    logger.info("Inicializando Pipeline RAG Completo (Singleton garantizado para optimización)...")
    embed_manager = EmbeddingManager()
    embeddings = embed_manager.get_embeddings()
    
    # Motor Qdrant Vectorial Temporal (Memoria pura)
    client = QdrantClient(":memory:")
    # bge-m3 devuelve vectores de 1024 
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        embedding=embeddings,
    )
    
    advanced_retriever = AdvancedRetriever(vector_store=vector_store, chunker=None)
    
    # 2. Ingesta Test
    prefill_test_data(advanced_retriever)
    
    # 3. Instanciar Agente Memorable
    agent = RAGAgent(retriever=advanced_retriever)
    
    SESSION_ID = "sesion_auditoria_cx490"
    
    print("\n" + "="*70)
    print("🤖 INICIANDO SIMULACIÓN DE AGENTE MLOPS (FASE 3)")
    print("="*70)
    
    # PASO 1: Pregunta técnica estricta (Retrieval puro)
    q1 = "¿Qué porcentaje de depreciación sufrió la CX-490 este trimestre?"
    print(f"\n🗣️ [Usuario]: {q1}")
    res1 = agent.ask(q1, session_id=SESSION_ID)
    print(f"🤖 [Agente]: {res1['respuesta']}")
    
    # PASO 2: Pregunta de seguimiento (Consciencia de memoria persistente)
    q2 = "¿Y cuál fue la razón principal de esa venta según indicas?"
    print(f"\n🗣️ [Usuario]: {q2}")
    res2 = agent.ask(q2, session_id=SESSION_ID)
    print(f"🤖 [Agente]: {res2['respuesta']}")
    
    # PASO 3: Pregunta fuera de contexto (Guardrail de umbral MLOps de alucinación)
    q3 = "¿Me podrías decir además cuál es la receta para hacer un buen pastel de chocolate?"
    print(f"\n🗣️ [Usuario]: {q3}")
    res3 = agent.ask(q3, session_id=SESSION_ID)
    print(f"🛑 [Guardrail Agente]: {res3['respuesta']}")
    
    print("\n" + "="*70)
    print("✅ SIMULACIÓN COMPLETADA Y TRAZADA EN LOGS")
    print("="*70)

if __name__ == "__main__":
    run_simulation()
