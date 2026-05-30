# 🏛️ Arquitectura Enterprise MLOps (RAG Avanzado)

Nuestra tubería RAG (Retrieval-Augmented Generation) se construyó sobre las prácticas comprobadas para reducir alucinaciones empíricas y elevar el techo del "Faithfulness" de las respuestas de IA Generativa.

## 🌊 Diagrama de Flujo del Pipeline (Data to Generation)

```mermaid
graph TD
    A[PDF/XLSX Ingestion] --> B[Extracción Resumen Global Pydantic]
    B --> C[DocumentChunker]
    C --> D[Embeddings BGE-M3]
    D --> E[(Qdrant Vector DB)]
    A --> F[(Parent DocStore JSON)]
    
    U[User Query] --> G[Hybrid Retrieval: 50/50 Ensemble]
    G --> H[Semantic Chunks BGE-M3]
    G --> I[BM25 Lexical Engine]
    H -.-> J[BGE Reranker-v2 CUDA 12.4]
    I -.-> J
    J --> K{Threshold 0.15 Guardrail}
    K -- Pasó la barrera --> L[Mapeo a Documento Padre Original]
    L --> M[RAGAgent OpenAI/Groq]
    K -- Falló barrera --> N[Short-Circuit: Respuesta Defensiva Vacía]
    M --> O[Producción de Respuesta Verificada]
```

## 🧩 Estrategia de Retrieval: Patrón Parent-Child
En lugar de cargar pesados documentos enteros o diminutos trozos de texto inconexos al LLM, hemos partido la tubería empleando heurística *Parent / Child*:

- **Los Hijos (Child Chunks):** Son fragmentos de 400 a 600 tokens con fuerte densidad semántica inyectados directo en Qdrant. Debido a su diminuta agudeza focal, la distancia cosenoidal arroja aciertos certeros en la topología vectorial.
- **Los Padres (Parent Documents):** Una vez que un Child Chunk es marcado como "relevante", no le damos el pequeño corte de información al Agente. En cambio, trazamos su UUID para mapear y devolver de nuevo a la tubería **todo el archivo Padre Original íntegro**, proporcionándole a ChatGPT o Llama un contexto infinito y robusto.

## ⚖️ Búsqueda Híbrida Balanceada (50/50 Ensemble)
Sabemos empíricamente que los Embeddings (búsqueda densa) fallan groseramente ante acrónimos puros o jerga corporativa ultratécnica.
Nuestro motor implementa un pipeline `EnsembleRetriever` ponderado a `0.5` Vectorial y `0.5` Léxico (`BM25`).
- Permite detectar el *sentido abstracto* de la pregunta (Vectores).
- Evita el *"Zero Match"*, siendo letal contra papers altamente densificados con fórmulas ("Retinex", o acrónimos "SSR").

## 🛡️ Guardrails de Seguridad (0.15 Logit Threshold)
Al final de la recuperación, el Cross Encoder ejecuta una regresión que descarta los falsos positivos. Todo tensor que obtenga menos de `0.15` en confianza es **destruido**.
De este modo, evitamos prompts vacíos que degeneran en alucinaciones puras; si no poseemos un contexto verídico recuperado, el bot se cruza de brazos protegiendo la identidad de la App.

## ⚡ Optimizaciones Recientes de Rendimiento y Concurrencia

### 1. Reranker de Lote Dinámico (Dynamic Batching)
En `TimedCrossEncoderReranker.compress_documents`, se implementó procesamiento por lotes para el Cross-Encoder utilizando `settings.RERANKER_BATCH_SIZE`. En lugar de evaluar secuencialmente cada texto con un lote implícito de 1 (lo que destruía el paralelismo de tensores de CUDA), segmentamos el corpus de textos y utilizamos la API predictiva del cliente subyacente (`self.model.client.predict`) para paralelizar la inferencia sin riesgo de desbordar la VRAM en la GPU.

### 2. Ingesta BM25 Segmentada e Incremental
El método `update_bm25_en_caliente` ahora evita la reconstrucción total y síncrona del corpus desde disco (que requería leer y des-serializar todos los documentos almacenados en `LOCAL_STORE_PATH` en cada nueva subida). 
Se diseñó un sistema de caché de documentos en memoria estructurado en un diccionario (`self._cached_docs_dict`), sincronizándose bidireccionalmente con el directorio local (insertando llaves nuevas y removiendo llaves eliminadas). Esto reduce el costo de disco a cero para documentos preexistentes.

### 3. Fortalecimiento de la Persistencia (Qdrant Client Fail-Safe)
En la inicialización del ciclo de vida (`main.py`), se eliminó la degradación silenciosa e inadvertida a base de datos en memoria ante fallos en la base de datos local física de Qdrant. 
- En entornos de producción (`ENV="production"`), la aplicación emite logs de criticidad alta y detiene ruidosamente el inicio del servidor (`sys.exit(1)`) para prevenir la pérdida de datos.
- En entornos de desarrollo, se permite la degradación con una advertencia visual masiva en la consola.
