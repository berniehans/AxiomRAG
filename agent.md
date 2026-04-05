# 🧠 Mapa Cognitivo para IA (Agente)

Este archivo sirve como punto de referencia universal para agentes o sistemas de LLM que requieran interactuar, debugar o escalar este repositorio.

## 📌 Contexto Principal
Este repositorio es un **Motor Avanzado de Retrieval-Augmented Generation (RAG)** enfocado en documentos administrativos y archivos complejos, estructurado bajo un paradigma moderno de ML Ops industrial.

- **Lenguaje Base**: Python 3.12
- **Dependencias Sensibles**: `uv` gestiona un `pyproject.toml` estricto con PyTorch `2.6.0+cu124` y el ecosistema LangChain 1.x.

## 🏗️ Topografía del Sistema
El repositorio está agrupado en directorios lógicos y encapsulados:
1. **`src/retrieval/`**: Lógica core de ingesta e inferencia de búsqueda.
   - `advanced_retrieval.py`: Archivo vital. Orquesta el Ensamble `BM25`, `Qdrant` y `BGE Cross-Encoder Reranker`. Además maneja el paradigma `ParentDocumentRetriever` (LangChain) usando `LocalFileStore` sincronizado a vectores hijos.
2. **`src/agent/`**: Motor Generativo Inteligente.
   - `rag_chain.py`: Empalma el `AdvancedRetriever` con el LLM externo. Gestiona estados `ChatMessageHistory` en RAM, formatea prompts sistemáticos inyectando constraints rígidos (citations), y monitorea umbrales (`confidence_threshold`) para activar "Guardrails" de contención ante scores insuficientes, evadiendo alucinaciones.
3. **`scripts/`**: Puntos de entrada para CLI/Pipelines de MLOps.
   - `run_ingestion.py`: Levanta Pydantic Model Extraction + BGE-M3 local y genera las jerarquías en BD.
   - `test_retrieval.py`: Ejecuta simulaciones de atención cruzada sobre VectorSearch + BM25 Lexical base.
4. **`docs/`**: Base de conocimiento extendida de arquitectura. Detalla flujos, mitigación CVEs de Torch (`gpu_optimization.md`), límites de HW e Ingestión.

## 🧱 Flujos de Mutación y Extensibilidad
Si debes modificar código o expandir el framework, toma estas salvaguardas arquitectónicas:
- **Nuevos Metadatos:** Siempre utiliza herencia fuerte de Pydantic `BaseModel` en las clases de metadata con sus respectivos `Field(description="...")`. El auto-healing depende de the validation error propagation si la LLM alucina un JSON.
- **FS (File Storage):** La clase personalizada `DocumentJsonFileStore` extenderá `LocalFileStore` iterando las estructuras; todo Key-Store físico de padres debe prever una creación automática de path (`os.makedirs`).
- **Límites de CUDA/Hardware:** NUNCA actualices la dependencia de PyTorch ciegamente vía pip. La versión es estrictamente `pytorch-cu124` mapeada al índice oficial dentro del toolchain `uv` en `pyproject.toml`.

> **Mantra del Proyecto:** Fiabilidad, tipado estricto y blindaje anti-alucinación por Reranker local prevalecen frente a dependencias externas inestables. Todo modelo fundacional idealmente residirá localmente en GPU.
