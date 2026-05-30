# 📋 Tablero de Tareas y Roadmap (AxiomRAG)

Este documento centraliza el estado del desarrollo, hitos de ingeniería operativa (MLOps) y las tareas activas para la escalabilidad del motor RAG.

## 🚀 ROADMAP ACTUAL

A continuación, se detalla el backlog técnico de tareas en curso o planificadas basadas en nuestras Fases de desarrollo pendientes.

### 🟡 En Progreso (In Progress)
*(Sin tareas en progreso actual - Esperando asignación del Backlog)*

### 🔴 Pendientes (To Do)
- [ ] **Telemetría y Profiling de Latencia**
  - **Contexto:** Logs estructurados sub-militamétricos para observabilidad.
  - **Acción:** Medir de manera desagregada los tiempos de ejecución para procesos de `Embedding` vs `Retrieval Híbrido` vs `Generation`. Requerido para la trazabilidad MLOps local.
- [ ] **Dashboard de Calidad Visual (Streamlit)**
  - Crear una interfaz web interna y ligera para interactuar con el pipeline, permitiendo a los ingenieros evaluar chuks de información recuperados y testear latencia en tiempo real.
- [ ] **Fine-tuning de Pesos Híbridos**
  - Evaluar a través del *Golden Dataset* el balance actual (50/50 BM25-Vectorial) para ajustarlos experimentalmente minimizando falsos positivos en léxico.
- [ ] **Razonamiento Chain-of-Thought (CoT)**
  - Inyectar promts lógicos ocultos pre-generación para que el modelo construya internamente una reflexión técnica antes de proyectar la respuesta consolidada al usuario, validando su propia inferencia.

## ✅ Hitos Alcanzados (Done)
- [x] **Inferencia por Lote Dinámico en Reranker (CUDA):** Batching dinámico mediante `RERANKER_BATCH_SIZE` delegando inferencia al cliente subyacente de `sentence_transformers`, optimizando el paralelismo de tensores.
- [x] **Ingesta Incremental y Segmentada de BM25:** Diccionario de caché bidireccional en memoria (`self._cached_docs_dict`) para evitar costosas lecturas y deserializaciones de disco en cada indexación.
- [x] **Fail-Safe de Persistencia en Inicialización (Qdrant):** Control preventivo de fallos en base de datos física, con parada inmediata en producción (`ENV="production"`) y alerta visual en desarrollo con fallback controlado a memoria.
- [x] **Dockerización Industrial:** Despliegue empaquetado multi-stage alineado estrictamente a NVIDIA CUDA 12.4.1.
- [x] **API Guardrails (Memoria):** Subidas limitadas asíncronamente a 15MB y prompts restringidos con Pydantic a 600 caracteres para evasión de OOM en RTX 3060.
- [x] Ensamble estructural de búsqueda híbrida y Reranker local.
- [x] Motor de Ingesta Semántica asíncrono implementando `FastAPI BackgroundTasks`.
- [x] Guardrails lógicos de seguridad pre-Generación (Cutoff restrictivo en Reranker < 0.15).
- [x] Validación Cuantitativa base sobre MLOps generando reportes de precisión via Ragas.
- [x] Consolidación de unificado del archivo `agents.md` dictando flujos y normativas de desarrollo.
- [x] Estructura desacoplada y clean architecture bajo tipado Python 3.12 y PEP 484.

---
_Cualquier adición, refactorización tecnológica o issue de investigación debe volcarse en este documento y transicionar en el tablero siguiendo el flujo MLOps establecido en `agents.md`._
