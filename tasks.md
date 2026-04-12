# 📋 Tablero de Tareas y Roadmap (AxiomRAG)

Este documento centraliza el estado del desarrollo, hitos de ingeniería operativa (MLOps) y las tareas activas para la escalabilidad del motor RAG.

## 🚀 ROADMAP ACTUAL

A continuación, se detalla el backlog técnico de tareas en curso o planificadas basadas en nuestras Fases de desarrollo pendientes.

### 🟡 En Progreso (In Progress)
- [ ] **Dockerización Industrial (NVIDIA Container)**
  - **Objetivo:** Empaquetar un entorno multi-stage puro.
  - **Requisito:** Debe soportar la inyección nativa del driver mediante `nvidia-container-toolkit` y enlazar correctamente CUDA 12.4 para acelerar las inferencias BGE locales.
- [ ] **Validación de Contratos (API Security)**
  - **Objetivo:** Auditar y blindar los endpoints expuestos en el backend (FastAPI/Pydantic) evitando desbordamientos de VRAM con validación estricta de payloads entrantes.

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
- [x] Ensamble estructural de búsqueda híbrida y Reranker local.
- [x] Motor de Ingesta Semántica asíncrono implementando `FastAPI BackgroundTasks`.
- [x] Guardrails lógicos de seguridad pre-Generación (Cutoff restrictivo en Reranker < 0.15).
- [x] Validación Cuantitativa base sobre MLOps generando reportes de precisión via Ragas.
- [x] Consolidación de unificado del archivo `agents.md` dictando flujos y normativas de desarrollo.
- [x] Estructura desacoplada y clean architecture bajo tipado Python 3.12 y PEP 484.

---
_Cualquier adición, refactorización tecnológica o issue de investigación debe volcarse en este documento y transicionar en el tablero siguiendo el flujo MLOps establecido en `agents.md`._
