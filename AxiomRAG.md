# Blueprint: Sistema RAG Autónomo de Grado Industrial
**Rol Objetivo:** Senior AI Engineer / ML Ops
**Estado:** Fases 1, 2 y 3 Finalizadas | Fase 4 y 5 en Desarrollo (Optimización de Baseline)

## 🎯 Visión General
Desarrollo de un motor de **Generación Aumentada por Recuperación (RAG)** autónomo y de alta fidelidad. El sistema está diseñado para procesar documentación técnica y administrativa compleja con un enfoque prioritario en la **privacidad de datos**, **aceleración por hardware local (NVIDIA RTX 3060)** y **evaluación científica cuantitativa**. Se diferencia de los prototipos convencionales por su arquitectura asíncrona y su capacidad de "Caja Blanca" para auditar cada etapa del proceso de recuperación.

## 🛠️ Stack Tecnológico Consolidado
* **Runtime:** Python 3.12 (Gestionado con `uv` para entornos deterministas).
* **Aceleración:** CUDA 12.4 / PyTorch 2.4+ (Optimizado para arquitectura Ampere).
* **Modelos Locales:** BGE-M3 (Embeddings) y BGE-Reranker-v2-m3 (Cross-Encoders).
* **Orquestación:** LangChain (Patrones avanzados: Parent-Child & Hybrid Search).
* **Base de Datos:** Qdrant (Modo persistente local con indexación HNSW).
* **Evaluación:** Ragas (Baseline actual: 0.60 Faithfulness / 0.66 Context Precision).

---

## 🏗️ Desglose de Hitos Técnicos

### Fase 1: Ingeniería de Ingesta Robusta 
- [x] **Hito 1.1: Chunking Semántico Avanzado**
  - Implementación de `SemanticChunker` basado en gradientes de significado para preservar la cohesión lógica de párrafos técnicos.
- [x] **Hito 1.2: Extracción de Metadatos con Auto-Sanado**
  - Uso de `Structured Output` (Pydantic) para categorizar documentos y generar resúmenes, con lógica de re-intento ante errores de parseo.
- [x] **Hito 1.3: Embeddings de Alta Densidad**
  - Despliegue de `BAAI/bge-m3` en GPU para representación vectorial de 1024 dimensiones sin latencia de API externa.

### Fase 2: Ingeniería de Búsqueda Avanzada
- [x] **Hito 2.1: Ensamble de Búsqueda Híbrida**
  - Integración de Qdrant + BM25 (Léxico). Balance optimizado para capturar términos técnicos únicos (ej. "Retinex", "SSR").
- [x] **Hito 2.2: Pipeline de Reranking (Cross-Encoders)**
  - Implementación secuencial en VRAM para re-ordenar el Top-10 y entregar los 3 fragmentos de "oro" al LLM en < 0.5s.
- [x] **Hito 2.3: Patrón Parent-Child Retrieval**
  - Indexación de fragmentos "hijos" para precisión vectorial con restitución de documentos "padres" completos para contexto del LLM.

### Fase 3: Lógica de Agente y Guardrails
- [x] **Hito 3.1: Gestión de Sesiones Persistentes**
  - Almacenamiento local de historial para mantener coherencia en conversaciones técnicas largas.
- [x] **Hito 3.2: Generación Blindada (RAG Chain)**
  - Prompt Engineering con instrucciones de citación estricta y prohibición de uso de conocimiento general.
- [x] **Hito 3.3: Guardrail de Confianza (Thresholding)**
  - Intercepción de logits del Reranker: Bloqueo de respuestas con score < 0.15 para mitigar alucinaciones de raíz.

### Fase 4: Infraestructura y Alta Disponibilidad (En curso)
- [x] **Hito 4.1: Motor de Ingesta Asíncrono**
  - Uso de `FastAPI BackgroundTasks` para procesar PDFs masivos sin bloquear el ciclo de respuesta HTTP.
- [ ] **Hito 4.2: Dockerización Industrial (NVIDIA Container)**
  - Empaquetamiento multietapa con soporte nativo para `nvidia-container-toolkit` y drivers CUDA.
- [ ] **Hito 4.3: Validación de Contratos (API Security)**
  - Blindaje de endpoints con validaciones Pydantic estrictas para prevenir desbordamientos de VRAM.

### Fase 5: MLOps y Observabilidad Local
- [x] **Hito 5.1: Evaluación Cuantitativa con Ragas**
  - Medición del Baseline inicial y generación de reportes `ragas_eval_metrics.json`.
- [ ] **Hito 5.2: Telemetría y Profiling de Latencia**
  - Implementación de logs estructurados para medir el tiempo exacto por componente (Embedding vs Retrieval vs Gen).
- [ ] **Hito 5.3: Dashboard de Calidad (Streamlit)**
  - Interfaz visual para monitorear la salud del índice y visualizar los chunks almacenados.

### Fase 6: Optimización de Fidelidad (Target: 0.90+)
- [ ] **Hito 6.1: Fine-tuning de Pesos Híbridos**
  - Ajuste dinámico de pesos (Semántica vs Léxica) basado en el análisis de fallos del Golden Dataset.
- [ ] **Hito 6.2: Razonamiento Chain-of-Thought (CoT)**
  - Refinamiento del LLM para forzar un paso de análisis previo a la respuesta técnica final.

---

## 📈 Impacto Profesional (Criterios de Auditoría)
Este repositorio demuestra maestría en:
1.  **Manejo de Hardware:** Configuración avanzada de entornos CUDA 12.4 en Windows/Linux.
2.  **Arquitectura de Datos:** Solución al problema de "Lost in the Middle" mediante Parent-Child Retrieval.
3.  **Cultura MLOps:** No se asume que el sistema funciona; se mide con métricas de fidelidad y precisión.
4.  **Soberanía Digital:** Sistema capaz de operar en entornos aislados (Air-gapped) sin fugas de información a APIs de terceros.

---
*Documentación generada bajo estándares de código limpio y tipado estricto (PEP 484).*