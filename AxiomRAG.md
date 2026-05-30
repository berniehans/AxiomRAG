# Blueprint: Sistema RAG Autónomo de Grado Industrial
**Rol Objetivo:** Senior AI Engineer / ML Ops
**Estado:** Fases 1 a 5 Finalizadas | Fase 6 en Monitoreo Continuo y Escalabilidad

## 🎯 Visión General
Desarrollo de un motor de **Generación Aumentada por Recuperación (RAG)** autónomo y de alta fidelidad. El sistema está diseñado para procesar documentación técnica y administrativa compleja con un enfoque prioritario en la **privacidad de datos**, **aceleración por hardware local (NVIDIA RTX 3060)** y **evaluación científica cuantitativa**. Se diferencia de los prototipos convencionales por su arquitectura asíncrona, capacidad de "Caja Blanca" para auditar cada etapa de recuperación, inferencia por lotes en GPU (CUDA) e ingesta incremental con almacenamiento en caché inteligente.

## 🛠️ Stack Tecnológico Consolidado
* **Runtime:** Python 3.12 (Gestionado con `uv` para entornos deterministas).
* **Aceleración:** CUDA 12.4 / PyTorch 2.4+ (Optimizado para arquitectura Ampere con batching dinámico del Reranker).
* **Modelos Locales:** BGE-M3 (Embeddings) y BGE-Reranker-v2-m3 (Cross-Encoders).
* **Orquestación:** LangChain (Patrones avanzados: Parent-Child, Hybrid Search con BM25 incremental indexado en caliente).
* **Base de Datos:** Qdrant (Modo persistente local blindado contra fallas silenciosas en producción con fallback controlado en desarrollo).
* **Evaluación:** Ragas (Baseline actual: 0.60 Faithfulness / 0.66 Context Precision / Suite completa de 13 pruebas automatizadas passing en sub-minuto).

---

## 🏗️ Desglose de Hitos Técnicos y Tareas Activas

> 👉 **Los hitos operacionales de las fases 1 a 6, incluyendo las tareas pendientes (backlog local) y los logros alcanzados, han sido migrados estructuralmente al tablero Kanban en [tasks.md](./tasks.md).**

---

## 📈 Impacto Profesional (Criterios de Auditoría)
Este repositorio demuestra maestría en:
1.  **Manejo de Hardware:** Configuración avanzada de entornos CUDA 12.4 en Windows/Linux.
2.  **Arquitectura de Datos:** Solución al problema de "Lost in the Middle" mediante Parent-Child Retrieval.
3.  **Cultura MLOps:** No se asume que el sistema funciona; se mide con métricas de fidelidad y precisión.
4.  **Soberanía Digital:** Sistema capaz de operar en entornos aislados (Air-gapped) sin fugas de información a APIs de terceros.

---
*Documentación generada bajo estándares de código limpio y tipado estricto (PEP 484).*