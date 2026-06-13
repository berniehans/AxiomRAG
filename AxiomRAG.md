# Blueprint: Sistema RAG Autónomo de Grado Industrial

## 🎯 Visión General
Desarrollo de un motor de **Generación Aumentada por Recuperación (RAG)** autónomo y de alta fidelidad. El sistema está diseñado para procesar documentación técnica y administrativa compleja con un enfoque prioritario en la **privacidad de datos**, **aceleración por hardware local (NVIDIA RTX 3060)** y **evaluación científica cuantitativa**. Se diferencia de los prototipos convencionales por su arquitectura asíncrona y su capacidad de "Caja Blanca" para auditar cada etapa del proceso de recuperación.

## 🛠️ Stack Tecnológico Consolidado
* **Runtime:** Python 3.12 (Gestionado con `uv` para entornos deterministas).
* **Aceleración:** CUDA 12.4 / PyTorch 2.4+ (Optimizado para arquitectura Ampere).
* **Modelos Locales:** BGE-M3 (Embeddings) y BGE-Reranker-v2-m3 (Cross-Encoders).
* **Orquestación:** LangChain & LangGraph (Patrones avanzados: Parent-Child, Hybrid Search & Self-Corrective RAG).
* **Base de Datos:** Qdrant (Modo persistente local con indexación HNSW).
* **Evaluación:** Ragas (Baseline actual: 0.60 Faithfulness / 0.66 Context Precision).

---

## 📈 Impacto Profesional (Criterios de Auditoría)
Este repositorio demuestra maestría en:
1.  **Manejo de Hardware:** Configuración avanzada de entornos CUDA 12.4 en Windows/Linux.
2.  **Arquitectura de Datos:** Solución al problema de "Lost in the Middle" mediante Parent-Child Retrieval.
3.  **Cultura MLOps:** No se asume que el sistema funciona; se mide con métricas de fidelidad y precisión.
4.  **Soberanía Digital:** Sistema capaz de operar en entornos aislados (Air-gapped) sin fugas de información a APIs de terceros.

---
*Documentación generada bajo estándares de código limpio y tipado estricto (PEP 484).*