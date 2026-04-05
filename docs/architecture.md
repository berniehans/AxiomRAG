# 🏛 Documentación de Arquitectura y MLOps (RAG Enterprise)

Este documento justifica detalladamente las decisiones de ingeniería estructurales tomadas para garantizar latencias de sub-segundo en hardware comercial (NVIDIA RTX 3060), manteniendo métricas corporativas de precisión y una mitigación agresiva de costos.

---

## 1. Flujo del Ciclo de Vida del Dato

El sistema RAG ha sido desdoblado en dos conductos de ejecución independientes y asíncronos para evitar cuellos de botella en la inferencia real.

### Fase Offline (Digestión e Ingesta de Datos)
Encargada de transformar información no estructurada en un grafo vectorial altamente indexable:
1. **Extracción Multimodal (`parsers.py`):** Limpia artefactos en archivos binarios complejos (PDF, DOCX, XLSX).
2. **Segmentación Contextual (`chunking.py`):** Implementa un *Semantic Chunking* con solapamiento controlado (Overlap) usando `BAAI/bge-m3`. Se prefiere preservar la cohesión temática de los enunciados antes que depender de un corte rígido de caracteres.
3. **Inyección Estructurada de Metadatos (`metadata_extractor.py`):** Usa `gpt-4o-mini` para rotular el fragmento (Categoria, Origen y Resumen hiperestricto de 20 palabras máximo). 

### Fase Online (Consulta y Generación)
El cerebro en tiempo real que el usuario experimenta en el frontend:
1. **Búsqueda Avanzada Híbrida:** Cruza simultáneamente un algoritmo Dense-Vector (Qdrant Local) y uno Sparse-Lexical (BM25 Auto-Recargable en RAM).
2. **Atención por Cross-Encoder:** El Reranker evalúa qué nodos merecen atención prioritaria.
3. **Decodificación Final (`rag_chain.py`):** Si aprueba los esquemas de seguridad, se extrae el macro-documento y LangChain canaliza el LLM Generativo (Asistente), liberando el *JSON Mode* e inyectando un volumen máximo de 1000 tokens para producir explicaciones matemáticas, técnicas o de programación extensas.

---

## 2. Justificación de Rendimiento y Costos Operativos

Toda la base de esta arquitectura gravita en torno al aprovechamiento de la memoria VRAM y el coste marginal por inferencia LLM.

### El Patrón *Parent-Child Retrieval* y la RTX 3060 (12GB VRAM)
Inicialmente, saturar un modelo Cross-Encoder (`BGE-Reranker-v2-m3`) enviándole macro-documentos completos asfixiaba el hardware logrando latencias abismales de **259 segundos** por consulta, generando desbordamientos de OOM (Out Of Memory).
* **Solución Empírica:** Segmentamos el corpus en fragmentos Hijos microscópicos de máximo 400 tokens para la Vectorización Inicial. Qdrant recolecta los mejores (Ej: `top_k=10`). 
* El tensor del Cross-Encoder ahora aplica *self-attention* atómica y estricta en tiempo sub-segundo solo a estas pequeñas matrices. Una vez ranqueados, **remapeamos** ese micro-fragmento hacia su Documento Padre íntegro, pasándole este gran contexto al LLM generativo final.
* **Resultado:** Reducción computacional de latencia MLOps a **<0.3 segundos** para el Cross-Encoder, neutralizando completamente el síndrome de *"Lost in the Middle"* (Las respuestas correctas ya no están perdidas en un bloque masivo de texto ciego).

### Migración Efectiva a `gpt-4o-mini`
Mover la canalización desde APIs gratuitas genéricas hacia el modo de pago hiper-reducido generó eficiencias sistémicas enormes:
* **Fiabilidad Absoluta en Formato JSON:** Por contrato API (`response_format={"type": "json_object"}`), OpenRouter + OpenAI jamás retorna caracteres residuales; eliminando nativamente errores endémicos como el `EOF while parsing`, y simplificando el auto-healing de Pydantic dentro del código.
* **Presupuesto Marginal:** Permitir la inyección MLOps de ~2,000 papers íntegros por la ínfima tarifa de aproximadamente $1 USD.

---

## 3. Capa de Seguridad Activa (Filtros Anti-Alucinación)

Se ha instruido internamente al Proxy Intermediario (`rag_chain.py`) actuar como *Bouncer* de red impidiendo ataques de *Prompt Injection* simple y Alucinación LLM en caso de ruido.

* **[Guardrail G-01] Vacío Absoluto:** Si ni el Sparse Model ni el Dense Model hallan *score* base, la solicitud ni siquiera toca a OpenRouter y responde con una negativa en código duro.
* **[Guardrail G-02] Umbral de Certeza (Relevance Score):** El BGE-M3 otorga flotantes logíticos. Si el fragmento de mayor proximidad no supera estrictamente el umbral `Threshold = 0.15`, el sistema asume que la similitud es circunstancial (Ej: El usuario pregunta "¿Cuál es la capital de Francia?" en un repositorio de servidores Linux). Automáticamente se declara la carencia contextual al usuario truncando el flujo de red.
