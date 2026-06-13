---
trigger: always_on
---

# Load Project Docs

Al trabajar en este repositorio:

- Usa `agents.md` como la fuente principal de reglas generales del proyecto.
- Usa `ARCHITECTURE.md` como la referencia principal para respetar la arquitectura, capas, límites y decisiones técnicas.
- Usa `README.md` para entender el objetivo funcional, setup y forma de uso del proyecto.
- Usa UV para ejecutar comandos.

## Comportamiento esperado
- Antes de proponer cambios grandes, revisa `ARCHITECTURE.md`.
- Si una implementación contradice `agents.md` o `ARCHITECTURE.md`, prioriza esos archivos y avisa la contradicción.
- Si falta contexto, pide aclaración en lugar de asumir.

## Directrices Estrictas de Desarrollo (AxiomRAG Agentivo v2)
1. **Preservación del Rendimiento:** Está prohibido sustituir las consultas optimizadas a Qdrant o el cálculo local de BM25 por integraciones genéricas preconstruidas de LangChain si estas degradan los tiempos de respuesta inferiores al segundo actuales.
2. **Encapsulamiento de Abstracciones:** El motor de búsqueda híbrido se debe exponer extendiendo la clase abstracta `BaseRetriever` de LangChain.
3. **Estructura del Estado del Grafo:** Todo el flujo se controlará a través de un `StateGraph` de LangGraph, donde el estado (`GraphState`) rastreará estrictamente de forma asíncrona la query original, queries expandidas, documentos recuperados, puntuaciones de relevancia, conteo de re-intentos de bucles y la respuesta final generada.
4. **Manejo de Errores y Fallbacks:** Se mantendrán los mecanismos defensivos actuales (fallbacks si falla la API principal), integrándolos como rutas alternativas o nodos de contingencia dentro del grafo.