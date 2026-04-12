# AGENTS.md — Reglas y Mapa Cognitivo

Este archivo sirve como punto de referencia universal para agentes o sistemas de LLM que requieran interactuar, debugar o escalar este repositorio. Integra las reglas estándar de desarrollo con el contexto específico de AxiomRAG.

## 📌 Contexto Principal
Este repositorio es un **Motor Avanzado de Retrieval-Augmented Generation (RAG)** enfocado en documentos administrativos y archivos complejos, estructurado bajo un paradigma moderno de ML Ops industrial.

> **Mantra del Proyecto:** Fiabilidad, tipado estricto y blindaje anti-alucinación por Reranker local prevalecen frente a dependencias externas inestables. Todo modelo fundacional idealmente residirá localmente en GPU.

## Stack Base
- Python 3.12
- `uv` para gestión de dependencias (`pyproject.toml` estricto con PyTorch `2.6.0+cu124` y ecosistema LangChain 1.x)
- FastAPI / Pydantic para APIs y validación
- PostgreSQL / Redis / Qdrant para persistencia y búsqueda vectorial
- Docker para infraestructura
- pytest para testing

## Estructura de Proyecto y Topografía del Sistema
El repositorio está agrupado en directorios lógicos y encapsulados:
```text
AxiomRAG/
├── README.md
├── ARCHITECTURE.md
├── TASKS.md
├── agents.md              ← (Este archivo) Reglas y mapa cognitivo
├── docs/                  ← Base de conocimiento de arquitectura (flujos, límites, CVEs)
├── scripts/               ← Puntos de entrada MLOps (ej. run_ingestion.py, test_retrieval.py)
├── src/
│   ├── api/               ← Endpoints
│   ├── agent/             ← Motor Generativo (ej. rag_chain.py)
│   ├── retrieval/         ← Lógica core (ej. advanced_retrieval.py, ParentDocumentRetriever)
│   ├── services/          ← Lógica de negocio
│   └── repositories/      ← Acceso a datos
├── tests/                 ← Unit e integration tests
├── docker-compose.yml
└── pyproject.toml
```

## 🧱 Flujos de Mutación y Extensibilidad
Si debes modificar código o expandir el framework, toma estas salvaguardas arquitectónicas:
- **Nuevos Metadatos:** Siempre utiliza herencia fuerte de Pydantic `BaseModel` en las clases de metadata con sus respectivos `Field(description="...")`. El auto-healing depende de la propagación de errores de validación si el LLM alucina un JSON.
- **FS (File Storage):** La clase personalizada `DocumentJsonFileStore` extenderá `LocalFileStore` iterando las estructuras; todo Key-Store físico de padres debe prever una creación automática de path (`os.makedirs`).
- **Límites de CUDA/Hardware:** NUNCA actualices la dependencia de PyTorch ciegamente vía pip. La versión es estrictamente `pytorch-cu124` mapeada al índice oficial dentro del toolchain `uv` en `pyproject.toml`.

## Calidad de Código
- Funciones < 30 líneas, archivos < 300 líneas
- Complejidad ciclomática < 10
- Tipar TODO lo público (Pydantic models)
- Docstrings en services y utils: breve + ejemplos
- No console.log/debug prints en prod

## Testing Obligatorio
- Unit tests para utils/services (pytest)
- Integration tests para endpoints
- Coverage >80% en código nuevo
- Mockear externas (DB, APIs)

## Seguridad
- Validar TODO input de usuario
- Secrets SOLO en .env, nunca hardcode
- No loggear tokens/keys
- Try/catch en APIs externas + logs con contexto

## Git Workflow
- Conventional commits: feat:, fix:, docs:, chore:
- Branches: feature/nombre-descriptivo
- PRs <400 líneas diff
- Tests + lint antes de merge

## Boundaries del Agente
- ✅ Siempre: proponer tests primero, usar src/ structure
- ⚠️ Consultar: nuevas deps, cambios en DB schema
- 🚫 Nunca: commitear .env/secrets, borrar tests, push directo a main

## Prompt Style
- Sé conciso pero completo
- Explica RAZÓN de cambios
- Propón alternativas si hay trade-offs