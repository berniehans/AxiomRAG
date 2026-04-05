# 🏢 Motor RAG de Grado Industrial con Reranking de sub-segundo

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi)
![NVIDIA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia)
![Qdrant](https://img.shields.io/badge/Qdrant-Persistent-red?style=for-the-badge&logo=qdrant)

Repositorio de grado de producción que implementa una arquitectura **Retrieval-Augmented Generation (RAG)** asíncrona, enfocándose en la soberanía de los datos (local-first) y el rendimiento determinista escalado sobre aceleradores de hardware NVIDIA. 

## ⚙️ Enterprise Features (Data Engineering & Hardware)

El sistema integra componentes rigurosos para solventar los fallos típicos (alucinaciones, pérdida del contexto y cuellos de botella CPU) de los RAG convencionales:

- **Aceleración Nativa CUDA:** Inferencia optimizada explícitamente para arquitecturas NVIDIA Ampere empleando CUDA 12.4. Esto permitió reducir la latencia del componente de Re-Ranking logrando tiempos consistentes de `< 500ms` en una RTX 3060.
- **Recuperación *"Parent-Child"*: ** Implementación de separación estricta: Búsqueda focalizada vectorial sobre fragmentos agudos (Hijos Semánticos de 600 tokens) para obtener puntajes de precisión de ~0.95, combinada con la inyección del archivo Global (Padres Completos) al payload del LLM, erradicando el problema de la pérdida de contexto.
- **Búsqueda Híbrida Ponderada:** Ensamble matemático (50/50 Ensemble) de motor léxico `BM25` (Sparse) y motor vectorial semántico `BGE-M3` (Dense) previniendo "Zero Matches" en terminología técnica y acrónimos severos.
- **Ingesta Asíncrona (Non-Blocking):** Pipeline encapsulado sobre `FastAPI BackgroundTasks`. Absorbe 289 documentos técnicos persistiendo la evaluación global sin detener el hilo principal ni agotar el Thread Pool de las peticiones HTTP del usuario.

## 📊 Observabilidad y MLOps (Baseline Heurístico)

No se asume el rendimiento; se mide. Operamos auditorías automatizadas contra un **"Golden Dataset"** (batería de pruebas de ingenieros humanos) garantizando que nuestras refactorizaciones no degraden la calidad generativa previniendo cualquier alucinación.

**Línea Base Cuantitativa con Framework `Ragas v0.2+`:**

| Métrica MLOps | Score Evaluado | Significado Operativo |
| :--- | :--- | :--- |
| **Faithfulness** | `0.6061` | Nivel de fidelidad restrictiva de la IA contra el contexto extraído. |
| **Context Precision** | `0.6667` | Exactitud del Reranker midiendo el ratio de ruido (basura léxica o vectorial) del material recuperado. |

## 🌊 Arquitectura de Ingestión y Recuperación

```mermaid
flowchart LR
    %% Definición de Estilos
    classDef ingestion fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;
    classDef retrieval fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32;
    classDef generation fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#ef6c00;
    classDef eval fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2;

    subgraph Fase_Ingesta ["📦 INGESTA CRÍTICA (GPU)"]
        direction TB
        A([📄 Docs: PDF/XLSX]) --> B[✂️ Semantic Chunking]
        B --> C{Indexación Dual}
        C --> D[(🗂️ Qdrant: Child Chunks)]
        C --> E[(💾 LocalStore: Parent Docs)]
    end

    subgraph Fase_Busqueda ["🔍 RECUPERACIÓN HÍBRIDA"]
        direction TB
        F[❓ Query] --> G(Hybrid Search: BM25 + BGE-M3)
        G --> H[⚖️ Re-Ranking: Cross-Encoder]
        H --> I[🏆 Top 3 Gold Context]
    end

    subgraph Fase_Gen ["🤖 GENERACIÓN & SEGURIDAD"]
        direction TB
        I --> J[🧠 LLM: OpenRouter/Groq]
        J --> K[🛡️ Guardrails: Score Check]
        K --> L[✅ Respuesta Citada]
    end

    subgraph Fase_MLOps ["🧪 EVALUACIÓN & OBSERVABILIDAD"]
        direction TB
        L --> M[📊 RAGAS Metrics]
        M -- "Feedback Loop" --> G
        M -- "Tuning" --> B
    end

    %% Conexiones entre subgrafos
    Fase_Ingesta -.-> Fase_Busqueda
    Fase_Busqueda ==> Fase_Gen
    Fase_Gen -.-> Fase_MLOps

    %% Aplicación de Clases
    class A,B,C,D,E ingestion;
    class F,G,H,I retrieval;
    class J,K,L generation;
    class M eval;
```

## 🏗️ Estructura del Código Fuente

El diseño modular respeta el patrón de "Separation of Concerns" bajo tipado estricto `PEP 484`:

```text
📦 RAG_Project
 ┣ 📂 scripts/        # Orquestadores ejecutivos (run_ingestion.py, run_evals.py)
 ┣ 📂 src/ 
 ┃ ┣ 📂 agent/        # Lógica conversacional, prompts restrictivos de no-alucinación.
 ┃ ┣ 📂 evals/        # Motor iterativo para Ragas evaluando métricas MLOps.
 ┃ ┣ 📂 ingestion/    # Modelos locales, Semantic Chunking, y parseo Pydantic global.
 ┃ ┣ 📂 retrieval/    # Ensamble avanzado: BM25, Qdrant Client, Cross-Encoders.
 ┃ ┗ 📜 main.py       # ASGI FastAPI Server, Gestión HW (empty_cache via lifespan).
 ┗ 📜 pyproject.toml  # Lock dependencies (uv) apuntando a index pytorch-cu124.
```

## 🗺️ Roadmap de Producto

Hitos de estabilización pendientes enfocados en escalar la IA a nivel Institucional:

1. **Dockerización Nativa:** Despliegue empaquetado multi-stage soportando injecciones de driver NVIDIA Runtime Container Toolkit.
2. **Telemetría de Red Local:** Instrumentación de logs sub-militamétricos separando el tiempo de procesamiento (Embeddings vs Híbrido vs Generación).
3. **Chain-of-Thought (CoT):** Inyecciones de prompts ocultos forzando comprobaciones de razonamiento técnico en modelos locales Open-Source.
