# 📥 Ingestion Pipeline & Metadata

El módulo de Ingesta no solo inserta datos; es una capa de preprocesamiento, limpieza técnica (Data Wrangling), y transformación dimensional. Se destaca el "Auto-Sanado" al contactar LLMs para extraer atributos JSON.

## 1. Semantic Chunking (Fragmentación Recusiva)

Descartamos el agrupamiento tradicional basado netamente en contadores de palabras y empleamos `RecursiveCharacterTextSplitter` con heurísticas pragmáticas.

- **Micro-Matches Dinámicos:** Mantenemos la cohesión sintáctica buscando límites duros (como puntos y aparte, saltos de línea y tabulaciones de listas).
- **Relaciones Padre-Hijo (Parent Document Paradigm):** Partimos grandes PDFs en fragmentos diminutos (`chunk_size=400`, `chunk_overlap=50`). Estos *micro-chunks* vuelan de forma optimizada hacia nuestra Base Vectorial Qdrant, mientras mantenemos un local file key-store mapeando ese micro-fragmento hacia su macro-documento de origen.

## 2. Metadata Extraction con Auto-Healing (Pydantic)

Archivos burocráticos implican grandes bloques de basura estática. Invocamos un parsing estructurado con **Pydantic** a través de `Structured Output` de LangChain.

### Estructura de Salida Modelada (Definición Fuerte)
```python
class DocumentMetadata(BaseModel):
    document_type: str = Field(description="Tipo de contrato: SLA, Manual Escolar, Acuerdo legal.")
    authors: List[str] = Field(description="Partes firmantes o autores definidos.")
    summary: str = Field(description="Briefing de no más de 20 palabras.")
```

### Auto-Sanado (Auto-Healing mechanism)
El framework atrapa automáticamente deserciones JSON generadas por el LLM mediante la mecánica de "Re-try with correction".
1. Si el LLM retorna texto truncado (e.g. cortes por fallos en la estructura de comillas).
2. Se levanta un interceptor `ValidationError` de Pydantic.
3. Se invuelve el mismo error como Prompt Correctivo sumado a la string cruda previa y se lanza la corrección de vuelta a la red neuronal hasta parsear un AST limpio.

### Inserción Local (LocalStorage / Qdrant)
Los metadatos purificados se adhieren al Parent Document (al File Store `data/kv_store/`) permitiendo a futuro aplicar "Filtros Metadata" (ej. "Trae todo lo coincidente al tema X, *filtrado exclusivamente al contratista Y*").

## 3. Flujo Lógico de Extracción

```mermaid
sequenceDiagram
    participant Archivos
    participant MultimodalParser
    participant SemanticChunker
    participant MetadataExtractor
    participant VectorDB
    
    Archivos->>MultimodalParser: Batch PDF / Excel Corporativos
    MultimodalParser->>SemanticChunker: Texto Crudo (MultiIdiomas auto-detectados)
    SemanticChunker->>MetadataExtractor: Mapeo de Sub-Elementos
    MetadataExtractor-->>LLM: with_structured_output (OpenRouter)
    LLM-->>MetadataExtractor: Atributos Pydantic (Autor, Categoria, Resumen)
    MetadataExtractor->>VectorDB: Ingesta Definitiva con Metadatos Aglomerados
```

### Hitos MLOps Claves adicionales:
- **Robustez Multilingüe con Tesseract:** Al procesar mediante dependencias `unstructured`, la partición respeta el estándar `languages=["spa", "eng"]` evitando fallos nativos con contratos en Spanglish, o donde coexiste fuertemente literatura técnica en inglés.
