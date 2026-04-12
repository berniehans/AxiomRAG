# 🖥️ GPU HW Optimization & Security Mitigation

El entorno base opera intensivamente sobre una GPU **NVIDIA RTX 3060 (12GB VRAM)**. Esta documentación proporciona los lineamientos estructurales definidos para maximizar la inferencia con PyTorch 2.6 localmente limitando las vulnerabilidades (CVE-2025-32434 mitigada).

## Especificaciones Tecnológicas (CUDA 12.4 + Torch 2.6.0)

Al gestionar flujos de inferencia densos (Cross-Encoder Re-Ranking + Embeddings `BGE-M3`), es mandatorio la alineación de frameworks:

- **PyTorch Engine**: Ver. `2.6.0+cu124`
- **CUDA Runtime**: `12.4`

### Resolución de Vulnerabilidad (Security Patch) a nivel FW
> **Problema**: CVE-2025-32434 afectaba los tensores distribuidos y lecturas corruptas de memoria en PyTorch versiones <= 2.5, escalando el riesgo durante el alojamiento local de Modelos HuggingFace de origen no confiable (Pickle/Safetensors exploit vector).
> **Solución Implementada**: Nuestro `pyproject.toml` especifica estrictamente the wheel source en `torch>=2.6.0` con origin check directo de repositorios `pytorch-cu124`. Por ende, el RAG ha mitigado los *Buffer Overflows* locales.

## Manejo de VRAM (`RTX 3060` 12GB)

El hardware restringe nuestro scope a 12GB de Memoria de Video. Para evitar OOM (*Out Of Memory* errors) con Transformers, se han configurado los siguientes perfiles restrictivos:

1. **Pooling Contextual Limitado**: Los Embeddings `BGE-M3` y `BGE-Reranker` son forzados a cargar exclusivamente la mitad requerida de pesos o utilizan carga iterativa asíncrona a nivel script.
2. **Batch Processing (Document Chunking)**: La ingesta de archivos pre-vectorización usa lotes delimitados en vez del pipeline masivo habitual (`chunk_size=400`, `overlap=50` en Qdrant + local JSON Store, ahorrando picos de inferencia CUDA).
3. **Carga Segura de Pytorch**: El flag de `trust_remote_code` es permanentemente deshabilitado a menos de ser auditable directamente. Todo peso de HuggingFace descargado se valida en `.safetensors`.

### Scripts y Troubleshooting:

Si surgen problemas de CUDA con `uv`:
```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```



Confirmar alocación de tensores en placa:
```python
import torch
print(f"CUDA status: {torch.cuda.is_available()}")
print(f"Alocación: {torch.cuda.get_device_name(0)}")
```

## Gestión Dinámica de VRAM (API Lifespan)

Nuestra API web MLOps previene desbordamientos de Memoria mediante un bloqueador y despachador de eventos asíncronos nativo del Router de FastAPI (Lifespan).

El servidor hace un vaciado activo por evento de cierre y apertura de ciclos en el hardware gráfico:
```python
import torch
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # 1. Empuje inicial. Vacía por completo la VRAM heredada de la tarjeta gráfica
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ [HARDWARE] Limpieza inicial de VRAM superada.")

    # 2. Inicialización estricta MLOps
    yield

    # Limpieza final en terminación
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

De este modo mitigamos fugas temporales y protegemos el canal PCIe de desbordarse masivamente al procesar lotes asíncronos en paralelo.
