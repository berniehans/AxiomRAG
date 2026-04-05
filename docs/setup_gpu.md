# 🖥️ Configuración de Hardware: NVIDIA RTX & MLOps

Hemos migrado toda la orquestación a tensores acelerados en GPU, diseñando específicamente un backend en torno a **NVIDIA RTX 3060**, estandarizando las librerías a CUDA 12.4 para Python 3.12.

## 📦 Inicialización con `uv` y PyTorch Indexes

Para forzar la plataforma a ingerir los sub-módulos C++ compilados con CUDA, `pyproject.toml` especifica un anclaje forzado sobre el flag index explícito en la herramienta UV:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
```
Este puente explícito exige estrictamente la resolución nativa de hardware, bloqueando las descargas fallidas por la clásica ruta efímera de CPU.

## 🧠 Gestión Dinámica de VRAM (FastAPI Lifespan)

Nuestra API web MLOps previene desbordamientos de Memoria (OOM - Out of Memory) mediante un bloqueador y despachador de eventos asíncronos nativo del Router de FastAPI (Lifespan).

El servidor hace un vaciado activo por evento de cierre y apertura de ciclos en el hardware gráfico:
```python
import torch

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # 1. Empuje inicial. Vacía por completo la VRAM heredada de la tarjeta gráfica
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ [HARDWARE] Limpieza inicial de VRAM superada.")

    # 2. Inicialización estricta instanciando model_kwargs={"device": "cuda"}
    ...
```

Al terminar el API, un `gc.collect()` apoyado con un robusto desfragmentador `torch.cuda.empty_cache()` evita las clásicas fugas (Mem-Leaks) que fracturarían el ancho de banda del bus PCIe a corto plazo.

De modo que se garantiza un sistema sumamente seguro, **Scalable** y **Ready for Production** sin depender de caídas de proceso asíncrono.
