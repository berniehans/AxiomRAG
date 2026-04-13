# ==========================================
# Etapa 1: Builder (Compilación UV + Dependencias)
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias de sistema mínimas para compilar (si alguna lib lo requiere)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Instalación limpia de UV (Binario directo)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copiamos archivos de configuración para aprovechar el caché de Docker
COPY pyproject.toml uv.lock README.md ./

# Sincronizamos dependencias usando el lockfile (Esto crea el .venv automáticamente)
# --frozen: Garantiza que no se actualice el lockfile durante el build
# --no-install-project: Instala solo dependencias externas primero para optimizar cache
RUN uv sync --frozen --no-cache --no-install-project

# Ahora copiamos el código fuente e instalamos el proyecto local
COPY src/ src/
RUN uv sync --frozen

# ==========================================
# Etapa 2: Runtime (Imagen Ligera Prod)
# ==========================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia del entorno virtual completo y código
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/
COPY pyproject.toml .

EXPOSE 8000

# Ejecución usando el uvicorn del venv
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]