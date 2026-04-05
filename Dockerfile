# ==========================================
# Etapa 1: Builder (Compilación UV + Dependencias)
# ==========================================
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

# Instalar dependencias en el Virtual Env
RUN uv venv --python 3.12 /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache-dir .

# ==========================================
# Etapa 2: Runtime (Imagen Ligera Prod)
# ==========================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

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

# Copia del VENV limpio del builder
COPY --from=builder /app/.venv /app/.venv
# Copia del código fuente
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
