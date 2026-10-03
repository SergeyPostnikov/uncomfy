# CUDA 12.6 + Python 3.12 — matches the ComfyUI venv torch build
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # ComfyUI root inside the container
    COMFY_PATH=/comfyui \
    # uncomfy DB path
    DATABASE_URL=sqlite+aiosqlite:////data/uncomfy.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# --- ComfyUI ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Install PyTorch with CUDA 12.6 first (heavy, separate layer for cache)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install ComfyUI requirements (minus torch)
RUN pip install --no-cache-dir \
    comfyui-frontend-package \
    comfyui-workflow-templates \
    comfyui-embedded-docs \
    torchsde einops transformers tokenizers sentencepiece \
    safetensors aiohttp yarl pyyaml Pillow scipy tqdm psutil \
    alembic gguf protobuf

# --- uncomfy deps ---
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" taskiq taskiq-redis \
    "sqlalchemy[asyncio]" aiosqlite alembic \
    pydantic-settings httpx python-multipart greenlet

# --- uncomfy source ---
COPY uncomfy/ ./uncomfy/

# models volume mount point
RUN mkdir -p /comfyui/models /comfyui/output /comfyui/input /data

EXPOSE 8000

CMD ["uvicorn", "uncomfy.main:app", "--host", "0.0.0.0", "--port", "8000"]
