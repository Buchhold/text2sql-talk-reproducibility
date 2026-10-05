FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXPECTED_CUDA=13.0 \
    FLA_CACHE_MODE=default

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.11

WORKDIR /opt/qwen38-27b-sft
COPY requirements.txt .
# Exact dependency sequence from Unsloth's Qwen3.8 Kaggle notebook. This
# empty CUDA image has no Torch installed, so requirements.txt uses that
# notebook's fallback branch: unpinned torch and torchvision.
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --python /opt/venv/bin/python --torch-backend=cu130 -r requirements.txt && \
    uv pip install --python /opt/venv/bin/python --no-deps --upgrade "torchao>=0.16.0" && \
    uv pip install --python /opt/venv/bin/python --upgrade transformers==5.15.1 && \
    uv pip install --python /opt/venv/bin/python --no-deps --upgrade trl==0.22.2 && \
    uv pip install --python /opt/venv/bin/python --no-deps --upgrade \
      "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
      "unsloth[base] @ git+https://github.com/unslothai/unsloth"

COPY training_preflight.py train.py entrypoint.py ./
RUN /opt/venv/bin/python training_preflight.py --expected-cuda=13.0 --model-id=unsloth/Qwen3.8-27B-unsloth-bnb-4bit
RUN /opt/venv/bin/python -m py_compile entrypoint.py train.py training_preflight.py
ENV PATH="/opt/venv/bin:${PATH}"
ENTRYPOINT ["python", "/opt/qwen38-27b-sft/entrypoint.py"]