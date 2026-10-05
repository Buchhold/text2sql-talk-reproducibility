FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EXPECTED_CUDA=13.0 \
    UNSLOTH_RETURN_LOGITS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.11

WORKDIR /opt/qwen35-eval
COPY requirements.txt .
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --python /opt/venv/bin/python --torch-backend=cu130 -r requirements.txt && \
    uv pip check --python /opt/venv/bin/python

COPY container_preflight.py training_preflight.py train.py ./
RUN /opt/venv/bin/python container_preflight.py \
    --expected-cuda=13.0 \
    --model-id=Qwen/Qwen3.5-4B
RUN /opt/venv/bin/python training_preflight.py \
    --expected-cuda=13.0 \
    --model-id=unsloth/Qwen3.5-4B

COPY entrypoint.py eval_metadata.py offline_eval.py ./
RUN /opt/venv/bin/python -m py_compile \
    entrypoint.py eval_metadata.py offline_eval.py train.py training_preflight.py
ENV PATH="/opt/venv/bin:${PATH}"
ENTRYPOINT ["python", "/opt/qwen35-eval/entrypoint.py"]