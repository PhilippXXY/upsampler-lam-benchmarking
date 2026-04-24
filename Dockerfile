# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.13.3

FROM python:${PYTHON_VERSION}-slim-bookworm

# Metadata labels
LABEL org.opencontainers.image.title="Upsampler-LAM Benchmarking"
LABEL org.opencontainers.image.description="Benchmarking framework for evaluating upsamplers with the Latent Acoustic Mapping (LAM) model"
LABEL org.opencontainers.image.authors="Philipp Schmidt"
LABEL org.opencontainers.image.url="https://github.com/PhilippXXY/upsampler-lam-benchmarking"
LABEL org.opencontainers.image.source="https://github.com/PhilippXXY/upsampler-lam-benchmarking"
LABEL org.opencontainers.image.documentation="https://github.com/PhilippXXY/upsampler-lam-benchmarking/blob/main/README.md"

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgeos-dev \
    libgeos-c1v5 \
    proj-bin \
    proj-data \
    libproj-dev \
    gcc \
    g++ \
    python3-dev \
    && uv sync --frozen --no-dev --no-install-project \
    && apt-get purge -y --auto-remove \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/uv

COPY src/ ./src/
COPY config/inference_config.yaml ./config/inference_config.yaml

RUN mkdir -p /app/output /app/logs

# Set entrypoint to run inference
ENTRYPOINT ["python", "src/infer.py"]
CMD ["--config", "/app/config/inference_config.yaml"]
