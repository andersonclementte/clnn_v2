# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PATH="/home/appuser/.local/bin:${PATH}"

# SO mínimo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 🔧 Instala deps Python GLOBALMENTE (root) — ficam em /usr/local/lib/... (visíveis para qualquer UID)
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir git+https://github.com/yahoojapan/geobleu.git fastdtw

# Usuário não-root para rodar a app
RUN groupadd -g ${GID} appuser && useradd -m -u ${UID} -g ${GID} appuser

# Código da app
COPY --chown=appuser:appuser . /workspace

# Entrypoint
COPY --chown=appuser:appuser entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
USER appuser
ENTRYPOINT ["/usr/bin/tini","--","/workspace/entrypoint.sh"]

EXPOSE 5000
