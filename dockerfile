# syntax=docker/dockerfile:1.6
# Base default: PyTorch 2.7.0 + CUDA 12.8 (serve para Nitro e CIA6)
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

# Usuário não-root
RUN groupadd -g ${GID} appuser && useradd -m -u ${UID} -g ${GID} appuser
WORKDIR /workspace
USER appuser

# Dependências Python
COPY --chown=appuser:appuser requirements.txt /workspace/requirements.txt
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt


COPY --chown=appuser:appuser . /workspace

# Entrypoint: sanity de GPU e repassa o comando
COPY --chown=appuser:appuser entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini","--","/workspace/entrypoint.sh"]

# (Opcional) Porta do MLflow server, se for usar
EXPOSE 5000
