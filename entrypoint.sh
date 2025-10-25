#!/usr/bin/env bash
set -euo pipefail

# Sanity rápido da GPU
python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU name unavailable:", e)
PY

# Executa o comando passado ao container
exec "$@"
