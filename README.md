# CLNN v2 — Human Mobility Prediction

Modelo de predição de mobilidade humana baseado em LSTM + fusão de informações externas, desenvolvido para o [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023).

O objetivo é prever a próxima localização (x, y) de usuários em uma grade 200×200, dadas suas trajetórias históricas e contexto (cidade, horário, POIs).

---

## Arquitetura do Modelo

```
┌──────────────────────────────────┐    ┌──────────────────────────┐
│   External Information Fusion    │    │   Partial Information    │
│                                  │    │                          │
│  User Embedding ─┐               │    │  Coord Sequence ──► Bi-LSTM
│  City Embedding ──┤               │    │  (seq_len × 2)          │
│  Day Projection ──┼──► Concat ──► Dense│    └──────────┬───────────┘
│  Time Projection ─┤               │                   │
│  POI Projection ──┘               │                   │
└──────────────┬───────────────────┘                   │
               │                                       │
               └──────────┬───────────────────────────┘
                          ▼
                  Weighted Fusion
                  (w_r · static + w_e · dynamic)
                          │
                          ▼
                  Destination Head
                  MLP → Softmax → Σ(prob × KMeans centers)
                          │
                          ▼
                  Predicted (x, y) ∈ [0, 1]²
```

**Componentes principais:**

| Módulo | Arquivo | Função |
|---|---|---|
| `ExternalInformationFusionNormalized` | `src/models/external_info.py` | Embeddings de usuário/cidade + projeções temporais e de POI |
| `CoordLSTM` | `src/models/partial_info.py` | LSTM bidirecional sobre a sequência de coordenadas |
| `WeightedFusion` | `src/models/humob_model.py` | Soma ponderada aprendível entre informação estática e dinâmica |
| `DestinationHead` | `src/models/humob_model.py` | MLP → softmax sobre K centros de cluster (KMeans) |

---

## Pré-requisitos

- **Docker** com [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado
- **GPU NVIDIA** com suporte a CUDA (testado com RTX A4000, 16 GB VRAM)
- **~5 GB de disco** para o arquivo de dados Parquet

> **Sem GPU?** O código funciona em CPU (mais lento). O dispositivo é detectado automaticamente; você pode forçar `device: "cpu"` no YAML.

---

## Setup

### 1. Clone o repositório

```bash
git clone https://github.com/andersonclementte/clnn_v2.git
cd clnn_v2
```

### 2. Coloque os dados

Baixe o arquivo `humob_all_cities_v2_normalized.parquet` e coloque-o em `data/`:

```
data/
└── humob_all_cities_v2_normalized.parquet   (~4.5 GB)
```

O Parquet contém dados já normalizados com as seguintes colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `uid` | int | ID do usuário |
| `city_encoded` | int | Cidade (0=A, 1=B, 2=C, 3=D) |
| `d_norm` | float | Dia normalizado [0, 1] |
| `t_sin`, `t_cos` | float | Timeslot codificado como seno/cosseno [-1, 1] |
| `x_norm`, `y_norm` | float | Coordenadas normalizadas [0, 1] |
| `POI_norm` | float[85] | Vetor de POIs normalizado |

### 3. Build da imagem Docker

```bash
docker build -t humob:cu128 .
```

A imagem é baseada em `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` e inclui todas as dependências Python.

---

## Como Rodar

### Execução completa via YAML

```bash
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace \
  -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD:/workspace" \
  humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/<ARQUIVO>.yaml
```

### Configs disponíveis

| Config | `sequence_length` | `n_clusters` | Épocas (base/ft) | Uso |
|---|---|---|---|---|
| `smoke.yaml` | 12 | 16 | 1/1 | Teste rápido (~minutos) |
| `big_smoke.yaml` | 720 | 512 | 8/5 | Teste com seq. longa |
| `exp_full_15d_a4000.yaml` | 720 | 512 | 8/5 | Experimento completo (A4000) |

### Pipeline (4 etapas)

O `run_automate.py` executa automaticamente:

1. **KMeans** — Calcula centros de cluster a partir das coordenadas (cidade base)
2. **Treino base** — Treina o modelo na cidade A
3. **Fine-tuning sequencial** — Fine-tuna em B → C → D (cada cidade parte do checkpoint anterior)
4. **Avaliação** — Compara zero-shot vs. fine-tuned + gera Pred+GT pairs

### Exemplo: teste rápido

```bash
# Limpe checkpoints antigos (obrigatório ao trocar de config)
rm -rf outputs/models/checkpoints/

docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace \
  -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD:/workspace" \
  humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/smoke.yaml
```

> ⚠️ **Importante:** Sempre limpe `outputs/models/checkpoints/` ao trocar de config, pois checkpoints com `n_clusters` diferente causam erro de shape mismatch.

---

## Saídas

Todos os artefatos são salvos em `outputs/`:

```
outputs/
├── models/            # Checkpoints (.pt)
│   └── checkpoints/   # Checkpoints intermediários por época
├── eval/              # Parquet com pares (predição, ground truth)
├── plots/             # Gráficos de treino
└── submissions/       # Submissões para o challenge
```

O MLflow salva logs em `mlruns/`. Para visualizar:

```bash
pip install mlflow
mlflow ui --backend-store-uri ./mlruns
# Acesse http://localhost:5000
```

---

## Estrutura do Projeto

```
clnn_v2/
├── config/                            # Configurações YAML dos experimentos
│   ├── smoke.yaml                     #   Teste rápido
│   ├── big_smoke.yaml                 #   Teste intermediário
│   ├── exp_full_15d_a4000.yaml        #   Experimento completo
│   └── humob_config_15days.yaml       #   Config legado
│
├── src/
│   ├── data/
│   │   └── dataset.py                 # HuMobNormalizedDataset (IterableDataset)
│   ├── models/
│   │   ├── humob_model.py             # HuMobModel (modelo principal)
│   │   ├── external_info.py           # Fusão de informações externas
│   │   └── partial_info.py            # CoordLSTM (LSTM bidirecional)
│   ├── training/
│   │   ├── train.py                   # Treino base + avaliação
│   │   ├── finetune.py                # Fine-tuning sequencial + comparação
│   │   └── pipeline.py                # Geração de submissão + Pred+GT pairs
│   └── utils/
│       ├── mlflow_tracker.py          # Wrapper MLflow
│       ├── simple_checkpoint.py       # Save/load/cleanup de checkpoints
│       └── pytorch_compat.py          # Compatibilidade entre versões PyTorch
│
├── scripts/
│   ├── train.py                       # Script de treino alternativo
│   ├── evaluate.py                    # Avaliação completa com ranking
│   └── setup_check.py                # Verificação de ambiente
│
├── data/                              # Dados (não versionados)
├── outputs/                           # Artefatos de saída
├── mlruns/                            # Logs MLflow (não versionados)
│
├── run_automate.py                    # ★ Ponto de entrada principal (YAML → pipeline)
├── run.py                             # Ponto de entrada legado
├── dockerfile                         # Imagem Docker (CUDA 12.8 + PyTorch 2.7)
├── entrypoint.sh                      # Verifica GPU antes de executar
├── requirements.txt                   # Dependências Python
└── PRD.md                             # Plano de desenvolvimento futuro
```

---

## Criando seu próprio config

Copie um YAML existente e ajuste:

```yaml
experiment_name: meu_experimento
logging:
  tracking_uri: "file:./mlruns"

data:
  parquet_file: "humob_all_cities_v2_normalized.parquet"
  base_city: "A"                    # Cidade para treino base
  finetune_cities: ["B","C","D"]    # Cidades para fine-tuning

model:
  n_clusters: 512                   # Clusters KMeans (16, 128, 256, 512)
  n_users: 100000                   # Nº total de usuários no dataset
  sequence_length: 720              # Janela temporal (720 = 15 dias × 48 slots/dia)

training:
  base:
    n_epochs: 8
    learning_rate: 3e-4
    batch_size: 64                  # Ajustar conforme VRAM disponível
    mixed_precision: true
  finetune:
    n_epochs: 5
    learning_rate: 5e-5
    batch_size: 16

hardware:
  device: "cuda"                    # "cuda" ou "cpu"
  pin_memory: true
  num_workers: 4

eval:
  compare: true
  pred_gt:
    enabled: true
    split: "val"                    # "val" (tem ground truth) ou "test"
    day_range: [61, 75]
    batch_size: 2048
    target_cities: ["B","C","D"]

submission:
  enabled: false                    # Ativar para gerar arquivo de submissão
  days: [61, 75]
  target_cities: ["B","C","D"]

outputs:
  models_path: "outputs/models/"
  submissions_path: "outputs/submissions/"
  plots_path: "outputs/plots/"
  eval_path: "outputs/eval/"
```

**Regra prática para `batch_size`:** Com sequence_length=720 na A4000 (16 GB), use batch_size=64 para treino base e 16 para fine-tuning.

---

## Referências

- [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023) — Competição de predição de mobilidade humana
- [PyTorch](https://pytorch.org/) — Framework de deep learning
- [MLflow](https://mlflow.org/) — Tracking de experimentos
