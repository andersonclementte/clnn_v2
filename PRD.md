# PRD — CLNN v2: Human Mobility Prediction

> **Última atualização:** 2026-03-31
> **Contexto:** Este documento consolida todo o conhecimento adquirido até março de 2026, incluindo bugs corrigidos, resultados reais, limitações conhecidas e próximos passos. Escrito para que qualquer LLM ou desenvolvedor possa retomar o trabalho sem perda de contexto.

---

## 1. Contexto do Desafio

### 1.1 Histórico dos Challenges

| Edição | Nome | Dataset | Cidades | Métricas Oficiais |
|--------|------|---------|---------|-------------------|
| 2023 | HuMob Challenge 2023 | YJMob100K (cidade única, Japão) | 1 | GEO-BLEU, DTW |
| 2024 | HuMob Challenge 2024 | YJMob multi-cidade (~4.5 GB) | A, B, C, D | GEO-BLEU, DTW |
| 2025 | SIGSPATIAL GIS CUP 2025 | LY Corporation (~160 GB) | Multi-cidades | GEO-BLEU, DTW |

**Este projeto usa o dataset do HuMob 2024** (cidades A, B, C, D), mas o objetivo é publicar um artigo e potencialmente adaptar para o SIGSPATIAL GIS CUP 2025 (dados abrem em 30/04/2025 — já passados; o prazo final era 10/09/2025).

### 1.2 Definição do Problema

- **O que é previsto:** A localização (x, y) de um indivíduo em uma grade 200×200 a cada intervalo de 30 minutos
- **Período de predição:** Dias 61–75 (15 dias × 48 slots = 720 passos por usuário)
- **Período de treino:** Dias 1–60
- **Resolução espacial:** Grade 200×200, cada célula = 500m × 500m
- **Área total:** ~100km × 100km por cidade

### 1.3 Estrutura do Dataset (HuMob 2024)

- **Cidade A:** 100.000 usuários, dados completos (dias 1–75) — usada para treino base
- **Cidades B, C, D:** Dados de dias 1–60 disponíveis; dias 61–75 são o alvo de predição
  - Cidade B: ~25.000 usuários
  - Cidade C: ~20.000 usuários
  - Cidade D: ~6.000 usuários
- **Importante:** Os usuários **transitam entre cidades** — um indivíduo pode aparecer em A num dia e em B em outro. A esparsidade nos dados de B, C, D é consequência direta disso, não de ausência de dados.
- **Colunas principais:** `uid`, `city_encoded` (0=A,1=B,2=C,3=D), `d_norm` [0,1], `t_sin`, `t_cos`, `x_norm`, `y_norm`, `POI_norm` (85-D)
- **Arquivo:** `humob_all_cities_v2_normalized.parquet` (~4.5 GB, já normalizado)

### 1.4 Métricas Oficiais do Challenge

✅ **GEO-BLEU e DTW já implementados** (`src/utils/metrics.py`, `evaluate_metrics.py`).

| Métrica | Descrição | Melhor valor | Top 2024 | Nosso resultado (médio) |
|---------|-----------|-------------|----------|------------------------|
| **GEO-BLEU** | Similaridade geográfica de trajetórias (dia a dia) | Maior = melhor | 0.319 (Team 22) | ~0.079 (B+C+D) |
| **DTW** | Dynamic Time Warping — similaridade temporal de trajetórias | Menor = melhor | 27.15 (Team 24) | ~189 (B+C+D) |

Implementação de referência: https://github.com/yahoojapan/geobleu

---

## 2. Arquitetura do Modelo

### 2.1 Visão Geral

```
External Information Fusion          Partial Information (LSTM)
─────────────────────────────        ──────────────────────────
User Embedding (4-D)  ─┐             Coord Sequence
City Embedding (4-D)  ─┤             (seq_len × 2) ──► BiLSTM
Day Projection        ─┼──► Concat ──► Dense(8-D)
Time Projection       ─┤
POI Projection        ─┘
        │                                    │
        └────────────┬───────────────────────┘
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

### 2.2 Parâmetros do Modelo (config atual)

| Parâmetro | Valor | Observação |
|-----------|-------|------------|
| `sequence_length` | 720 | 15 dias × 48 slots |
| `n_clusters` | 512 | KMeans sobre coordenadas da cidade A |
| `n_users` | 100.000 | Total de usuários no dataset |
| Parâmetros treináveis | ~662K | Modelo pequeno para o problema |
| `lstm_hidden` | pequeno | Provavelmente subparametrizado |
| `user_emb_dim` | 4 | Muito pequeno para 100K usuários |

### 2.3 Pipeline de Treinamento

1. **KMeans** — Calcula 512 centros de cluster sobre coordenadas da cidade A
2. **Treino base** — 8 épocas na cidade A (AdamW, ReduceLROnPlateau, AMP)
3. **Fine-tuning sequencial** — B → C → D, cada um partindo do checkpoint anterior (5 épocas, CosineAnnealingLR)
4. **Avaliação** — Compara zero-shot vs fine-tuned + gera Pred+GT pairs

---

## 3. Resultados Obtidos (Experimento: exp_full_15d_a4000, rodado em março 2026)

### 3.1 Treinamento Base (Cidade A)

| Época | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 0.00550 | 0.00409 |
| 4 | 0.00364 | 0.00390 |
| 8 | 0.00351 | 0.00384 |

### 3.2 Fine-tuning por Cidade

| Cidade | Épocas rodadas | Val Loss final |
|--------|---------------|----------------|
| B | 5 (early stop na 5ª) | 0.00268 |
| C | 5 (early stop na 5ª) | 0.00234 |
| D | 3 (verificar logs) | 0.00231 |

### 3.3 Avaliação Final — cell_error (Zero-shot vs Fine-tuned)

Avaliado com `n_samples=1000` usando `evaluate_model`:

| Cidade | Zero-shot cell_error | FT cell_error | Melhoria | Zero-shot km | FT km |
|--------|---------------------|---------------|----------|-------------|-------|
| B | 4.59 células | 3.87 células | -15.7% | 2.30 km | 1.94 km |
| C | 11.36 células | 10.04 células | -11.6% | 5.68 km | 5.02 km |
| D | 4.86 células | 4.57 células | -6.0% | 2.43 km | 2.28 km |

### 3.4 Cobertura de Usuários (pred_gt_all_users.parquet)

Avaliado sobre todos os usuários com dados suficientes (`test_days=(0.55, 1.0)`, `max_seqs=30`):

| Cidade | Usuários avaliados | Total do dataset | Cobertura | Mediana | Média | P90 |
|--------|-------------------|-----------------|-----------|---------|-------|-----|
| B | 2.334 | ~25.000 | 9.4% | **2.12 km** | 4.41 km | 11.34 km |
| C | 1.406 | ~20.000 | 7.1% | **4.30 km** | 5.48 km | 11.01 km |
| D | 1.970 | ~6.000 | 32.9% | **2.06 km** | 3.87 km | 9.55 km |

**Por que só 5–33% dos usuários?** Com `seq_len=720` (15 dias de contexto), o usuário precisa ter 720 registros consecutivos na cidade. Como os usuários transitam entre cidades, a maioria só aparece esporadicamente em B, C, D.

### 3.5 Métricas Oficiais — GEO-BLEU e DTW (rodado em 2026-03-31)

Avaliado sobre `pred_gt_all_users.parquet` (130.007 predições, 5.033 usuários):

| Cidade | Usuários | GEO-BLEU | DTW | GEO-BLEU top 2024 | DTW top 2024 |
|--------|----------|----------|-----|-------------------|--------------|
| B | 2.334 | **0.1017** | **174.70** | 0.319 | 27.15 |
| C | 1.406 | **0.0276** | **236.92** | 0.319 | 27.15 |
| D | 1.970 | **0.1076** | **156.91** | 0.319 | 27.15 |

**Interpretação:**
- B e D atingem GEO-BLEU ~0.10, que é um baseline razoável considerando que o modelo não foi otimizado para essa métrica
- C apresenta GEO-BLEU 0.027, consistente com o cell_error 2× maior observado anteriormente
- DTW alto (~175–237 vs 27 do top) indica desvios temporais significativos nas trajetórias previstas
- A diferença para o top é esperada: os líderes do HuMob 2024 usaram Transformers com arquiteturas muito maiores

---

## 4. Bugs Corrigidos (histórico importante)

### Bug 1 — `val_days` insuficiente no fine-tuning (`finetune.py`)
- **Problema:** `val_days=(0.8, 1.0)` = ~15 dias = 720 time steps = exatamente `sequence_length`. Impossível formar sequências válidas → `val_count=0` → `val_loss=0.0`
- **Sintoma:** Early stopping após 2 épocas (de 5 configuradas), val_loss sempre 0.0 no MLflow
- **Correção:** `val_days=(0.5, 1.0)` (30 dias de janela)
- **Arquivo:** `src/training/finetune.py`, linhas 128 e 138

### Bug 2 — Flush prematuro do spillover (`dataset.py`)
- **Problema:** Quando um chunk do parquet não tinha dados relevantes, o código fazia flush de TODOS os usuários acumulados, mesmo que tivessem dados chegando em chunks futuros
- **Sintoma:** Usuários com dados esparsos geravam sequências incompletas
- **Correção:** Remover o flush e reset de `prev_uids` quando `df is None`; apenas `continue`
- **Arquivo:** `src/data/dataset.py`, função `__iter__`

### Bug 3 — MLflow sobrescrevia métricas na comparação (`finetune.py`)
- **Problema:** `compare_models_performance` logava todos os modelos no mesmo run MLflow. Além disso, `model_type` era sempre `"zero_shot"` porque a detecção usava `"fine" in model_name.lower()` mas os nomes eram `"FT B"`
- **Correção:** Criar run MLflow separado por modelo; corrigir detecção para `any(k in model_name.upper() for k in ["FT", "FINE"])`
- **Arquivo:** `src/training/finetune.py`, função `compare_models_performance`

---

## 5. Limitações Conhecidas

### 5.1 Cobertura de Usuários
Com `seq_len=720`, apenas 7–33% dos usuários têm dados suficientes para avaliação. Isso é uma limitação intrínseca do dataset (usuários transitam entre cidades) e do `seq_len` longo.

### 5.2 Modelo Possivelmente Subparametrizado
~662K parâmetros para 100K usuários com sequências de 720 pontos pode ser insuficiente. Os top performers do HuMob 2024 usaram Transformers, BERT, e modelos maiores.

### 5.3 Transição entre Cidades não Modelada
O modelo trata cada cidade de forma independente. Na realidade, os usuários transitam entre cidades — uma previsão mais realista deveria considerar a trajetória global do indivíduo.

### 5.4 Conversão km não Verificada Empiricamente
O fator de conversão `1 célula = 0.5 km` foi inferido do código (`ft_avg * 0.5`). Não foi verificado contra a documentação oficial do dataset YJMob.

---

## 6. Estado Atual dos Arquivos

### 6.1 Modelos Salvos (no servidor `cia6`, pasta `~/clnn_v2/`)

| Arquivo | Descrição | Data |
|---------|-----------|------|
| `outputs/models/humob_model_A.pt` | Modelo base treinado (8 épocas, cidade A) | 2026-03-30 |
| `humob_model_finetuned_B.pt` | Fine-tuned cidade B (val_loss=0.00268) | 2026-03-31 |
| `humob_model_finetuned_C.pt` | Fine-tuned cidade C (val_loss=0.00234) | 2026-03-31 |
| `humob_model_finetuned_D.pt` | Fine-tuned cidade D (última da cadeia) | 2026-03-31 |

### 6.2 Artefatos de Avaliação

| Arquivo | Descrição |
|---------|-----------|
| `outputs/eval/pred_gt_all_users.parquet` | 130.007 predições, 5.033 usuários, dias 61–75 |
| `mlruns/` | Todos os runs MLflow (treino, finetuning, comparação, métricas oficiais) |

### 6.3 Scripts Importantes

| Script | Descrição |
|--------|-----------|
| `run_automate.py` | Pipeline principal (YAML → KMeans → treino → FT → avaliação) |
| `eval_comparison.py` | Re-avaliação zero-shot vs FT com MLflow correto |
| `generate_full_eval.py` | Gera pred_gt para todos os usuários disponíveis |
| `evaluate_metrics.py` | ★ Calcula GEO-BLEU e DTW oficiais a partir do parquet existente |

---

## 7. Infraestrutura de Acesso

### 7.1 Servidor de Treinamento

| Item | Valor |
|------|-------|
| Host | `192.168.222.252` (acesso via VPN) |
| Usuário | `anderson.clemente` |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| Repositório | `~/clnn_v2/` |
| Branch atual | `feat/official-metrics` |
| Docker image | `humob:cu128` (~9.2 GB, já construída) |

### 7.2 Acesso (do WSL Ubuntu no Windows)

```bash
# 1. VPN (terminal 1 — deixar rodando)
sudo openvpn --config ~/client.ovpn

# 2. SSH (terminal 2)
wsl -d Ubuntu   # se abrir PowerShell
ssh anderson.clemente@192.168.222.252

# 3. Tmux no servidor
tmux new -s experimento   # nova sessão
tmux attach -t experimento   # retomar sessão existente

# 4. Rodar experimento
cd ~/clnn_v2
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace \
  -e MPLCONFIGDIR=/tmp/mpl \
  -v "$(pwd):/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml

# 5. Calcular métricas oficiais (GEO-BLEU + DTW)
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace \
  -e MPLCONFIGDIR=/tmp/mpl \
  -v "$(pwd):/workspace" humob:cu128 \
  python /workspace/evaluate_metrics.py
```

**Notas de acesso:**
- Anderson NÃO tem sudo no servidor (equipamento compartilhado)
- Não mexer em `/datastore/docker`; não desinstalar drivers de vídeo
- Chave SSH configurada em `~/.ssh/id_ed25519` (sem senha)
- VPN não afeta internet geral (só roteia 192.168.22.0/24 e 192.168.222.0/24)

---

## 8. Próximos Passos para o Artigo

### 8.1 ✅ Implementar Métricas Oficiais — CONCLUÍDO

GEO-BLEU e DTW implementados em `src/utils/metrics.py` e `evaluate_metrics.py`.
Resultados: B=0.1017/174.70 | C=0.0276/236.92 | D=0.1076/156.91

### 8.2 🔴 Crítico — Avaliação no Formato do Challenge

O challenge pede predição **completa** dos dias 61–75 para todos os usuários alvo (3.000 por cidade). O modelo atual prediz sequências amostradas, não o rollout completo.

```
Tarefas:
- Verificar se generate_humob_submission.py gera o formato correto
- Rodar submissão para as 3 cidades e calcular GEO-BLEU/DTW
- Comparar com baseline e top performers do HuMob 2024
```

### 8.3 🟡 Importante — Aumentar Capacidade do Modelo

Os top performers do HuMob 2024 usaram Transformers. Melhorias imediatas sem mudar a arquitetura:

```
Tarefas:
- Aumentar user_emb_dim: 4 → 32
- Aumentar lstm_hidden (verificar valor atual em partial_info.py)
- Aumentar fusion_dim: 8 → 64
- Re-treinar e comparar resultados
```

### 8.4 🟢 Desejável — Ampliar Cobertura de Usuários

Para cobrir mais usuários no artigo:

```
Opção A (sem re-treino): Avaliar usuários com seq_len adaptativo
  - Usuários com 360-719 registros → usar seq_len=360 para avaliação

Opção B (com re-treino): Treinar com seq_len menor (360 ou 240)
  - Mais usuários cobertos, menos contexto por predição
  - Estimativa: 2× mais usuários com seq_len=360
```

### 8.5 🟢 Desejável — Modelar Transição entre Cidades

```
Ideia: Treinar um modelo unificado com todos os usuários de todas as cidades,
usando city_embedding para diferenciar contextos, mas sem fine-tuning separado.
```

---

## 9. Estrutura do Projeto

```
clnn_v2/
├── config/
│   ├── smoke.yaml                  # seq_len=12, n_clusters=16, 1 época (teste rápido)
│   ├── big_smoke.yaml              # seq_len=720, n_clusters=512, 8/5 épocas, split="test"
│   └── exp_full_15d_a4000.yaml     # ★ Config de produção (split="val", num_workers=4)
├── src/
│   ├── data/dataset.py             # HuMobNormalizedDataset (IterableDataset + spillover)
│   ├── models/
│   │   ├── humob_model.py          # HuMobModel + discretize_coordinates
│   │   ├── external_info.py        # ExternalInformationFusionNormalized
│   │   └── partial_info.py         # CoordLSTM (BiLSTM)
│   ├── training/
│   │   ├── train.py                # train_humob_model + evaluate_model + compute_cluster_centers
│   │   ├── finetune.py             # finetune_model + sequential_finetuning + compare_models_performance
│   │   └── pipeline.py             # generate_humob_submission + generate_pred_gt_parquet
│   └── utils/
│       ├── metrics.py              # ★ compute_geobleu() + compute_dtw() (métricas oficiais)
│       ├── mlflow_tracker.py       # HuMobMLflowTracker
│       └── simple_checkpoint.py    # save/load/cleanup checkpoints
├── run_automate.py                 # ★ Ponto de entrada principal
├── evaluate_metrics.py             # ★ GEO-BLEU + DTW sobre pred_gt_all_users.parquet
├── eval_comparison.py              # Re-avaliação MLflow correto (zero-shot vs FT)
├── generate_full_eval.py           # Pred+GT para todos os usuários disponíveis
├── dockerfile                      # pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
└── data/
    └── humob_all_cities_v2_normalized.parquet  # ~4.2 GB
```

---

## 10. Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Framework ML | PyTorch 2.7 |
| GPU | CUDA 12.8 + cuDNN 9 |
| Dados | Parquet (PyArrow) ~4.2 GB |
| Tracking | MLflow (local, `file:./mlruns`) |
| Container | Docker `humob:cu128` (~9.2 GB) |
| Hardware | NVIDIA RTX A4000 (16 GB VRAM) |
| OS servidor | Linux (domínio `orion.net`, UFAL) |
| Métricas oficiais | geobleu (Yahoo Japan) + fastdtw |
