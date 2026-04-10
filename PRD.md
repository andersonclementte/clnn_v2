# PRD — CLNN v2: Human Mobility Prediction

> **Última atualização:** 2026-04-10
> **Contexto:** Este documento consolida todo o conhecimento adquirido até abril de 2026, incluindo bugs corrigidos, resultados reais com métricas oficiais, diagnóstico do modelo atual e plano de implementação para as próximas melhorias. Escrito para que qualquer LLM ou desenvolvedor possa retomar o trabalho sem perda de contexto.

---

## 1. Contexto do Desafio

### 1.1 Histórico dos Challenges

| Edição | Nome | Dataset | Cidades | Métricas Oficiais |
|--------|------|---------|---------|-------------------|
| 2023 | HuMob Challenge 2023 | YJMob100K (cidade única, Japão) | 1 | GEO-BLEU, DTW |
| 2024 | HuMob Challenge 2024 | YJMob multi-cidade (~4.5 GB) | A, B, C, D | GEO-BLEU, DTW |
| 2025 | SIGSPATIAL GIS CUP 2025 | LY Corporation (~160 GB) | Multi-cidades | GEO-BLEU, DTW |

**Este projeto usa o dataset do HuMob 2024** (cidades A, B, C, D), com objetivo de publicar um artigo.

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

✅ **GEO-BLEU e DTW implementados** (`src/utils/metrics.py`, `evaluate_metrics.py`).

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

### 2.2 Parâmetros do Modelo (versão atual — v1)

| Parâmetro | Valor | Problema identificado |
|-----------|-------|----------------------|
| `sequence_length` | 720 | 15 dias × 48 slots |
| `n_clusters` | 512 | KMeans sobre coordenadas da cidade A |
| `n_users` | 100.000 | Total de usuários no dataset |
| `lstm_hidden` | **4** (bidirecional → 8-D) | LSTM de 160 params para sequências de 720 passos |
| `user_emb_dim` | **4** | 400K params gastos em 4 dimensões para 100K usuários |
| `fusion_dim` | **8** | Toda trajetória + contexto comprimida em 8 números |
| `poi_out_dim` | **4** | 85 features POI → 4-D é perda brutal |
| Total params | ~662K | 60% são embedding de usuário; LSTM tem apenas 160 params |

### 2.3 Pipeline de Treinamento (v1)

1. **KMeans** — Calcula 512 centros de cluster sobre coordenadas da cidade A
2. **Treino base** — 8 épocas na cidade A (AdamW, ReduceLROnPlateau, AMP)
3. **Fine-tuning sequencial** — B → C → D, cada um partindo do checkpoint anterior (5 épocas, CosineAnnealingLR)
4. **Avaliação** — Compara zero-shot vs fine-tuned + gera Pred+GT pairs

---

## 3. Resultados Obtidos (Experimento: exp_full_15d_a4000, março 2026)

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

**Por que só 5–33% dos usuários?** Com `seq_len=720`, o usuário precisa ter 720 registros consecutivos na cidade. Como os usuários transitam entre cidades, a maioria só aparece esporadicamente em B, C, D.

### 3.5 Métricas Oficiais — GEO-BLEU e DTW (rodado em 2026-03-31)

Avaliado sobre `pred_gt_all_users.parquet` (130.007 predições, 5.033 usuários):

| Cidade | Usuários | GEO-BLEU | DTW | GEO-BLEU top 2024 | DTW top 2024 |
|--------|----------|----------|-----|-------------------|--------------|
| B | 2.334 | **0.1017** | **174.70** | 0.319 | 27.15 |
| C | 1.406 | **0.0276** | **236.92** | 0.319 | 27.15 |
| D | 1.970 | **0.1076** | **156.91** | 0.319 | 27.15 |

**Interpretação:**
- B e D atingem GEO-BLEU ~0.10 — baseline razoável, mas 3× abaixo do top
- C apresenta GEO-BLEU 0.027, consistente com cell_error 2× maior observado
- DTW alto (~175–237 vs 27 do top) indica desvios temporais significativos
- A diferença para o top é esperada: líderes do HuMob 2024 usaram Transformers com arquiteturas muito maiores

---

## 4. Diagnóstico do Modelo Atual

Análise realizada em 2026-04-10 após inspeção dos pesos salvos (`humob_model_finetuned_B.pt`).

### 4.1 Modelo Severamente Subparametrizado

| Camada | Tamanho atual | Tamanho adequado | Impacto |
|--------|--------------|-----------------|---------|
| `lstm_hidden` | **4** (BiLSTM → 8-D) | 64–128 | LSTM com 160 params não captura 720 passos |
| `user_emb_dim` | **4** para 100K usuários | 32–64 | 4-D não representa identidade do usuário |
| `fusion_dim` | **8** | 64–128 | Trajetória inteira + contexto em 8 números |
| `poi_out_dim` | **4** | 16–32 | 85 features POI perdidas |

Os 662K params são enganosos: **400K (60%) são a embedding de usuário `(100000, 4)`** — que tem quase nenhum poder expressivo com apenas 4 dimensões. O LSTM real tem **160 params** apenas.

### 4.2 Loss Function Desalinhada das Métricas

O modelo treina com `MSELoss` em coordenadas `[0,1]`, mas é avaliado com **GEO-BLEU e DTW** que medem similaridade de trajetórias. Não há sinal de treinamento sobre consistência temporal de sequências.

Além disso, o modelo já gera **logits sobre 512 clusters** e depois converte para coordenadas — mas é treinado com MSE nas coordenadas finais, não com cross-entropy nos clusters. O sinal de treinamento é indireto.

### 4.3 LSTM Desperdiça o Contexto da Sequência

`CoordLSTM` usa apenas `h_n[-1]` (último hidden state) para uma sequência de 720 passos. Os 719 hidden states intermediários — que contêm os padrões de movimento ao longo do tempo — são completamente descartados.

### 4.4 Head MLP Desproporcional

A head `8 → 500 → 512` tem **262K params** para uma entrada de **8 dimensões**. É o maior componente do modelo depois das embeddings, mas alimentado por um vetor minúsculo.

### 4.5 Desalinhamento Treino/Avaliação

O modelo treina passo-a-passo (`forward_single_step`), mas é avaliado com rollout autoregressivo de 720 passos. Erros se acumulam durante a avaliação de forma que o modelo nunca viu durante o treino.

---

## 5. Plano de Implementação — v2

### Fase 1 — Reparametrização (🔴 Crítico — sem mudar arquitetura)

**O maior ganho com menor risco.** Apenas mudança de hiperparâmetros no YAML:

```yaml
# config/exp_v2_reparametrizado.yaml
model:
  user_emb_dim: 32      # era 4  → mais expressividade por usuário
  city_emb_dim: 8       # era 4
  temporal_dim: 16      # era 8
  poi_out_dim: 16       # era 4  → aproveita os 85 features POI
  lstm_hidden: 64       # era 4  → bidirecional produz 128-D
  fusion_dim: 128       # era 8  → head MLP tem entrada decente
  n_clusters: 512       # mantém
```

**Impacto estimado:** O modelo passa de ~2.4M params funcionais (excluindo embedding de usuário que é inutilizada), com o LSTM capaz de modelar dependências temporais reais.

**Arquivos a editar:** `config/exp_full_15d_a4000.yaml` + adicionar args ao `HuMobModel.__init__` lido do YAML.

### Fase 2 — Loss Function (🔴 Crítico — alto impacto)

Substituir `MSELoss` por **cross-entropy sobre os índices de cluster** como loss primária:

```python
# src/training/train.py — dentro do loop de treino
# 1. Obter logits (antes da softmax) — requer expor no modelo
cluster_logits = model.destination_head.mlp(fused)  # (B, n_clusters)

# 2. Calcular índice do cluster mais próximo do target
target_cluster = find_nearest_cluster(target_coords, cluster_centers)  # (B,)

# 3. Loss principal: CE nos clusters + MSE auxiliar nas coordenadas
loss_ce = F.cross_entropy(cluster_logits, target_cluster)
loss_mse = F.mse_loss(pred_coords, target_coords)
loss = loss_ce + 0.1 * loss_mse
```

**Por que isso melhora GEO-BLEU/DTW:** O modelo passa a aprender explicitamente "em qual região geográfica o usuário estará", que é o que GEO-BLEU mede. O MSE sobre coordenadas contínuas é um proxy fraco para esse objetivo.

**Arquivos a editar:** `src/models/humob_model.py` (expor logits), `src/training/train.py` (loss), `src/training/finetune.py` (loss).

### Fase 3 — Attention Pooling no LSTM (🟡 Importante)

Em vez de descartar 719 hidden states, adicionar atenção aprendível sobre toda a sequência:

```python
# src/models/partial_info.py — substituir último hidden por attention pooling
class CoordLSTM(nn.Module):
    def __init__(self, ...):
        ...
        self.attention = nn.Linear(hidden_size * num_directions, 1)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)      # output: (B, 720, hidden*2)
        scores = self.attention(output)        # (B, 720, 1)
        weights = F.softmax(scores, dim=1)     # (B, 720, 1)
        context = (weights * output).sum(dim=1)  # (B, hidden*2)
        return context
```

**Por que isso melhora:** O modelo pode aprender a focar em partes relevantes da trajetória (ex: padrões do dia anterior, horários de pico) em vez de usar apenas o último passo.

**Arquivo a editar:** `src/models/partial_info.py`.

### Fase 4 — Scheduled Sampling (🟡 Importante)

Reduz o mismatch entre treino (coordenadas reais como input) e avaliação (coordenadas preditas como input):

```python
# Durante treino: com probabilidade p_scheduled, usar predição anterior
# como input em vez da coordenada real. p cresce de 0→0.5 ao longo das épocas.
if random.random() < p_scheduled:
    next_input = pred.detach()  # usa predição própria
else:
    next_input = ground_truth   # usa dado real (teacher forcing)
```

**Arquivo a editar:** `src/training/train.py`.

### Fase 5 — Arquitetura Transformer (🟢 Desejável — maior ganho potencial)

Substituir o BiLSTM por um **Transformer Encoder com atenção local** (janela de 48 slots = 1 dia). Os top performers do HuMob 2024 usaram exatamente essa abordagem.

```
Coord Sequence (720 × 2)
    ↓ positional encoding
Transformer Encoder (local attention, window=48)
    ↓ [CLS] token ou mean pooling
Context Vector (256-D)
```

**Considerações:** Sequências de 720 passos com atenção global são O(720²) — pesado. Usar atenção local com janela de 48 (1 dia) ou atenção esparsa é essencial.

**Arquivo a criar:** `src/models/partial_info_transformer.py`.

### Fase 6 — Ampliar Cobertura de Usuários (🟢 Desejável)

Para cobrir mais usuários no artigo (atualmente 7–33%):

```
Opção A (sem re-treino): seq_len adaptativo na avaliação
  - Usuários com 360-719 registros → usar seq_len=360
  - Limitação: comparação direta não é possível

Opção B (re-treino com seq_len=360):
  - Estimativa: 2× mais usuários cobertos
  - Menos contexto por predição, potencialmente menor qualidade

Opção C (modelo unificado multi-cidade):
  - Treinar sem fine-tuning separado, usando city_embedding
  - Captura padrões de transição entre cidades
```

---

## 6. Bugs Corrigidos (histórico importante)

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
- **Problema:** `compare_models_performance` logava todos os modelos no mesmo run MLflow. `model_type` era sempre `"zero_shot"` porque a detecção usava `"fine" in model_name.lower()` mas os nomes eram `"FT B"`
- **Correção:** Criar run MLflow separado por modelo; corrigir detecção para `any(k in model_name.upper() for k in ["FT", "FINE"])`
- **Arquivo:** `src/training/finetune.py`, função `compare_models_performance`

---

## 7. Limitações Conhecidas

### 7.1 Cobertura de Usuários
Com `seq_len=720`, apenas 7–33% dos usuários têm dados suficientes para avaliação. Limitação intrínseca do dataset (usuários transitam entre cidades).

### 7.2 Transição entre Cidades não Modelada
O modelo trata cada cidade de forma independente. Na realidade, os usuários transitam entre cidades — uma previsão mais realista deveria considerar a trajetória global.

### 7.3 Conversão km não Verificada Empiricamente
O fator de conversão `1 célula = 0.5 km` foi inferido do código (`ft_avg * 0.5`). Não foi verificado contra a documentação oficial do dataset YJMob.

---

## 8. Estado Atual dos Arquivos

### 8.1 Modelos Salvos (no servidor `cia6`, pasta `~/clnn_v2/`)

| Arquivo | Descrição | Data |
|---------|-----------|------|
| `outputs/models/humob_model_A.pt` | Modelo base treinado (8 épocas, cidade A) | 2026-03-30 |
| `humob_model_finetuned_B.pt` | Fine-tuned cidade B (val_loss=0.00268) | 2026-03-31 |
| `humob_model_finetuned_C.pt` | Fine-tuned cidade C (val_loss=0.00234) | 2026-03-31 |
| `humob_model_finetuned_D.pt` | Fine-tuned cidade D (última da cadeia) | 2026-03-31 |

### 8.2 Artefatos de Avaliação

| Arquivo | Descrição |
|---------|-----------|
| `outputs/eval/pred_gt_all_users.parquet` | 130.007 predições, 5.033 usuários, dias 61–75 |
| `mlruns/` | Todos os runs MLflow (treino, finetuning, comparação, métricas oficiais) |

### 8.3 Scripts Importantes

| Script | Descrição |
|--------|-----------|
| `run_automate.py` | Pipeline principal (YAML → KMeans → treino → FT → avaliação) |
| `eval_comparison.py` | Re-avaliação zero-shot vs FT com MLflow correto |
| `generate_full_eval.py` | Gera pred_gt para todos os usuários disponíveis |
| `evaluate_metrics.py` | ★ Calcula GEO-BLEU e DTW oficiais a partir do parquet existente |

---

## 9. Infraestrutura de Acesso

### 9.1 Servidor de Treinamento

| Item | Valor |
|------|-------|
| Host | `192.168.222.252` (acesso via VPN) |
| Usuário | `anderson.clemente` |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| Repositório | `~/clnn_v2/` |
| Branch principal | `main` |
| Docker image | `humob:cu128` (~9.2 GB, já construída) |

### 9.2 Acesso (do WSL Ubuntu no Windows)

```bash
# 1. VPN (terminal 1 — deixar rodando)
sudo openvpn --config ~/client.ovpn

# 2. SSH (terminal 2)
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

## 10. Estrutura do Projeto

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
│   │   └── partial_info.py         # CoordLSTM (BiLSTM) — Fase 3: adicionar attention pooling
│   ├── training/
│   │   ├── train.py                # train_humob_model + evaluate_model — Fase 2: loss CE
│   │   ├── finetune.py             # finetune_model + sequential_finetuning
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

## 11. Stack Tecnológica

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
