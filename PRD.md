# PRD — CLNN v2: Human Mobility Prediction

> **Última atualização:** 2026-04-15
> **Status:** v2 (BiLSTM + CE loss + Attention pooling) rodou com sucesso na epoch 1. Migração para Transformer (Fase 5) em planejamento.

---

## 1. Contexto do Desafio

### 1.1 O Problema

Predição de mobilidade humana para o **HuMob Challenge 2024** (Yahoo Japan / SIGSPATIAL). Dado um usuário com histórico de localizações nos dias 1–60, prever suas coordenadas (x, y) para cada intervalo de 30 minutos nos dias 61–75.

- **Grade espacial:** 200×200 células (cada célula = 500m × 500m, área total ~100km²)
- **Resolução temporal:** 48 slots/dia × 15 dias = **720 passos** a predizer
- **Métricas oficiais:** GEO-BLEU (similaridade geográfica) e DTW (similaridade temporal)
- **Referência de métricas:** https://github.com/yahoojapan/geobleu
- **Objetivo:** Publicar artigo com resultados competitivos com o top do HuMob 2024 (GEO-BLEU 0.319, DTW 27.15).

### 1.2 Dataset (`humob_all_cities_v2_normalized.parquet`, ~4.5 GB)

| Cidade | Rows | Users (aprox.) | Uso |
|--------|------|----------------|-----|
| A | 111.5M | 100.000 | Treino base |
| B | 24.4M | 25.000 | Fine-tuning |
| C | 18.5M | 20.000 | Fine-tuning |
| D | 7.6M | 6.000 | Fine-tuning |

**Observações críticas:**
- Usuários **transitam entre cidades** — um mesmo `uid` pode aparecer em A num dia e em B em outro. A esparsidade de B/C/D é consequência direta dessa mobilidade, não de ausência de dados.
- Colunas: `uid`, `city_encoded` (0–3), `d_orig` (0–74), `d_norm` [0,1], `t_sin`, `t_cos`, `x_norm`, `y_norm`, `POI_norm` (array 85-D).
- Todos os valores numéricos já normalizados.

---

## 2. Arquitetura Atual (v2)

```
External Information                  Partial Information (BiLSTM)
────────────────────                  ─────────────────────────────
User Emb (64-D)      ─┐               Coord Sequence (T × 2)
City Emb (16-D)      ─┤                        ↓
Day Proj (32-D GELU) ─┼─ Concat ─► Dense(256)  BiLSTM(hidden=128)
Time Proj (32-D GELU)─┤                        ↓ (T × 256)
POI Proj (32-D GELU) ─┘                 Attention Pooling
                                        (temperature = √256)
                                                ↓
                                           LayerNorm(256)
       │                                        │
       └──────────────── WeightedFusion ────────┘
                    (w_r · static + w_e · dynamic)
                               ↓
                       Destination Head
            MLP(256 → 1024 → 512 logits) ─► softmax
                               ↓
                   Σ(p_k · cluster_centers_k)
                               ↓
                     Predicted (x, y) ∈ [0,1]²
```

### 2.1 Fases Implementadas (v2)

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Reparametrização (`lstm_hidden=128`, `user_emb_dim=64`, `fusion_dim=256`) | ✅ |
| 2 | Loss **CE sobre clusters KMeans** + MSE auxiliar (α=0.1) | ✅ |
| 3 | **Attention pooling** sobre todos os outputs do LSTM | ✅ |
| 4 | **Scheduled sampling** (ramp linear 0 → `max_scheduled_p`) | ✅ |
| **5** | **Substituir BiLSTM por Transformer Encoder** | 🔵 Próxima |
| 6 | Cobertura de usuários (seq_len adaptativo para C/D) | 🔵 Futuro |

### 2.2 Correções de Arquitetura (2026-04-14)

Bugs identificados e corrigidos após análise detalhada:

| Problema | Arquivo | Correção |
|----------|---------|----------|
| `criterion` undefined no val loop | `train.py` | `F.mse_loss(pred, target)` direto |
| Attention collapse em seq longas | `partial_info.py` | Scores divididos por √output_dim |
| LSTM output sem normalização | `humob_model.py` | LayerNorm(256) antes do WeightedFusion |
| ReLU zerando 50% dos neurônios | `external_info.py` | ReLU → GELU em todas projeções |

### 2.3 Invariantes Críticos da Arquitetura

- **`fusion_dim` DEVE ser igual a `lstm_hidden * 2`** — `WeightedFusion` tem assert que quebra em runtime caso contrário
- **`poi_proj = Linear(85, poi_out_dim)`** — POI_norm é array de 85 dimensões (coluna `object` no parquet)
- **`n_users >= max(uid) + 1`** — uids vão de 0 a 99999 em todas as cidades
- **Cluster centers são fixos** (`register_buffer`, não treináveis)
- **`tracking_uri`** — use `file:/tmp/mlruns` em smoke para evitar erros de permissão

---

## 3. Status Atual (2026-04-15)

### 3.1 Primeira Epoch de Treino v2 Completada

Config: `config/exp_full_15d_a4000.yaml` (seq_len=336, lr=1e-4, max_scheduled_p=0.3)

| Métrica | Valor |
|---------|-------|
| Train loss (CE+MSE) | **4.1226** |
| Val loss (MSE) | **0.0071** |
| Tempo por epoch | ~9h |
| Batches por epoch | ~24.500 |
| Val batches | 199 (de `max_val_batches=200`) |

**Observações:**
- Val loss **real** pela primeira vez (antes sempre 0.0000 devido ao bug do `criterion`)
- Train loss caindo (baseline uniforme CE sobre 512 clusters seria log(512)=6.24)
- Gradientes **continuam explodindo** (pre-clip entre 11 e 59 em quase todo batch, clip em 2.0)

### 3.2 Problema Estrutural: BPTT em BiLSTM

Com seq_len=336, o LSTM faz backprop por 336 passos. Os gradientes se multiplicam recursivamente pelas matrizes de transição — resultado: **pre-clip entre 11-59 em quase todo batch**, com clip em 2.0 cortando >90% do sinal.

Essa é uma limitação intrínseca de LSTMs sobre sequências longas. Truncated BPTT mitigaria, mas a solução arquiteturalmente correta é migrar para Transformer.

### 3.3 Baseline v1 (Para Comparação)

Arquitetura subparametrizada anterior (`lstm_hidden=4`, `fusion_dim=8`, MSE loss):

| Métrica | Cidade B | Cidade C | Cidade D | Top HuMob 2024 |
|---------|----------|----------|----------|----------------|
| GEO-BLEU | 0.1017 | 0.0276 | 0.1076 | **0.319** |
| DTW | 174.70 | 236.92 | 156.91 | **27.15** |

---

## 4. Fase 5 — Migração para Transformer Encoder

### 4.1 Motivação

1. **Elimina BPTT** — atenção processa todos os timesteps em paralelo, sem propagação recursiva de gradiente. Fim da explosão estrutural.
2. **Teto de performance mais alto** — top HuMob 2024 usou Transformers; literatura consolidada para mobilidade.
3. **Paraleliza no tempo** — LSTM é sequencial; Transformer pode ser 3–5× mais rápido por epoch em GPU.
4. **Padrões de longo alcance** — atenção global captura naturalmente "mesmo horário na semana passada" sem degradação.

### 4.2 Design do `CoordTransformer`

```python
class CoordTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=512, max_len=720,
                 dropout=0.1):
        # 1. Projeção de entrada (2-D → 256-D)
        self.input_proj = nn.Linear(input_size, d_model)

        # 2. Positional encoding aprendível (seq_len fixo por config)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # 3. Encoder: N camadas de self-attention + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True  # pre-norm: mais estável
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. Pooling: mean pooling ou CLS token (TBD)
        self.output_dim = d_model

    def forward(self, x):
        # x: (B, T, 2)
        h = self.input_proj(x) + self.pos_embedding[:, :x.size(1)]
        h = self.encoder(h)              # (B, T, d_model)
        return h.mean(dim=1)             # mean pooling → (B, d_model)
```

### 4.3 Parâmetros Escolhidos e Justificativa

| Parâmetro | Valor | Por quê |
|-----------|-------|---------|
| `d_model` | 256 | Mesmo valor de `fusion_dim` atual — mantém `WeightedFusion` funcionando sem mudança |
| `nhead` | 8 | Head size = 32-D, razoável para coordenadas |
| `num_layers` | 4 | Começar com 4; 6 se houver memória sobrando |
| `dim_feedforward` | 512 | 2× d_model, conservador |
| `dropout` | 0.1 | Baixo; sequências longas e batches grandes ajudam a regularizar |
| `norm_first` | True | Pre-norm: mais estável em treinamento do zero |
| `activation` | GELU | Consistente com resto do modelo |
| Pooling | Mean | Simples e robusto; considerar CLS token em experimento posterior |

### 4.4 Complexidade Computacional

- **Atenção global** em seq_len=336: O(336²) = 113k operações por head
- Com 8 heads × 4 layers = 3.6M operações de atenção por sample
- Batch=64 → 230M operações por batch — cabe facilmente em 16GB VRAM
- **Memória de atenção:** 336² × 8 × 4 bytes × 64 batch ≈ 230MB por layer → ~1GB total, aceitável

Se no futuro quisermos voltar a `seq_len=720`, aí entra **atenção local** (janela de 48 = 1 dia), que é O(720 × 48) em vez de O(720²).

### 4.5 Arquivos a Modificar

| Arquivo | Mudança |
|---------|---------|
| `src/models/partial_info_transformer.py` | **Novo** — classe `CoordTransformer` |
| `src/models/humob_model.py` | Condicional `encoder_type` no `__init__` para escolher LSTM ou Transformer |
| `config/exp_full_15d_a4000.yaml` | Adicionar bloco `encoder:` com hiperparâmetros do Transformer |
| `run_automate.py` | Ler `encoder_type` e hiperparâmetros do YAML |
| `PRD.md` e `CLAUDE.md` | Documentar a nova arquitetura |

### 4.6 Estratégia de Migração

1. **Implementar `CoordTransformer`** em paralelo ao `CoordLSTM` (não remover ainda)
2. **Config condicional:** `encoder_type: "lstm" | "transformer"` no YAML
3. **Rodar smoke** com Transformer para validar forward/backward/loss
4. **Rodar experimento completo com Transformer** (`n_epochs: 15`, finetune como sempre)
5. **Comparar GEO-BLEU/DTW** com v2 BiLSTM (epoch 1 atual como baseline)
6. **Se Transformer vencer** — remover CoordLSTM em um commit separado

### 4.7 Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Overfitting em seq longas com poucos dados por user | Dropout 0.1 + weight decay; monitorar val_loss |
| Positional encoding fixo não generaliza para seq_len variáveis | Usar seq_len fixo no config; avaliar com mesmo seq_len |
| Transformer não aprende padrões curtos tão bem quanto LSTM | Mean pooling sobre toda sequência captura ambos |
| Memória explodir com batch grande | Começar com batch=64 (igual atual); `torch.compile` se precisar |

---

## 5. Próximas Fases (Pós-Transformer)

### Fase 6 — Cobertura de Usuários em C e D

Com seq_len=336, City D (média 76 pts/user) ainda filtra maioria dos usuários. Opções:

- **Seq_len adaptativo na avaliação** — usar 144 para users com <336 pontos
- **Re-treino multi-seq-len** — modelo aceita qualquer seq_len fixo ≥ 48
- **Modelo unificado multi-cidade** — treinar sem fine-tuning usando `city_emb` como diferenciador

### Fase 7 — Ablation Study para o Artigo

- Transformer vs BiLSTM (same hyperparams)
- Com/sem attention pooling
- CE vs MSE loss
- Scheduled sampling on/off
- Com/sem POI features

---

## 6. Bugs Conhecidos e Histórico

### Bugs Ativos (não corrigidos)

- **BPTT estrutural em BiLSTM** — gradientes explodem (pre=11-59, clip=2.0). **Solução: Fase 5 (Transformer).**
- **Cobertura em City D** — 76 pts/user, seq_len=336 requer 337. Solução: Fase 6.

### Bugs Corrigidos em 2026-04-14 (sessão de diagnóstico)

1. **Val loop `criterion` undefined** — `criterion` só existia em `evaluate_model`, não em `train_humob_model`. Todas as batches de val lançavam `NameError` silenciosamente → `val_count=0` → `val_loss=0.0` → early stopping falso.
2. **Attention collapse** — softmax sem temperatura sobre 336 timesteps colapsa em 1-2 posições.
3. **LSTM output sem LayerNorm** — magnitude de `dyn_emb` divergia de `static_red` (que tinha LayerNorm), desbalanceando o WeightedFusion.
4. **ReLU matando features temporais/POI** — Xavier init + features em [0,1] zera metade dos neurônios.
5. **Config `sequence_length=720`** — descartava maioria dos users de C/D (média 76 pts/user em D). Reduzido para 336.
6. **Config `max_scheduled_p=0.7`** — agressivo demais treinando do zero. Reduzido para 0.3.
7. **Config `learning_rate=3e-5`** — conservador pós-fixes de arquitetura. Aumentado para 1e-4.

### Bugs Corrigidos Anteriormente

- **`val_days` insuficiente no finetune** — `(0.8, 1.0)` = 15 dias = seq_len → zero sequências. Corrigido para `(0.5, 1.0)`.
- **Flush prematuro do spillover** em `dataset.py` — usuários com dados esparsos perdiam sequências.
- **MLflow sobrescrevia métricas** — `compare_models_performance` agrupava todos os modelos no mesmo run.

---

## 7. Infraestrutura

### 7.1 Servidor de Treinamento (cia6)

| Item | Valor |
|------|-------|
| Host | `192.168.222.252` (acesso via VPN) |
| Usuário | `anderson.clemente` (**sem sudo**) |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| Repositório | `~/clnn_v2/` |
| Docker image | `humob:cu128` (~9.2 GB, já construída) |

### 7.2 Execução

```bash
# VPN (terminal 1, deixar rodando)
sudo openvpn --config ~/client.ovpn

# SSH + tmux (terminal 2)
ssh anderson.clemente@192.168.222.252
tmux new -s humob_full

# Docker SEM sudo (usuário está no grupo docker)
cd ~/clnn_v2
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml

# Avaliação oficial (GEO-BLEU + DTW)
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/evaluate_metrics.py
```

### 7.3 Permissões (gotchas)

- Container roda como `laccan` (UID 1000, appuser da imagem)
- Arquivos no host criados por `anderson.clemente` (UID diferente) precisam `chmod o+w`
- `mlruns/` deve ser 757 ou compatível para o container escrever
- Checkpoints em `outputs/models/checkpoints/` ficam owned por `laccan` — o container pode sobrescrever nas próximas epochs

---

## 8. Estrutura do Projeto

```
clnn_v2/
├── config/
│   ├── smoke_v2.yaml               # Teste rápido (seq_len=12, 2 epochs, checkpoints)
│   └── exp_full_15d_a4000.yaml     # ★ Config de produção
├── src/
│   ├── data/dataset.py             # HuMobNormalizedDataset (IterableDataset + spillover)
│   ├── models/
│   │   ├── humob_model.py          # HuMobModel (escolhe encoder via config)
│   │   ├── external_info.py        # Embeddings + projeções GELU
│   │   ├── partial_info.py         # CoordLSTM com attention pooling + temperatura
│   │   └── partial_info_transformer.py  # (a criar — Fase 5)
│   ├── training/
│   │   ├── train.py                # train_humob_model com CE+MSE loss
│   │   ├── finetune.py             # Fine-tuning sequencial B→C→D
│   │   └── pipeline.py             # generate_humob_submission + pred_gt_parquet
│   └── utils/
│       ├── metrics.py              # compute_geobleu() + compute_dtw()
│       ├── mlflow_tracker.py       # HuMobMLflowTracker
│       └── simple_checkpoint.py    # save/load/cleanup
├── run_automate.py                 # ★ Ponto de entrada (YAML → pipeline completo)
├── evaluate_metrics.py             # ★ GEO-BLEU + DTW a partir do pred_gt_all_users.parquet
├── PRD.md                          # ★ Este arquivo
├── CLAUDE.md                       # Contexto rápido para novas sessões
├── dockerfile                      # pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
└── data/
    └── humob_all_cities_v2_normalized.parquet  # ~4.2 GB, normalizado
```

---

## 9. Stack Tecnológica

| Componente | Tecnologia |
|------------|------------|
| Framework ML | PyTorch 2.7 |
| GPU | CUDA 12.8 + cuDNN 9 |
| Dados | Parquet (PyArrow) ~4.2 GB |
| Tracking | MLflow local (`file:./mlruns`) |
| Container | Docker `humob:cu128` (~9.2 GB) |
| Hardware | NVIDIA RTX A4000 (16 GB VRAM) |
| Métricas | geobleu (Yahoo Japan) + fastdtw |

---

## 10. Convenções

- **Sempre rodar via Docker** — nunca instalar dependências direto no servidor
- **Dados em [0,1]** — não desnormalizar antes do modelo
- **Coordenadas discretas** via `discretize_coordinates()` para submissão (grade 0–199)
- **Branch única `main`** — trabalho em feature branches, merge ao concluir
- **Checkpoints** em `outputs/models/checkpoints/` — mantém os 3 últimos
- **Nunca executar experimentos remotamente via SSH na sessão** — o usuário roda em seu próprio tmux
