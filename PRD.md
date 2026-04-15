# PRD — CLNN v2: Human Mobility Prediction

> **Última atualização:** 2026-04-15
> **Status:** v2 BiLSTM em treino (epoch 2, City A). Transformer Encoder implementado na branch `feature/transformer-encoder`, aguardando conclusão do baseline BiLSTM para comparação.

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

O modelo suporta **dois encoders intercambiáveis** via `encoder_type` no YAML. O resto do pipeline (external info, WeightedFusion, destination head) é idêntico — condição necessária para comparação justa.

```
External Information                  Partial Information (encoder configurável)
────────────────────                  ─────────────────────────────────────────
User Emb (64-D)      ─┐               Coord Sequence (T × 2)
City Emb (16-D)      ─┤                        ↓
Day Proj (32-D GELU) ─┼─ Concat ─►    ┌────────┴────────┐
Time Proj (32-D GELU)─┤               │                 │
POI Proj (32-D GELU) ─┘          CoordLSTM       CoordTransformer
      ↓                          (hidden=128)    (d_model=256, 4 layers,
  Dense(256)                     BiLSTM → 256-D   8 heads, ff=512, pre-norm GELU)
      ↓                                │                 │
      ↓                          Attention Pool    Mean Pooling
      ↓                          (temp = √256)    sobre tokens (T)
      ↓                                └────────┬────────┘
      ↓                                         ↓
      ↓                                  LayerNorm(256)
      ↓                                         ↓
      └────────────── WeightedFusion ───────────┘
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
| **5** | **Transformer Encoder como alternativa ao BiLSTM (`encoder_type` switch)** | ✅ Implementado — aguardando run de comparação |
| 6 | Cobertura de usuários (seq_len adaptativo para C/D) | 🔵 Futuro |

### 2.2 Correções de Arquitetura (2026-04-14)

| Problema | Arquivo | Correção |
|----------|---------|----------|
| `criterion` undefined no val loop | `train.py` | `F.mse_loss(pred, target)` direto |
| Attention collapse em seq longas | `partial_info.py` | Scores divididos por √output_dim |
| LSTM output sem normalização | `humob_model.py` | LayerNorm(256) antes do WeightedFusion |
| ReLU zerando 50% dos neurônios | `external_info.py` | ReLU → GELU em todas projeções |

### 2.3 Invariantes Críticos da Arquitetura

- **`fusion_dim` DEVE ser igual à saída do encoder** — `WeightedFusion` tem assert que quebra em runtime. Para BiLSTM: `fusion_dim == lstm_hidden * 2`. Para Transformer: `fusion_dim == d_model` (por construção `d_model = fusion_dim`).
- **`poi_proj = Linear(85, poi_out_dim)`** — POI_norm é array de 85 dimensões (coluna `object` no parquet)
- **`n_users >= max(uid) + 1`** — uids vão de 0 a 99999 em todas as cidades
- **Cluster centers são fixos** (`register_buffer`, não treináveis)
- **`tracking_uri`** — use `file:/tmp/mlruns` em smoke para evitar erros de permissão
- **`encoder_type` + hiperparâmetros do Transformer** são salvos no `config` do checkpoint — fine-tuning e avaliação reconstroem o encoder correto automaticamente.

---

## 3. Status Atual (2026-04-15)

### 3.1 Treino BiLSTM em andamento (baseline para comparação)

Config: `config/exp_full_15d_a4000.yaml` (seq_len=336, lr=1e-4, max_scheduled_p=0.3)

**Epoch 1 completada:**
| Métrica | Valor |
|---------|-------|
| Train loss (CE+MSE) | **4.1226** |
| Val loss (MSE) | **0.0071** |
| Tempo por epoch | ~9h |
| Batches por epoch | ~24.500 |

**Epoch 2 em andamento (snapshot ~12.9k/24.5k it):**
- Train loss oscilando em ~3.3–4.3 (running avg = 3.41)
- Val loss real pela primeira vez (antes 0.0000 pelo bug do `criterion`)
- Baseline uniforme CE sobre 512 clusters = log(512) ≈ 6.24 → o modelo está aprendendo, mas desacelerando

### 3.2 Gargalo estrutural: BPTT em BiLSTM

Com seq_len=336, o LSTM faz backprop por 336 passos. Os gradientes se multiplicam recursivamente pelas matrizes de transição — observado empiricamente:

- **Pre-clip entre 12–59** em virtualmente todo batch da epoch 2
- **GradNorm pós-clip saturada em 2.0** (valor do clip) constantemente
- Clip cortando >90% do sinal em praticamente todo step

Essa saturação é a principal motivação para avaliar o Transformer Encoder: atenção processa todos os timesteps em paralelo, sem recursão no backward → elimina a fonte estrutural da explosão.

### 3.3 Baseline v1 (subparametrizado, referência histórica)

| Métrica | Cidade B | Cidade C | Cidade D | Top HuMob 2024 |
|---------|----------|----------|----------|----------------|
| GEO-BLEU | 0.1017 | 0.0276 | 0.1076 | **0.319** |
| DTW | 174.70 | 236.92 | 156.91 | **27.15** |

---

## 4. Fase 5 — Comparação BiLSTM vs Transformer Encoder

### 4.1 Decisão de Design: Comparação, Não Substituição

O Transformer **não substitui** o BiLSTM — é implementado como **encoder alternativo** selecionável via YAML. Motivos:

1. **Ablação obrigatória para o paper.** A pergunta "quanto o Transformer realmente ajudou?" exige os dois números lado a lado com mesma config, mesmo seed, mesmo dataset.
2. **Narrativa empírica.** "Identificamos BPTT saturando o clip (evidência: grad pre-clip 12–59 em toda epoch 2) e migramos para atenção" é história completa com os dois treinos; é anedota com um só.
3. **Fallback sem custo.** Se Transformer overfittar em B/C/D (dados esparsos), o BiLSTM continua disponível sem reversão de código.
4. **Aproveita trabalho já feito.** Quase 2 epochs de BiLSTM já foram pagas; descartar = desperdício.

### 4.2 Design do `CoordTransformer`

```python
class CoordTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=256, nhead=8,
                 num_layers=4, dim_feedforward=512, max_len=720,
                 dropout=0.1):
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_dim = d_model

    def forward(self, x):                               # (B, T, 2)
        h = self.input_proj(x) + self.pos_embedding[:, :x.size(1)]
        h = self.encoder(h)                             # (B, T, d_model)
        return h.mean(dim=1)                            # (B, d_model)
```

### 4.3 Parâmetros Escolhidos

| Parâmetro | Valor | Por quê |
|-----------|-------|---------|
| `d_model` | 256 | Igual a `fusion_dim` — mantém `WeightedFusion` sem mudança |
| `nhead` | 8 | Head size = 32-D, adequado |
| `num_layers` | 4 | Começar com 4; 6 se sobrar memória |
| `dim_feedforward` | 512 | 2× d_model, conservador |
| `dropout` | 0.1 | Baixo; batches grandes regularizam |
| `norm_first` | True | Pre-norm: mais estável treinando do zero |
| `activation` | GELU | Consistente com o resto do modelo |
| Pooling | Mean | Simples; CLS token em experimento futuro |

### 4.4 Plano de Comparação

**Precondições (antes de parar o BiLSTM):**
1. Deixar BiLSTM completar suficientes epochs para ser baseline sólido — idealmente até convergência ou platô claro (val_loss sem melhoria em 2–3 epochs seguidas).
2. Rodar fine-tuning automático em B → C → D (pipeline já faz).
3. Gerar `pred_gt_all_users.parquet` e calcular GEO-BLEU + DTW para A/B/C/D com `evaluate_metrics.py`.
4. **Salvar artefatos com nomes distintos:**
   ```
   outputs/models/humob_model_A_lstm_baseline.pt
   outputs/models/humob_model_finetuned_{B,C,D}_lstm_baseline.pt
   outputs/eval/pred_gt_all_users_lstm.parquet
   outputs/eval/metrics_lstm.json
   ```
5. Taggear o run MLflow com `encoder_type=lstm`.

**Rodada Transformer (mesma config, só trocando encoder):**
1. Checkout `feature/transformer-encoder` e confirmar `encoder.type: transformer` no YAML.
2. **Smoke test primeiro** (`config/smoke_v2.yaml` com `encoder.type: transformer`) — valida forward/backward/save em minutos, evita descobrir bug de shape após 1h de setup.
3. Limpar checkpoints do LSTM (eles são incompatíveis):
   ```bash
   rm -f outputs/models/humob_model_A.pt outputs/models/checkpoints/*.pt
   ```
4. Rodar pipeline completo com mesmo seed, mesmo dataset, mesmo lr/bs/seq_len/scheduled_sampling.
5. Gerar artefatos `*_transformer.*` análogos.

**Comparação final (tabela principal do paper):**

| Modelo | City A GEO-BLEU | City A DTW | City B GEO-BLEU | City B DTW | City C GEO-BLEU | City C DTW | City D GEO-BLEU | City D DTW | Grad satura clip? | Tempo/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| BiLSTM (v2)    | ? | ? | ? | ? | ? | ? | ? | ? | ✓ | ~9h |
| Transformer    | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Top HuMob 2024 | — | — | — | — | — | — | 0.319 | 27.15 | — | — |

**Critérios de decisão para o paper:**
- Se Transformer vence em ≥3 das 4 cidades com margem > 5% relativo → arquitetura principal proposta.
- Se empata ou perde → reportar ambos como "arquiteturas equivalentes; BiLSTM suficiente para este domínio" (resultado negativo ainda é resultado).
- Figura obrigatória: **curva de GradNorm pre-clip × step** dos dois — demonstra visualmente o gargalo resolvido.

### 4.5 Complexidade Computacional

- Atenção global em seq_len=336: O(336²) ≈ 113k ops por head
- 8 heads × 4 layers × 64 batch ≈ 230M ops/batch — cabe em 16GB VRAM
- Memória de atenção: ~1GB total, aceitável
- Para seq_len=720 futuro, considerar atenção local (janela=48 = 1 dia), O(720×48)

### 4.6 Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Transformer não testado end-to-end | Smoke test obrigatório antes do full run |
| Overfitting em seq longas com poucos dados por user | Dropout 0.1 + weight decay; monitorar val_loss |
| Positional encoding aprendido não generaliza seq variáveis | seq_len fixo por config; avaliação com mesmo seq_len |
| Comparação injusta por seed/timing diferente | Mesma seed, mesma config YAML, mesmo hardware |

---

## 5. Próximas Fases

### Fase 6 — Cobertura de Usuários em C e D

Com seq_len=336, City D (média 76 pts/user) ainda filtra maioria dos usuários. Opções:
- Seq_len adaptativo na avaliação (144 para users <336 pontos)
- Re-treino multi-seq-len (modelo aceita qualquer seq_len ≥ 48)
- Modelo unificado multi-cidade (sem FT, usando `city_emb`)

### Fase 7 — Ablações Adicionais para o Paper

- Com/sem attention pooling (no BiLSTM)
- CE vs MSE loss
- Scheduled sampling on/off
- Com/sem POI features
- Transformer: 4 vs 6 layers, mean vs CLS pooling

---

## 6. Bugs Conhecidos e Histórico

### Bugs Ativos

- **BPTT estrutural em BiLSTM** — pre-clip 12–59, GradNorm saturada em 2.0. É o gargalo que motiva a comparação com Transformer (Fase 5).
- **Cobertura em City D** — 76 pts/user vs seq_len=336. Solução: Fase 6.

### Bugs Corrigidos em 2026-04-14

1. **Val loop `criterion` undefined** — `criterion` só existia em `evaluate_model`. Val silenciosamente lançava `NameError` → `val_loss=0.0`.
2. **Attention collapse** — softmax sem temperatura colapsa em 1–2 posições.
3. **LSTM output sem LayerNorm** — magnitude divergia de `static_red`, desbalanceando WeightedFusion.
4. **ReLU matando features temporais/POI** — Xavier + inputs em [0,1] zera metade dos neurônios.
5. **`sequence_length=720`** — descartava maioria de C/D. Reduzido para 336.
6. **`max_scheduled_p=0.7`** — agressivo do zero. Reduzido para 0.3.
7. **`learning_rate=3e-5`** — conservador pós-fixes. Aumentado para 1e-4.

### Bugs Corrigidos Anteriormente

- `val_days=(0.8, 1.0)` no finetune → zero sequências. Corrigido para `(0.5, 1.0)`.
- Flush prematuro do spillover em `dataset.py`.
- MLflow sobrescrevia métricas em `compare_models_performance`.

---

## 7. Infraestrutura

### 7.1 Servidor cia6

| Item | Valor |
|------|-------|
| Host | `192.168.222.252` (VPN) |
| Usuário | `anderson.clemente` (**sem sudo**) |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| Repositório | `~/clnn_v2/` |
| Docker image | `humob:cu128` (~9.2 GB) |
| Branch Transformer | `feature/transformer-encoder` |

### 7.2 Execução

```bash
# VPN (terminal 1)
sudo openvpn --config ~/client.ovpn

# SSH + tmux (terminal 2)
ssh anderson.clemente@192.168.222.252
tmux new -s humob_full

# Treino BiLSTM (branch main)
cd ~/clnn_v2
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml

# Treino Transformer (branch feature/transformer-encoder, mesma config com encoder.type=transformer)
git checkout feature/transformer-encoder
rm -f outputs/models/humob_model_A.pt outputs/models/checkpoints/*.pt
# (mesmo comando docker run acima)

# Avaliação oficial
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace -v "$PWD:/workspace" humob:cu128 \
  python /workspace/evaluate_metrics.py
```

### 7.3 Permissões

- Container roda como `laccan` (UID 1000); host como `anderson.clemente` → sempre rodar com `-u $(id -u):$(id -g)`
- `mlruns/` em 757 funciona
- Checkpoints anteriores owned por `laccan` podem precisar `chmod o+w`

---

## 8. Estrutura do Projeto

```
clnn_v2/
├── config/
│   ├── smoke_v2.yaml               # Smoke (seq_len=12, 2 epochs)
│   └── exp_full_15d_a4000.yaml     # ★ Produção (encoder.type configurável)
├── src/
│   ├── data/dataset.py             # HuMobNormalizedDataset (IterableDataset + spillover)
│   ├── models/
│   │   ├── humob_model.py          # HuMobModel (encoder_type: lstm|transformer)
│   │   ├── external_info.py        # Embeddings + projeções GELU
│   │   ├── partial_info.py         # CoordLSTM (attention pool + temperatura)
│   │   └── partial_info_transformer.py  # CoordTransformer (mean pool, pre-norm GELU)
│   ├── training/
│   │   ├── train.py                # train_humob_model (CE+MSE, encoder passthrough)
│   │   ├── finetune.py             # FT sequencial; reconstrói encoder do checkpoint
│   │   └── pipeline.py             # generate_humob_submission + pred_gt_parquet
│   └── utils/
│       ├── metrics.py              # compute_geobleu() + compute_dtw()
│       ├── mlflow_tracker.py
│       └── simple_checkpoint.py
├── run_automate.py                 # ★ Entrada: YAML → pipeline completo
├── evaluate_metrics.py             # ★ GEO-BLEU + DTW a partir de pred_gt
├── PRD.md                          # ★ Este arquivo
├── CLAUDE.md
├── dockerfile
└── data/
    └── humob_all_cities_v2_normalized.parquet  # ~4.2 GB
```

---

## 9. Stack

| Componente | Tecnologia |
|------------|------------|
| Framework | PyTorch 2.7 |
| GPU | CUDA 12.8 + cuDNN 9 |
| Dados | Parquet (PyArrow) |
| Tracking | MLflow local |
| Container | Docker `humob:cu128` |
| Hardware | NVIDIA RTX A4000 (16 GB) |
| Métricas | geobleu (Yahoo) + fastdtw |

---

## 10. Convenções

- **Docker para tudo** — não instalar deps no host
- **Dados em [0,1]** — não desnormalizar
- **`discretize_coordinates()`** para converter pred em célula 0–199
- **Feature branches** → merge ao concluir; `feature/transformer-encoder` é a ativa
- **Checkpoints** em `outputs/models/checkpoints/` (3 últimos)
- **Nunca executar experimentos via SSH na sessão** — usuário roda em seu tmux
- **Comparação justa**: mesma seed, mesmo YAML (só trocar `encoder.type`), mesma GPU
