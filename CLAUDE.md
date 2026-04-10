# CLAUDE.md — clnn_v2

Contexto essencial para novas sessões de trabalho neste projeto.

## O que é este projeto

Modelo de predição de mobilidade humana para o **HuMob Challenge 2024** (Yahoo Japan / SIGSPATIAL). Prediz a localização (x, y) de usuários em uma grade 200×200 a cada 30 minutos, para os dias 61–75 com base nos dias 1–60. Dataset: cidades A, B, C, D (~4.5 GB parquet). Objetivo: publicar artigo com resultados comparáveis ao top do challenge.

**Leia o PRD.md completo** antes de qualquer trabalho — ele contém histórico de bugs corrigidos, resultados reais, diagnóstico do modelo atual e o plano de implementação v2.

## Acesso ao servidor de treino

```bash
# Já conectado à VPN (192.168.222.0/24)
ssh anderson.clemente@192.168.222.252   # senha: &J!^o&4#$WSKX!

# Projeto
cd ~/clnn_v2

# Rodar experimento
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace -e MPLCONFIGDIR=/tmp/mpl \
  -v "$(pwd):/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml

# Calcular métricas oficiais (GEO-BLEU + DTW)
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace -e MPLCONFIGDIR=/tmp/mpl \
  -v "$(pwd):/workspace" humob:cu128 \
  python /workspace/evaluate_metrics.py
```

- **GPU:** NVIDIA RTX A4000 (16 GB VRAM)
- **Anderson NÃO tem sudo** no servidor (equipamento compartilhado da UFAL)
- Não mexer em `/datastore/docker` nem em drivers de vídeo
- Docker image `humob:cu128` já construída (~9.2 GB) — não rebuildar sem necessidade

## Estado atual do modelo (v1 — baseline)

| Métrica | Cidade B | Cidade C | Cidade D | Top HuMob 2024 |
|---------|---------|---------|---------|---------------|
| GEO-BLEU | 0.1017 | 0.0276 | 0.1076 | 0.319 |
| DTW | 174.70 | 236.92 | 156.91 | 27.15 |

Cobertura de usuários: 9% (B), 7% (C), 33% (D) — limitação do seq_len=720.

## Problemas críticos conhecidos no modelo v1

1. **lstm_hidden=4** — BiLSTM com 160 params para sequências de 720 passos
2. **user_emb_dim=4** — 400K params gastos em 4 dimensões para 100K usuários
3. **fusion_dim=8** — trajetória + contexto comprimidos em 8 números
4. **Loss MSELoss** — desalinhada das métricas GEO-BLEU/DTW; não usa os logits de cluster
5. **Attention ausente** — LSTM descarta os 719 hidden states intermediários
6. **Mismatch treino/avaliação** — treina step-a-step, avalia com rollout de 720 passos

## Plano de implementação v2 (próximas tarefas)

Detalhes completos na **seção 5 do PRD.md**. Ordem de prioridade:

1. **Fase 1** 🔴 — Reparametrização: `lstm_hidden` 4→64, `user_emb_dim` 4→32, `fusion_dim` 8→128, `poi_out_dim` 4→16. Só muda o YAML + leitura dos hiperparâmetros no modelo.
2. **Fase 2** 🔴 — Loss CrossEntropy sobre clusters KMeans + MSE auxiliar (expor logits em `humob_model.py`, mudar loss em `train.py` e `finetune.py`).
3. **Fase 3** 🟡 — Attention pooling sobre outputs do LSTM em `src/models/partial_info.py`.
4. **Fase 4** 🟡 — Scheduled sampling em `src/training/train.py`.
5. **Fase 5** 🟢 — Substituir BiLSTM por Transformer Encoder com atenção local (janela=48).
6. **Fase 6** 🟢 — Ampliar cobertura de usuários (seq_len adaptativo ou re-treino com seq_len=360).

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `PRD.md` | Documentação completa — leia primeiro |
| `run_automate.py` | Pipeline principal (YAML → KMeans → treino → FT → avaliação) |
| `evaluate_metrics.py` | Calcula GEO-BLEU e DTW sobre `pred_gt_all_users.parquet` |
| `src/models/humob_model.py` | `HuMobModel` — arquitetura principal |
| `src/models/partial_info.py` | `CoordLSTM` — BiLSTM sobre sequências de coordenadas |
| `src/models/external_info.py` | `ExternalInformationFusionNormalized` — embeddings + projeções |
| `src/training/train.py` | Loop de treino, `MSELoss`, `evaluate_model` |
| `src/training/finetune.py` | Fine-tuning sequencial B→C→D |
| `src/utils/metrics.py` | `compute_geobleu()`, `compute_dtw()` |
| `config/exp_full_15d_a4000.yaml` | ★ Config de produção |
| `config/smoke.yaml` | Config de teste rápido (seq_len=12, 1 época) |

## Convenções do projeto

- **Sempre rodar via Docker** — nunca instalar dependências direto no servidor
- **Tracking via MLflow** local (`file:./mlruns`) — acessível em `mlruns/`
- **Checkpoints** em `outputs/models/checkpoints/` — mantém os 3 últimos
- **Dados normalizados** em `[0,1]` — não desnormalizar antes do modelo
- **Coordenadas discretas** via `discretize_coordinates()` — grade 0–199
- **Branch única:** `main` — trabalho novo em feature branches, merge ao concluir
