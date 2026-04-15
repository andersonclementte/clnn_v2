# CLAUDE.md — clnn_v2

Contexto rápido para novas sessões. **Leia `PRD.md` antes de qualquer mudança significativa.**

## O projeto

Predição de mobilidade humana para o HuMob Challenge 2024. Prevê (x, y) em grade 200×200 a cada 30 min para os dias 61–75. Cidades A (treino), B/C/D (fine-tuning). Objetivo: publicar artigo comparável ao top do challenge (GEO-BLEU 0.319, DTW 27.15).

## Servidor e execução

```bash
ssh anderson.clemente@192.168.222.252     # VPN necessária; senha: &J!^o&4#$WSKX!
cd ~/clnn_v2

# Treino — Docker SEM sudo (usuário está no grupo docker)
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml
```

- GPU: NVIDIA RTX A4000 (16 GB)
- Anderson **NÃO tem sudo** — nunca sugerir `sudo`
- Docker image `humob:cu128` já construída
- **Nunca executar experimentos via SSH durante a sessão** — usuário roda em seu próprio tmux

## Estado atual (2026-04-15)

### Fases
| Fase | Status |
|------|--------|
| 1 Reparametrização (lstm_hidden=128, fusion_dim=256) | ✅ |
| 2 Loss CE + MSE auxiliar | ✅ |
| 3 Attention pooling | ✅ |
| 4 Scheduled sampling (ramp 0 → 0.3) | ✅ |
| **5 Substituir BiLSTM por Transformer** | 🔵 Próxima — plano no PRD |
| 6 Cobertura de usuários em C/D | 🔵 Futuro |

### Última execução (epoch 1 v2)
- Train loss (CE+MSE): **4.12** | Val loss (MSE): **0.0071**
- Tempo: ~9h/epoch | ~24.5k batches
- **Problema ativo:** gradientes continuam explodindo (pre=11–59, clip=2.0) — causa estrutural do BPTT em LSTM. Fase 5 resolve.

## Invariantes arquiteturais (não quebrar)

- `fusion_dim == lstm_hidden * 2` — `WeightedFusion` tem assert que quebra em runtime
- `poi_proj = Linear(85, poi_out_dim)` — POI_norm é array 85-D (coluna `object`)
- `n_users >= max(uid)+1` — uids vão até 99999 em todas as cidades
- `tracking_uri: file:/tmp/mlruns` em smoke (permissões do container)

## Config de produção (`exp_full_15d_a4000.yaml`)

Decisões não óbvias:

```yaml
sequence_length: 336     # 7 dias — seq=720 descartava maioria dos users de D
max_scheduled_p: 0.3     # 0.7 era agressivo treinando do zero
learning_rate: 1e-4      # 3e-5 era conservador pós-fixes de arquitetura
lstm_hidden: 128         # → BiLSTM 256-D == fusion_dim (DEVE bater)
n_clusters: 512          # KMeans; ~25 clusters redundantes, tolerável
max_val_batches: 200     # val em ~4 min
```

## Permissões (gotchas Docker)

- Container roda como `laccan` (UID 1000), host como `anderson.clemente` (UID diferente)
- Arquivos criados no host precisam `chmod o+w` para o container sobrescrever
- `outputs/models/humob_model_A.pt` é o arquivo final salvo — se já existe, precisa permissão `o+w`
- `mlruns/` em 757 funciona

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `PRD.md` | ★ Documentação completa — leia primeiro |
| `run_automate.py` | Pipeline: YAML → KMeans → treino → FT → avaliação |
| `evaluate_metrics.py` | GEO-BLEU + DTW oficiais |
| `src/models/humob_model.py` | `HuMobModel` (escolhe encoder via config) |
| `src/models/partial_info.py` | `CoordLSTM` com attention pooling + temperatura |
| `src/models/external_info.py` | Embeddings + projeções GELU |
| `src/training/train.py` | `train_humob_model` (CE+MSE, val F.mse_loss) |
| `src/training/finetune.py` | Fine-tuning sequencial B→C→D |
| `config/exp_full_15d_a4000.yaml` | ★ Config de produção |
| `config/smoke_v2.yaml` | Smoke test (2 epochs, max_batches) |

## Convenções

- Docker para tudo — nunca instalar deps no host
- MLflow local em `file:./mlruns`
- Checkpoints em `outputs/models/checkpoints/` (últimos 3)
- Dados em [0,1] — não desnormalizar
- `discretize_coordinates()` para converter predição em célula 0–199
- Branch única `main`; feature branches → merge ao concluir
