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
sudo docker run --rm --gpus all \
  -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml
```

- **GPU:** NVIDIA RTX A4000 (16 GB VRAM)
- **Anderson NÃO tem sudo** no servidor (equipamento compartilhado da UFAL)
- Não mexer em `/datastore/docker` nem em drivers de vídeo
- Docker image `humob:cu128` já construída (~9.2 GB) — não rebuildar sem necessidade
- Docker precisa de sudo; sem sudo, omitir `--gpus all` funciona para testes sem GPU

## Estado atual do modelo (v2 — aguardando início, 2026-04-14)

### Fases implementadas
| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Reparametrização (lstm_hidden=128, user_emb_dim=64, fusion_dim=256) | ✅ |
| 2 | Loss CE + MSE auxiliar (CE sobre clusters KMeans, alpha=0.1) | ✅ |
| 3 | Attention pooling sobre todos os LSTM outputs | ✅ |
| 4 | Scheduled sampling (max_p=0.3, ramp linear) | ✅ |
| 5 | Transformer Encoder substituindo BiLSTM | 🔵 Próxima |

### Correções de arquitetura aplicadas (2026-04-14)
| Problema | Arquivo | Correção |
|----------|---------|----------|
| Attention collapse (seq longa sem temperatura) | partial_info.py | scores /= sqrt(output_dim) |
| LSTM output sem normalização | humob_model.py | LayerNorm(lstm_hidden*2) antes do WeightedFusion |
| ReLU zerando 50% das features temporais/POI | external_info.py | ReLU -> GELU em todas projeções |
| Val loop: criterion não definido no escopo | train.py | criterion(pred,target) -> F.mse_loss(pred,target) |

### Baseline v1 (para comparação)
| Métrica | Cidade B | Cidade C | Cidade D | Top HuMob 2024 |
|---------|---------|---------|---------|---------------|
| GEO-BLEU | 0.1017 | 0.0276 | 0.1076 | 0.319 |
| DTW | 174.70 | 236.92 | 156.91 | 27.15 |

## Config de produção atual (exp_full_15d_a4000.yaml)

Parâmetros relevantes para entender decisões:

```yaml
sequence_length: 336     # 7 dias — seq=720 descartava maioria dos users de C/D
max_scheduled_p: 0.3     # 0.7 era agressivo demais para treino do zero
learning_rate: 1e-4      # 3e-5 era conservador demais pos-fixes de arquitetura
lstm_hidden: 128         # BiLSTM -> 256-D output == fusion_dim (DEVE ser igual)
fusion_dim: 256          # WeightedFusion exige lstm_hidden*2 == fusion_dim
n_clusters: 512          # KMeans; ~25 clusters redundantes (dist<0.01), toleravel
max_val_batches: 200     # val em ~4 min; full val so no evaluate_model final
resume_from_checkpoint: false  # checkpoints antigos sao incompativeis com v2
```

## Invariantes críticos da arquitetura

- **fusion_dim DEVE ser igual a lstm_hidden * 2** — WeightedFusion tem assert que quebra caso contrário
- **poi_proj = Linear(85, poi_out_dim)** — dataset tem POI_norm de 85 dimensões (array dentro de coluna object)
- **n_users deve ser >= max(uid)+1** — uids vão de 0 a 99999 em todas as cidades
- **/74.0 no rollout** — d_orig vai de 0 a 74 (75 dias), normalização correta
- **tracking_uri em smoke:** usar file:/tmp/mlruns (container write-only em /tmp)

## Bugs críticos conhecidos (não corrigidos)

- **BPTT estrutural:** seq_len=336 ainda faz backprop por 336 passos em LSTM. Gradient clip em ~2.0 ainda vai disparar na maioria dos batches. Solução real: Transformer (Fase 5).
- **Cobertura em City D:** 7.6M rows / ~6k users ≈ 76 pts/user. seq_len=336 exige 337 pontos contíguos → ainda filtra parte dos users de D. Monitorar cobertura no evaluate_model.

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| PRD.md | Documentação completa — leia primeiro |
| run_automate.py | Pipeline principal (YAML -> KMeans -> treino -> FT -> avaliacao) |
| evaluate_metrics.py | Calcula GEO-BLEU e DTW sobre pred_gt_all_users.parquet |
| src/models/humob_model.py | HuMobModel — arquitetura principal |
| src/models/partial_info.py | CoordLSTM — BiLSTM com attention pooling + temperatura |
| src/models/external_info.py | ExternalInformationFusionNormalized — embeddings + GELU |
| src/training/train.py | Loop de treino com CE+MSE loss, val com F.mse_loss |
| src/training/finetune.py | Fine-tuning sequencial B->C->D |
| config/exp_full_15d_a4000.yaml | Config de producao |
| config/smoke_v2.yaml | Config de teste rapido com checkpoints |

## Convenções do projeto

- **Sempre rodar via Docker** — nunca instalar dependências direto no servidor
- **Tracking via MLflow** local (file:./mlruns) — acessível em mlruns/
- **Checkpoints** em outputs/models/checkpoints/ — mantém os 3 últimos
- **Dados normalizados** em [0,1] — não desnormalizar antes do modelo
- **Coordenadas discretas** via discretize_coordinates() — grade 0-199
- **Branch única:** main — trabalho novo em feature branches, merge ao concluir
