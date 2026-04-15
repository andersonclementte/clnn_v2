# CLAUDE.md — clnn_v2

Contexto rápido para novas sessões. **Leia `PRD.md` antes de qualquer mudança significativa.**

## O projeto

Predição de mobilidade humana para o HuMob Challenge 2024. Prevê (x, y) em grade 200×200 a cada 30 min para os dias 61–75. Cidades A (treino), B/C/D (fine-tuning). Objetivo: publicar artigo comparável ao top do challenge (GEO-BLEU 0.319, DTW 27.15).

## Estratégia central: **comparar BiLSTM vs Transformer**

Ambos os encoders estão implementados e selecionáveis via `encoder.type` no YAML. O paper apresentará ambos lado a lado — **não é substituição, é ablação**. BiLSTM está em treino como baseline; Transformer roda em seguida com mesma config.

## Servidor e execução

```bash
ssh anderson.clemente@192.168.222.252     # VPN; senha: &J!^o&4#$WSKX!
cd ~/clnn_v2

# Treino — Docker SEM sudo
docker run --rm --gpus all -u $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD:/workspace" humob:cu128 \
  python /workspace/run_automate.py --config /workspace/config/exp_full_15d_a4000.yaml
```

- GPU: NVIDIA RTX A4000 (16 GB)
- Anderson **NÃO tem sudo** — nunca sugerir `sudo`
- **Nunca executar experimentos via SSH durante a sessão** — usuário roda em seu próprio tmux

## Estado atual (2026-04-15)

### Fases
| Fase | Status |
|------|--------|
| 1 Reparametrização (lstm_hidden=128, fusion_dim=256) | ✅ |
| 2 Loss CE + MSE auxiliar | ✅ |
| 3 Attention pooling | ✅ |
| 4 Scheduled sampling (ramp 0 → 0.3) | ✅ |
| **5 Transformer Encoder como alternativa (encoder_type switch)** | ✅ Implementado (branch `feature/transformer-encoder`), aguardando run |
| 6 Cobertura de usuários em C/D | 🔵 Futuro |

### Treino BiLSTM em andamento (baseline)
- **Branch:** `main`
- **Epoch 1:** train loss 4.1226 | val loss 0.0071 (~9h)
- **Epoch 2 (em andamento):** running avg ~3.4, oscila 3.3–4.3
- **Sintoma estrutural:** grad pre-clip **12–59** constante, GradNorm pós-clip saturada em **2.0** em praticamente todo step → motivação empírica da comparação com Transformer

### Próximo passo
1. Deixar BiLSTM convergir (ou platô claro em val por 2–3 epochs) + FT B/C/D
2. Salvar artefatos com sufixo `_lstm_baseline`
3. Checkout `feature/transformer-encoder`, smoke test, depois full run com mesma config (só trocando `encoder.type`)
4. Comparação GEO-BLEU/DTW em tabela (seção 4.4 do PRD)

## Invariantes arquiteturais (não quebrar)

- **`fusion_dim == saída do encoder`** — BiLSTM: `lstm_hidden*2`; Transformer: `d_model`. WeightedFusion tem assert runtime.
- `poi_proj = Linear(85, poi_out_dim)` — POI_norm é array 85-D (coluna `object`)
- `n_users >= max(uid)+1` — uids até 99999
- `tracking_uri: file:/tmp/mlruns` em smoke (permissões do container)
- `encoder_type` + hiperparams do Transformer **são salvos no config do checkpoint** — FT e eval reconstroem o encoder certo automaticamente

## Config de produção (`exp_full_15d_a4000.yaml`)

Decisões não óbvias:

```yaml
sequence_length: 336     # 7 dias — seq=720 descartava maioria dos users de D
max_scheduled_p: 0.3     # 0.7 era agressivo treinando do zero
learning_rate: 1e-4      # 3e-5 era conservador pós-fixes de arquitetura
lstm_hidden: 128         # → BiLSTM 256-D == fusion_dim (DEVE bater)
fusion_dim: 256          # == d_model do Transformer também
n_clusters: 512          # KMeans; ~25 clusters redundantes, tolerável
max_val_batches: 200     # val em ~4 min

encoder:
  type: "transformer"    # "lstm" | "transformer"
  transformer_nhead: 8
  transformer_num_layers: 4
  transformer_ff_dim: 512
  transformer_dropout: 0.1
```

Para rodar o baseline BiLSTM, mudar `encoder.type: "lstm"`; para Transformer, manter como está.

## Artefatos da comparação (convenção de nomes)

Preservar os dois runs com sufixos:

```
outputs/models/humob_model_A_{lstm_baseline,transformer}.pt
outputs/models/humob_model_finetuned_{B,C,D}_{lstm_baseline,transformer}.pt
outputs/eval/pred_gt_all_users_{lstm,transformer}.parquet
outputs/eval/metrics_{lstm,transformer}.json
```

Taggear runs MLflow com `encoder_type` para filtragem.

## Permissões (gotchas Docker)

- Container roda como `laccan` (UID 1000); host como `anderson.clemente` → sempre `-u $(id -u):$(id -g)`
- `outputs/models/humob_model_A.pt` preexistente pode precisar `chmod o+w`
- `mlruns/` em 757 funciona
- **Ao trocar de encoder:** limpar `outputs/models/humob_model_A.pt` + `checkpoints/*.pt` (incompatíveis)

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `PRD.md` | ★ Documentação completa — leia primeiro |
| `run_automate.py` | Pipeline: YAML → KMeans → treino → FT → avaliação |
| `evaluate_metrics.py` | GEO-BLEU + DTW oficiais |
| `src/models/humob_model.py` | `HuMobModel` (switch `encoder_type`) |
| `src/models/partial_info.py` | `CoordLSTM` (attention pool + temperatura) |
| `src/models/partial_info_transformer.py` | `CoordTransformer` (mean pool, pre-norm GELU) |
| `src/models/external_info.py` | Embeddings + projeções GELU |
| `src/training/train.py` | `train_humob_model` (CE+MSE, val `F.mse_loss`, encoder passthrough) |
| `src/training/finetune.py` | FT sequencial B→C→D; reconstrói encoder do checkpoint |
| `config/exp_full_15d_a4000.yaml` | ★ Produção |
| `config/smoke_v2.yaml` | Smoke test (2 epochs, max_batches) |

## Convenções

- Docker para tudo — nunca instalar deps no host
- MLflow local em `file:./mlruns`
- Checkpoints em `outputs/models/checkpoints/` (últimos 3)
- Dados em [0,1] — não desnormalizar
- `discretize_coordinates()` para converter predição em célula 0–199
- **Comparação justa:** mesma seed, mesmo YAML (só trocar `encoder.type`), mesma GPU
- Branch ativa do Transformer: `feature/transformer-encoder`
