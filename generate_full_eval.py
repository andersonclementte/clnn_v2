"""
Gera pred_gt_pairs para TODOS os usuários das cidades B, C, D.
Usa test_days=(0.55, 1.0) como janela — cobre ~33 dias, dando margem
suficiente para formar sequências com sequence_length=720 (15 dias).
Filtra predições para day_range=(61, 75).
"""
import os, sys, torch, numpy as np
os.chdir("/workspace")
sys.path.insert(0, "/workspace")

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path

from src.models.humob_model import HuMobModel, discretize_coordinates
from src.data.dataset import HuMobNormalizedDataset
from torch.utils.data import DataLoader

# ── Configurações ──────────────────────────────────────────
CHECKPOINT    = "humob_model_finetuned_D.pt"   # último da cadeia B→C→D
PARQUET       = "data/humob_all_cities_v2_normalized.parquet"
OUTPUT        = "outputs/eval/pred_gt_all_users.parquet"
TARGET_CITIES = ["B", "C", "D"]
DAY_RANGE     = (61, 75)
BATCH_SIZE    = 1024
MAX_SEQS_USER = 30        # limita sequências por usuário (evita parquet gigante)
TEST_DAYS     = (0.55, 1.0)  # ~dias 41-75; garante contexto de 720 slots
# ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Carrega checkpoint
try:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(CHECKPOINT, map_location=device)

centers_obj = ckpt.get("centers") or ckpt.get("extra", {}).get("centers")
centers = centers_obj.to(device) if isinstance(centers_obj, torch.Tensor) \
          else torch.as_tensor(centers_obj, dtype=torch.float32, device=device)

cfg       = ckpt.get("config", ckpt.get("extra", {}).get("config", {}))
state     = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
seq_len   = int(cfg.get("sequence_length", 720))
n_users   = int(cfg.get("n_users", 100_000))
n_cities  = int(cfg.get("n_cities", 4))

model = HuMobModel(n_users=n_users, n_cities=n_cities,
                   cluster_centers=centers, sequence_length=seq_len).to(device)
model.load_state_dict(state, strict=True)
model.eval()
print(f"Modelo carregado: seq_len={seq_len}, n_users={n_users}")

# Dataset
ds = HuMobNormalizedDataset(
    parquet_path=PARQUET,
    cities=TARGET_CITIES,
    mode="test",
    sequence_length=seq_len,
    max_sequences_per_user=MAX_SEQS_USER,
    test_days=TEST_DAYS,
)

# LUTs seno/cosseno
slots     = torch.arange(48, device=device).float()
phase_lut = 2 * torch.pi * (slots / 48.0)
sin_lut   = torch.sin(phase_lut)
cos_lut   = torch.cos(phase_lut)

# Schema Parquet
schema = pa.schema([
    ("uid", pa.int64()), ("city", pa.int32()),
    ("day", pa.int32()), ("slot", pa.int32()),
    ("x_pred_norm", pa.float32()), ("y_pred_norm", pa.float32()),
    ("x_gt_norm",   pa.float32()), ("y_gt_norm",   pa.float32()),
    ("x_pred_cell", pa.int32()),   ("y_pred_cell", pa.int32()),
    ("x_gt_cell",   pa.int32()),   ("y_gt_cell",   pa.int32()),
])

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
writer    = pq.ParquetWriter(OUTPUT, schema, compression="zstd")
total     = 0
buf       = []

def flush(buf):
    global total
    if not buf: return
    uid_t   = torch.as_tensor([int(b[0].item()) for b in buf], device=device, dtype=torch.long)
    dnorm_t = torch.stack([b[1] for b in buf]).to(device)
    tsin_t  = torch.stack([b[2] for b in buf]).to(device)
    tcos_t  = torch.stack([b[3] for b in buf]).to(device)
    city_t  = torch.as_tensor([int(b[4].item()) for b in buf], device=device, dtype=torch.long)
    poi_t   = torch.from_numpy(np.stack([b[5].cpu().numpy() for b in buf])).to(device)
    coords_t= torch.from_numpy(np.stack([b[6].cpu().numpy() for b in buf])).to(device)
    target_t= torch.from_numpy(np.stack([b[7].cpu().numpy() for b in buf])).to(device)
    if target_t.ndim == 3: target_t = target_t.squeeze(1)

    # Avança 1 passo
    day_cur  = torch.round(dnorm_t * 74).long()
    angle    = torch.atan2(tsin_t, tcos_t)
    slot_cur = torch.remainder(torch.round((angle + 2*torch.pi)/(2*torch.pi)*48), 48).long()
    slot_next = (slot_cur + 1) % 48
    day_next  = day_cur + (slot_next == 0).long()

    # Filtra por day_range
    keep = (day_next >= DAY_RANGE[0]) & (day_next <= DAY_RANGE[1])
    if keep.sum() == 0: return

    with torch.inference_mode():
        pred = model.forward_single_step(
            uid_t[keep], day_next[keep].float()/74.0,
            sin_lut[slot_next[keep]], cos_lut[slot_next[keep]],
            city_t[keep], poi_t[keep], coords_t[keep]
        )
    if pred.ndim == 3: pred = pred.squeeze(1)

    gt_k       = target_t[keep]
    pred_cell  = discretize_coordinates(pred).to(torch.int32)
    gt_cell    = discretize_coordinates(gt_k).to(torch.int32)

    def t2n(x): return x.detach().cpu().numpy()

    table = pa.table({
        "uid":         t2n(uid_t[keep]).astype(np.int64),
        "city":        t2n(city_t[keep]).astype(np.int32),
        "day":         t2n(day_next[keep]).astype(np.int32),
        "slot":        t2n(slot_next[keep]).astype(np.int32),
        "x_pred_norm": t2n(pred)[:,0].astype(np.float32),
        "y_pred_norm": t2n(pred)[:,1].astype(np.float32),
        "x_gt_norm":   t2n(gt_k)[:,0].astype(np.float32),
        "y_gt_norm":   t2n(gt_k)[:,1].astype(np.float32),
        "x_pred_cell": t2n(pred_cell)[:,0].astype(np.int32),
        "y_pred_cell": t2n(pred_cell)[:,1].astype(np.int32),
        "x_gt_cell":   t2n(gt_cell)[:,0].astype(np.int32),
        "y_gt_cell":   t2n(gt_cell)[:,1].astype(np.int32),
    }, schema=schema)
    writer.write_table(table)
    total += table.num_rows

with torch.inference_mode():
    for sample in tqdm(ds, desc="gerando pred+gt"):
        buf.append(sample)
        if len(buf) >= BATCH_SIZE:
            flush(buf)
            buf.clear()
    if buf:
        flush(buf)
        buf.clear()

writer.close()
print(f"\n✅ Salvo em {OUTPUT}")
print(f"   Total de linhas: {total:,}")

# Estatísticas rápidas
import pyarrow.parquet as pq2, numpy as np2
df = pq2.read_table(OUTPUT).to_pandas()
df["cell_error"] = np2.sqrt((df["x_pred_cell"]-df["x_gt_cell"])**2 + (df["y_pred_cell"]-df["y_gt_cell"])**2)
df["km_error"]   = df["cell_error"] * 0.5
city_map = {1:"B", 2:"C", 3:"D"}
df["city_name"] = df["city"].map(city_map)
print(f"   Usuários únicos: {df['uid'].nunique():,}")
print("\n=== Erro por cidade ===")
for city in ["B","C","D"]:
    sub = df[df["city_name"]==city]
    if len(sub)==0: continue
    print(f"  {city}: n={len(sub):,} users={sub['uid'].nunique()} media={sub['km_error'].mean():.2f}km mediana={sub['km_error'].median():.2f}km p90={sub['km_error'].quantile(0.9):.2f}km")
