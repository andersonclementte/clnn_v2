import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from src.models.humob_model import HuMobModel, discretize_coordinates
from src.data.dataset import HuMobNormalizedDataset
from src.training.train import compute_cluster_centers, train_humob_model, evaluate_model


def run_full_pipeline(
    parquet_path: str,
    device: torch.device = None,
    n_clusters: int = 512,
    n_epochs: int = 5,
    sequence_length: int = 24,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_users_A: int = 100_000
):
    """
    Pipeline completo do HuMob Challenge:
    1. Calcula cluster centers
    2. Treina modelo na cidade A
    3. Avalia zero-shot em B, C, D
    4. Gera plots de resultado
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 PIPELINE COMPLETO HUMOB CHALLENGE")
    print(f"Device: {device}")
    print(f"Clusters: {n_clusters}")
    print(f"Epochs: {n_epochs}")
    print(f"Sequence length: {sequence_length}")
    print("=" * 60)
    
    # 1. CLUSTER CENTERS
    print("\n📍 ETAPA 1: Calculando cluster centers...")
    centers = compute_cluster_centers(
        parquet_path=parquet_path,
        cities=["A"],
        n_clusters=n_clusters,
        save_path=f"centers_A_{n_clusters}.npy"
    ).to(device)
    
    print(f"✅ Centers prontos: {centers.shape}")
    
    # 2. TREINAMENTO EM A
    print("\n🏋️ ETAPA 2: Treinando modelo na cidade A...")
    model, train_losses, val_losses = train_humob_model(
        parquet_path=parquet_path,
        cluster_centers=centers,
        device=device,
        cities=["A"],
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_users=n_users_A,
        save_path="humob_model_A.pt"
    )
    
    # 3. AVALIAÇÃO ZERO-SHOT EM B, C, D
    print("\n🎯 ETAPA 3: Avaliação zero-shot em B, C, D...")
    results = {}
    
    for city in ["B", "C", "D"]:
        print(f"\n--- Avaliando cidade {city} ---")
        try:
            mse, cell_error = evaluate_model(
                parquet_path=parquet_path,
                checkpoint_path="humob_model_A.pt",
                device=device,
                cities=[city],
                n_samples=3000,
                sequence_length=sequence_length
            )
            results[city] = {"mse": mse, "cell_error": cell_error}
        except Exception as e:
            print(f"❌ Erro avaliando cidade {city}: {e}")
            results[city] = {"mse": float('inf'), "cell_error": float('inf')}
    
    # 4. PLOTS DOS RESULTADOS
    print("\n📊 ETAPA 4: Gerando plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Curvas de treino
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Curvas de Treinamento (Cidade A)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Pesos da fusão
    w_r = model.weighted_fusion.w_r.item()
    w_e = model.weighted_fusion.w_e.item()
    axes[0, 1].bar(['Static (w_r)', 'Dynamic (w_e)'], [w_r, w_e], 
                   color=['lightcoral', 'lightblue'])
    axes[0, 1].set_ylabel('Peso')
    axes[0, 1].set_title('Pesos da Fusão Ponderada')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MSE por cidade
    cities = list(results.keys())
    mse_values = [results[city]["mse"] for city in cities]
    axes[1, 0].bar(cities, mse_values, color=['red', 'green', 'purple'])
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Zero-shot por Cidade')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Erro em células por cidade
    cell_errors = [results[city]["cell_error"] for city in cities]
    axes[1, 1].bar(cities, cell_errors, color=['red', 'green', 'purple'])
    axes[1, 1].set_ylabel('Erro Médio (células)')
    axes[1, 1].set_title('Erro em Células por Cidade')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('humob_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. RELATÓRIO FINAL
    print("\n📋 RELATÓRIO FINAL")
    print("=" * 60)
    print(f"✅ Treinamento concluído em {n_epochs} épocas")
    print(f"📈 Loss final de treino: {train_losses[-1]:.4f}")
    print(f"📉 Loss final de validação: {val_losses[-1]:.4f}")
    print(f"⚖️ Pesos da fusão: w_r={w_r:.3f}, w_e={w_e:.3f}")
    print("\n🎯 Resultados Zero-shot:")
    for city in cities:
        print(f"  Cidade {city}: MSE={results[city]['mse']:.4f}, "
              f"Erro células={results[city]['cell_error']:.2f}")
    
    print(f"\n💾 Arquivos gerados:")
    print(f"   - centers_A_{n_clusters}.npy (cluster centers)")
    print(f"   - humob_model_A.pt (modelo treinado)")
    print(f"   - humob_results.png (gráficos)")
    
    # 6. VERIFICAÇÕES DE QUALIDADE
    print("\n🔍 VERIFICAÇÕES:")
    converged = val_losses[-1] < val_losses[0] * 0.8
    avg_mse = np.mean([results[city]["mse"] for city in cities if results[city]["mse"] != float('inf')])
    reasonable_mse = avg_mse < 0.1  # Para dados normalizados [0,1]
    
    print(f"  Convergiu? {'✅' if converged else '❌'} (loss diminuiu 20%+)")
    print(f"  MSE razoável? {'✅' if reasonable_mse else '❌'} (< 0.1 para [0,1])")
    
    if converged and reasonable_mse:
        print("\n🎉 PIPELINE EXECUTADO COM SUCESSO!")
        print("   Pronto para gerar submissões HuMob!")
    else:
        print("\n⚠️ ALGUNS PROBLEMAS DETECTADOS")
        print("   Considere ajustar hiperparâmetros ou verificar dados")
    
    return {
        'model': model,
        'centers': centers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'results_by_city': results,
        'checkpoint_path': 'humob_model_A.pt'
    }

def generate_humob_submission(
    parquet_path: str,
    checkpoint_path: str,
    device: torch.device,
    target_cities: list[str] = ["B", "C", "D"],
    submission_days: tuple = (61, 75),
    sequence_length: int = 24,          # fallback se não houver no ckpt
    # output_file: str = "humob_submission.csv",
    output_file: str = "outputs/submissions/humob_submission.parquet",
    chunk_size: int = 500_000,           # tamanho do flush para disco
    users_batch: int = 512              # tamanho do lote de usuários na GPU
):
    """
    Gera submissão HuMob (uid, city, day, slot, x, y) de forma eficiente (streaming).
    """
    print(f"📄 Gerando submissão HuMob para {target_cities} | dias {submission_days[0]}-{submission_days[1]}")
    import math, gc
    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # ---------- 1) Carrega checkpoint (compatível com layouts novo/antigo)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    centers_obj = ckpt.get("centers", None) or ckpt.get("extra", {}).get("centers", None)
    if centers_obj is None:
        raise RuntimeError("Centers não encontrados no checkpoint (nem em 'centers' nem em 'extra.centers').")
    centers = centers_obj.to(device) if isinstance(centers_obj, torch.Tensor) \
             else torch.as_tensor(centers_obj, dtype=torch.float32, device=device)

    cfg   = ckpt.get("config", ckpt.get("extra", {}).get("config", {}))
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    seq_len_ckpt = int(cfg.get("sequence_length", sequence_length))

    model = HuMobModel(
        n_users=int(cfg.get("n_users", 100_000)),
        n_cities=int(cfg.get("n_cities", 4)),
        cluster_centers=centers,
        sequence_length=seq_len_ckpt,
        prediction_steps=int(cfg.get("prediction_steps", 1)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------- 2) Varredura leve: só a última sequência por usuário
    print("🔄 Coletando ÚLTIMA sequência por usuário (streaming)...")
    # last_seq_by_uid: dict[int, dict] = {}
    last_seq_by_key: dict[tuple[int, int], dict] = {}  # (city_id, uid) -> última seq

    observed_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=target_cities,
        mode="test",                   # usa tudo disponível
        sequence_length=seq_len_ckpt,
        max_sequences_per_user=float('inf')
    )

    for sample in tqdm(observed_ds, desc="scan"):
        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, _ = sample
        # uid_int = int(uid.item())
        uid_int  = int(uid.item())
        city_id  = int(city.item())
        key = (city_id, uid_int)
        dn = float(d_norm.item())

        # slot a partir de seno/cosseno
        angle = math.atan2(float(t_sin.item()), float(t_cos.item()))     # [-π, π]
        angle = (angle + 2*math.pi) % (2*math.pi)                        # [0, 2π)
        last_slot = int(round(angle / (2*math.pi) * 48)) % 48

        # cur = last_seq_by_uid.get(uid_int)
        # if (cur is None) or (dn > cur["d_norm"]):
        #     last_seq_by_uid[uid_int] = {
        #         "d_norm": dn,
        #         "t_sin": float(t_sin.item()),
        #         "t_cos": float(t_cos.item()),
        #         "city": int(city.item()),
        #         "poi_norm": poi_norm.detach().cpu().numpy(),
        #         "coords_seq": coords_seq.detach().cpu().numpy(),
        #         "last_observed_day": float(dn * 74),
        #         "last_observed_slot": last_slot,
        #     }
        cur = last_seq_by_key.get(key)
        if (cur is None) or (dn > cur["d_norm"]):
            last_seq_by_key[key] = {
                "d_norm": dn,
                "t_sin": float(t_sin.item()),
                "t_cos": float(t_cos.item()),
                "city":  city_id,
                "poi_norm": poi_norm.detach().cpu().numpy(),
                "coords_seq": coords_seq.detach().cpu().numpy(),
                "last_observed_day": float(dn * 74),
                "last_observed_slot": last_slot,
            }

    # n_users = len(last_seq_by_uid)
    # print(f"📊 Usuários únicos coletados: {n_users:,}")
    n_users = len(last_seq_by_key)
    print(f"📊 Pares (cidade, uid) coletados: {n_users:,}")

    # ---------- 3) Escrita incremental de CSV
    # total_rows  = 0

    # def flush_rows():
    #     nonlocal submission_rows, first_chunk, total_rows
    #     if not submission_rows:
    #         return
    #     df_chunk = (
    #         pd.DataFrame(submission_rows)
    #         .sort_values(["city", "uid", "day", "slot"])
    #         [["uid", "city", "day", "slot", "x", "y"]]            # ordem fixa
    #     )
    #     df_chunk.to_csv(
    #         output_file,
    #         index=False,
    #         mode=("w" if first_chunk else "a"),
    #         header=first_chunk,
    #     )
    #     total_rows += len(df_chunk)
    #     submission_rows.clear()
    #     first_chunk = False
    from pathlib import Path
    import pyarrow as pa, pyarrow.parquet as pq

    _parquet_schema = pa.schema([
        ("uid",  pa.int64()),
        ("city", pa.int32()),
        ("day",  pa.int32()),
        ("slot", pa.int32()),
        ("x",    pa.int32()),
        ("y",    pa.int32()),
    ])

    out_path = Path(output_file)
    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    submission_rows = []

    def flush_rows():
        nonlocal writer, total_rows, submission_rows
        if not submission_rows:
            return
        df_chunk = (pd.DataFrame(submission_rows)
                    .sort_values(["city","uid","day","slot"])
                    [["uid","city","day","slot","x","y"]])
        table = pa.Table.from_pandas(df_chunk, schema=_parquet_schema, preserve_index=False)

        # 1º uso: cria writer; próximos: reusa
        if writer is None:
            # dica: remova row_group_size aqui (ver passo 2)
            writer = pq.ParquetWriter(out_path.as_posix(), _parquet_schema,
                                    compression="zstd", use_dictionary=True)

        writer.write_table(table)  # 1 row-group por flush
        total_rows += len(df_chunk)
        submission_rows.clear()
        # sentinela de auditoria:
        print(f"[flush] wrote {len(df_chunk):,} rows • total_rows={total_rows:,}")

    # ---------- 4) Inferência em lote por usuários
    # uids = list(last_seq_by_uid.keys())
    pairs = list(last_seq_by_key.keys())  # lista de (city_id, uid)
    total_slots = (submission_days[1] - submission_days[0] + 1) * 48
    print(f"🚀 Inferindo em lotes de {users_batch} usuários • total_slots={total_slots}")

    slots = torch.arange(48, device=device).float()
    phase_lut = 2*torch.pi*(slots/48.0)
    sin_lut, cos_lut = torch.sin(phase_lut), torch.cos(phase_lut)

    with torch.inference_mode():
        for i in tqdm(range(0, n_users, users_batch), desc="infer"):
            # --- dentro do loop por lotes ---
            batch_pairs = pairs[i:i+users_batch]                  # [(city_id, uid), ...]
            batch = [last_seq_by_key[p] for p in batch_pairs]

            # vetores (NumPy -> tensor) para uid/city
            uid_arr  = np.asarray([p[1] for p in batch_pairs], dtype=np.int64)
            city_arr = np.asarray([p[0] for p in batch_pairs], dtype=np.int64)
            uid_tensor  = torch.as_tensor(uid_arr,  device=device)
            city_tensor = torch.as_tensor(city_arr, device=device)

            # poi / coords (stack mais rápido)
            poi_arr    = np.ascontiguousarray(np.stack([b["poi_norm"]  for b in batch], axis=0), dtype=np.float32)
            coords_arr = np.ascontiguousarray(np.stack([b["coords_seq"] for b in batch], axis=0), dtype=np.float32)
            # poi_tensor   = torch.from_numpy(poi_arr).to(device, non_blocking=True)
            # coords_seq_t = torch.from_numpy(coords_arr).to(device, non_blocking=True)
            poi_tensor   = torch.from_numpy(poi_arr)
            coords_seq_t = torch.from_numpy(coords_arr)

            if device.type == "cuda":
                torch.cuda.empty_cache()
                poi_tensor   = poi_tensor.pin_memory().to(device, non_blocking=True)
                coords_seq_t = coords_seq_t.pin_memory().to(device, non_blocking=True)
            else:
                poi_tensor   = poi_tensor.to(device)
                coords_seq_t = coords_seq_t.to(device)

            # tamanho do batch
            B = uid_arr.shape[0]

            # tempo inicial FIXO (dias 61–75)
            current_day_vec  = torch.full((B,), submission_days[0], dtype=torch.long, device=device)
            current_slot_vec = torch.zeros(B, dtype=torch.long, device=device)

            

            for step in range(total_slots):
                d_norm_current = (current_day_vec.float() / 74.0)
                # phase = 2 * torch.pi * (current_slot_vec.float() / 48.0)
                # t_sin_current = torch.sin(phase)
                # t_cos_current = torch.cos(phase)
                t_sin_current = sin_lut[current_slot_vec]
                t_cos_current = cos_lut[current_slot_vec]

                pred = model.forward_single_step(
                    uid_tensor, d_norm_current, t_sin_current, t_cos_current,
                    city_tensor, poi_tensor, coords_seq_t
                )
                pred_disc = discretize_coordinates(pred)

                # acumula linhas (usa arrays, não batch_ids)
                for j in range(B):
                    submission_rows.append({
                        "uid":  int(uid_arr[j]),
                        "city": int(city_arr[j]),
                        "day":  int(current_day_vec[j].item()),
                        "slot": int(current_slot_vec[j].item()),
                        "x":    int(pred_disc[j, 0].item()),
                        "y":    int(pred_disc[j, 1].item()),
                    })

                # teacher forcing batelado
                new_points = pred.clamp(0.0, 1.0).unsqueeze(1)  # [B,1,2]
                coords_seq_t = torch.cat([coords_seq_t[:, 1:, :], new_points], dim=1)

                # avança tempo
                current_slot_vec += 1
                rolled = current_slot_vec >= 48
                current_slot_vec[rolled] = 0
                current_day_vec[rolled] += 1

                if len(submission_rows) >= chunk_size:
                    flush_rows()

            # limpeza por lote
            del uid_tensor, city_tensor, poi_tensor, coords_seq_t
            del current_day_vec, current_slot_vec, pred, pred_disc, new_points
            torch.cuda.empty_cache()
            gc.collect()

    # último flush
    flush_rows()
    # fecha com segurança
    if writer is not None:
        writer.close()

    print(f"💾 Submissão salva: {out_path}")
    print(f"   Linhas: {total_rows:,}")
    return {"path": out_path.as_posix(), "rows": total_rows, "users": n_users}


# ------------------------------------------
# Predição + Ground Truth → Parquet
# ------------------------------------------
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

def generate_pred_gt_parquet(
    parquet_path: str,
    checkpoint_path: str,
    device: torch.device,
    target_cities: list[str] = ["B", "C", "D"],
    split: str = "test",                 # "val" ou "test"
    day_range: tuple[int, int] | None = (61, 75),  # manter apenas passos cujo (day_next) esteja neste range; None = todos
    sequence_length: int = 24,           # fallback se não houver no checkpoint
    output_file: str = "outputs/eval/pred_gt_pairs.parquet",
    batch_size: int = 2048,
    chunk_size: int = 500_000,
    grid_size: int = 200,
):
    """
    Gera um Parquet com (predições e ground truth) 1-passo (n+1) para cada amostra.
    Não grava métricas; você calcula depois (MSE normalizado, cell error, etc).
    Colunas:
      uid, city, day, slot, split, seq_len,
      x_pred_norm, y_pred_norm, x_gt_norm, y_gt_norm,
      x_pred, y_pred, x_gt, y_gt,
      x_pred_cell, y_pred_cell, x_gt_cell, y_gt_cell
    """

    def _process_buffer(
        buf,
        *,
        model,
        device,
        writer,
        sin_lut,
        cos_lut,
        seq_len_ckpt: int,
        split: str,
        grid_size: int,
        day_range: tuple[int, int] | None
    ) -> int:
        """Processa UM buffer (pode ser 'cheio' ou o resto), escreve no Parquet e retorna nº de linhas escritas."""
        if not buf:
            return 0

        # 1) Empacota tensores
        uid_t   = torch.as_tensor([int(b[0].item()) for b in buf], device=device, dtype=torch.long)
        dnorm_t = torch.stack([b[1] for b in buf]).to(device)    # [B]
        tsin_t  = torch.stack([b[2] for b in buf]).to(device)
        tcos_t  = torch.stack([b[3] for b in buf]).to(device)
        city_t  = torch.as_tensor([int(b[4].item()) for b in buf], device=device, dtype=torch.long)
        poi_t   = torch.from_numpy(np.stack([b[5].cpu().numpy() for b in buf], axis=0)).to(device)
        coords_t= torch.from_numpy(np.stack([b[6].cpu().numpy() for b in buf], axis=0)).to(device)  # [B, seq_len, 2]
        target_t= torch.from_numpy(np.stack([b[7].cpu().numpy() for b in buf], axis=0)).to(device)  # [B, 2] norm
        if target_t.ndim == 3 and target_t.shape[1] == 1:
            target_t = target_t.squeeze(1)          # [B,2]


        B = uid_t.shape[0]

        # 2) Reconstrói (day_cur, slot_cur) e avança 1 passo → (day_next, slot_next)
        day_cur  = torch.round(dnorm_t * 74).long()
        angle    = torch.atan2(tsin_t, tcos_t)  # [-π, π]
        slot_cur = torch.remainder(torch.round((angle + 2*torch.pi) / (2*torch.pi) * 48), 48).long()

        slot_next = (slot_cur + 1) % 48
        day_next  = day_cur + (slot_next == 0).long()

        # 3) (Opcional) filtra por range de dias do PASSO n+1
        if day_range is not None:
            keep = (day_next >= day_range[0]) & (day_next <= day_range[1])
        else:
            keep = torch.ones(B, dtype=torch.bool, device=device)

        if keep.sum().item() == 0:
            return 0

        # 4) Features temporais do n+1
        t_sin_next  = sin_lut[slot_next]
        t_cos_next  = cos_lut[slot_next]
        d_norm_next = day_next.float() / 74.0

        # 5) Forward 1-passo (n+1) no espaço NORMALIZADO
        model.eval()
        with torch.inference_mode():
            pred_t = model.forward_single_step(
                uid_t[keep], d_norm_next[keep], t_sin_next[keep], t_cos_next[keep],
                city_t[keep], poi_t[keep], coords_t[keep]
            )  # [b_keep, 2]
            if pred_t.ndim == 3 and pred_t.shape[1] == 1:
                pred_t = pred_t.squeeze(1)  

        # 6) Normalizado → grade (0..199) e discretização oficial
        def _denorm_xy(xy_norm_t: torch.Tensor, grid_size=200):
            return torch.clamp(xy_norm_t * (grid_size - 1), 0, grid_size - 1)

        pred_real = _denorm_xy(pred_t, grid_size)
        gt_keep   = target_t[keep]
        gt_real   = _denorm_xy(gt_keep, grid_size)

        pred_cell = discretize_coordinates(pred_t).to(torch.int32)
        gt_cell   = discretize_coordinates(gt_keep).to(torch.int32)

        # 7) Monta tabela e escreve
        def t2n(x): return x.detach().cpu().numpy()

        rows_dict = {
            "uid":           t2n(uid_t[keep]).astype(np.int64),
            "city":          t2n(city_t[keep]).astype(np.int32),
            "day":           t2n(day_next[keep]).astype(np.int32),
            "slot":          t2n(slot_next[keep]).astype(np.int32),
            "split":         np.array([split] * int(keep.sum().item())),
            "seq_len":       np.array([seq_len_ckpt] * int(keep.sum().item()), dtype=np.int32),

            "x_pred_norm": t2n(pred_t)[..., 0].astype(np.float32),
            "y_pred_norm": t2n(pred_t)[..., 1].astype(np.float32),
            "x_gt_norm":   t2n(gt_keep)[..., 0].astype(np.float32),
            "y_gt_norm":   t2n(gt_keep)[..., 1].astype(np.float32),

            "x_pred":      t2n(pred_real)[..., 0].astype(np.float32),
            "y_pred":      t2n(pred_real)[..., 1].astype(np.float32),
            "x_gt":        t2n(gt_real)[..., 0].astype(np.float32),
            "y_gt":        t2n(gt_real)[..., 1].astype(np.float32),

            "x_pred_cell": t2n(pred_cell)[..., 0].astype(np.int32),
            "y_pred_cell": t2n(pred_cell)[..., 1].astype(np.int32),
            "x_gt_cell":   t2n(gt_cell)[..., 0].astype(np.int32),
            "y_gt_cell":   t2n(gt_cell)[..., 1].astype(np.int32),
        }

        table = pa.table(rows_dict, schema=schema)
        writer.write_table(table)
        return table.num_rows
    print(f"📦 Pred+GT → Parquet | split={split} | cities={target_cities} | batch={batch_size}")

    # ---------- 0) Schema do Parquet
    schema = pa.schema([
        ("uid", pa.int64()), ("city", pa.int32()),
        ("day", pa.int32()), ("slot", pa.int32()),
        ("split", pa.string()), ("seq_len", pa.int32()),
        ("x_pred_norm", pa.float32()), ("y_pred_norm", pa.float32()),
        ("x_gt_norm", pa.float32()),   ("y_gt_norm", pa.float32()),
        ("x_pred", pa.float32()),      ("y_pred", pa.float32()),
        ("x_gt", pa.float32()),        ("y_gt", pa.float32()),
        ("x_pred_cell", pa.int32()),   ("y_pred_cell", pa.int32()),
        ("x_gt_cell", pa.int32()),     ("y_gt_cell", pa.int32()),
    ])
    out_path = Path(output_file)
    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path.as_posix(), schema, compression="zstd", use_dictionary=True)

    # ---------- 1) Carrega checkpoint/modelo (mesma lógica da submissão)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    centers_obj = ckpt.get("centers", None) or ckpt.get("extra", {}).get("centers", None)
    if centers_obj is None:
        raise RuntimeError("Centers não encontrados no checkpoint.")
    centers = centers_obj.to(device) if isinstance(centers_obj, torch.Tensor) \
             else torch.as_tensor(centers_obj, dtype=torch.float32, device=device)

    cfg   = ckpt.get("config", ckpt.get("extra", {}).get("config", {}))
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    seq_len_ckpt = int(cfg.get("sequence_length", sequence_length))

    model = HuMobModel(
        n_users=int(cfg.get("n_users", 100_000)),
        n_cities=int(cfg.get("n_cities", 4)),
        cluster_centers=centers,
        sequence_length=seq_len_ckpt,
        prediction_steps=int(cfg.get("prediction_steps", 1)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---------- 2) Dataset com alvo (val/test)
    ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=target_cities,
        mode=split,                   # "val" ou "test"
        sequence_length=seq_len_ckpt,
        max_sequences_per_user=float('inf'),
    )

    # Lookups de seno/cosseno dos 48 slots
    slots = torch.arange(48, device=device).float()
    phase_lut = 2 * torch.pi * (slots / 48.0)
    sin_lut, cos_lut = torch.sin(phase_lut), torch.cos(phase_lut)

    # ---------- 3) Loop em batches
    buf = []
    total_rows = 0

    with torch.inference_mode():
        for sample in tqdm(ds, desc=f"pred-gt-{split}"):
            buf.append(sample)

            if len(buf) >= batch_size:
                total_rows += _process_buffer(
                    buf,
                    model=model, device=device, writer=writer,
                    sin_lut=sin_lut, cos_lut=cos_lut,
                    seq_len_ckpt=seq_len_ckpt, split=split,
                    grid_size=grid_size, day_range=day_range
                )
                buf.clear()

    # restos (se sobrou < batch_size, processa igual)
    if buf:
        total_rows += _process_buffer(
            buf,
            model=model, device=device, writer=writer,
            sin_lut=sin_lut, cos_lut=cos_lut,
            seq_len_ckpt=seq_len_ckpt, split=split,
            grid_size=grid_size, day_range=day_range
        )
        buf.clear()

    writer.close()
    print(f"✅ Pred+GT salvo: {out_path} • linhas={total_rows:,}")
    return {"path": out_path.as_posix(), "rows": total_rows}



# Função principal de execução
def main():
    """Execução principal do pipeline HuMob."""
    
    # Configurações
    parquet_file = "humob_all_cities_v2_normalized.parquet"  # Seu arquivo normalizado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 HUMOB CHALLENGE - PIPELINE PRINCIPAL")
    print(f"Arquivo: {parquet_file}")
    print(f"Device: {device}")
    
    # Executa pipeline completo
    results = run_full_pipeline(
        parquet_path=parquet_file,
        device=device,
        n_clusters=256,        # Reduzido para ser mais rápido
        n_epochs=3,           # Poucas épocas para teste
        sequence_length=12,   # Histórico de 12 slots (6 horas)
        batch_size=64,
        learning_rate=1e-3,
        n_users_A=100_000
    )
    
    # Se o treinamento foi bem-sucedido, gera submissão
    # Depois de gerar a submissão:
    if results:
        print("\n📄 Gerando submissão HuMob...")
        submission_df = generate_humob_submission(
            parquet_path=parquet_file,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            submission_days=(61, 75),
            sequence_length=12,
            output_file=f"outputs/submissions/humob_submission_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
        )

        # 👇 NOVO: exporta Parquet com Pred+GT para análise posterior (MSE/cell error etc.)
        print("\n📦 Exportando Pred+GT (1 passo) para análise...")
        pairs_info = generate_pred_gt_parquet(
            parquet_path=parquet_file,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            split="test",                     # ou "val"
            day_range=(61, 75),               # manter apenas os últimos 15 dias
            sequence_length=12,
            output_file="outputs/eval/pred_gt_pairs_test.parquet",
            batch_size=2048,
            chunk_size=500_000,
            grid_size=200,
        )
        print(f"✅ Pred+GT: {pairs_info['rows']:,} linhas em {pairs_info['path']}")

    
    return results


if __name__ == "__main__":
    results = main()