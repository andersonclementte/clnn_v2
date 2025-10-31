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
    output_file: str = "humob_submission.csv",
    chunk_size: int = 50_000,           # tamanho do flush para disco
    users_batch: int = 512              # tamanho do lote de usuários na GPU
):
    """
    Gera submissão HuMob (uid, day, slot, x, y) de forma eficiente (streaming).
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
    last_seq_by_uid: dict[int, dict] = {}

    observed_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=target_cities,
        mode="test",                   # usa tudo disponível
        sequence_length=seq_len_ckpt,
        max_sequences_per_user=float('inf')
    )

    for sample in tqdm(observed_ds, desc="scan"):
        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, _ = sample
        uid_int = int(uid.item())
        dn = float(d_norm.item())

        # slot a partir de seno/cosseno
        angle = math.atan2(float(t_sin.item()), float(t_cos.item()))     # [-π, π]
        angle = (angle + 2*math.pi) % (2*math.pi)                        # [0, 2π)
        last_slot = int(round(angle / (2*math.pi) * 48)) % 48

        cur = last_seq_by_uid.get(uid_int)
        if (cur is None) or (dn > cur["d_norm"]):
            last_seq_by_uid[uid_int] = {
                "d_norm": dn,
                "t_sin": float(t_sin.item()),
                "t_cos": float(t_cos.item()),
                "city": int(city.item()),
                "poi_norm": poi_norm.detach().cpu().numpy(),
                "coords_seq": coords_seq.detach().cpu().numpy(),
                "last_observed_day": float(dn * 74),
                "last_observed_slot": last_slot,
            }

    n_users = len(last_seq_by_uid)
    print(f"📊 Usuários únicos coletados: {n_users:,}")

    # ---------- 3) Escrita incremental de CSV
    submission_rows = []
    first_chunk = True
    total_rows  = 0

    def flush_rows():
        nonlocal submission_rows, first_chunk, total_rows
        if not submission_rows:
            return
        df_chunk = pd.DataFrame(submission_rows).sort_values(["uid","day","slot"])
        df_chunk.to_csv(output_file, index=False,
                        mode=("w" if first_chunk else "a"),
                        header=first_chunk)
        total_rows += len(df_chunk)
        submission_rows.clear()
        first_chunk = False

    # ---------- 4) Inferência em lote por usuários
    uids = list(last_seq_by_uid.keys())
    total_slots = (submission_days[1] - submission_days[0] + 1) * 48
    print(f"🚀 Inferindo em lotes de {users_batch} usuários • total_slots={total_slots}")

    with torch.inference_mode():
        for i in tqdm(range(0, n_users, users_batch), desc="infer"):
            batch_ids = uids[i:i+users_batch]
            batch = [last_seq_by_uid[u] for u in batch_ids]

            city_tensor = torch.as_tensor(np.asarray([b["city"] for b in batch], dtype=np.int64), device=device)
            uid_tensor  = torch.as_tensor(np.asarray(batch_ids, dtype=np.int64), device=device)
            poi_arr   = np.ascontiguousarray(np.stack([b["poi_norm"]  for b in batch], axis=0), dtype=np.float32)
            coords_arr= np.ascontiguousarray(np.stack([b["coords_seq"] for b in batch], axis=0), dtype=np.float32)
            poi_tensor   = torch.from_numpy(poi_arr)
            coords_seq_t = torch.from_numpy(coords_arr)

            # Transferência para GPU
            if device.type == "cuda":
                poi_tensor   = poi_tensor.pin_memory().to(device, non_blocking=True)
                coords_seq_t = coords_seq_t.pin_memory().to(device, non_blocking=True)
            else:
                poi_tensor   = poi_tensor.to(device)
                coords_seq_t = coords_seq_t.to(device)

            current_day_vec  = torch.full((len(batch_ids),), submission_days[0],
                              dtype=torch.long, device=device)
            current_slot_vec = torch.zeros_like(current_day_vec)


            for step in range(total_slots):
                d_norm_current = (current_day_vec.float() / 74.0)
                phase = 2 * torch.pi * (current_slot_vec.float() / 48.0)
                t_sin_current = torch.sin(phase)
                t_cos_current = torch.cos(phase)

                pred = model.forward_single_step(
                    uid_tensor, d_norm_current, t_sin_current, t_cos_current,
                    city_tensor, poi_tensor, coords_seq_t
                )

                # discretização (ajuste grid_size se sua função aceitar)
                pred_disc = discretize_coordinates(pred)

                # acumula linhas
                for j, u in enumerate(batch_ids):
                    city_id = int(city_tensor[j].item())
                    submission_rows.append({
                        "uid": int(u),
                        "city": city_id,
                        "day": int(current_day_vec[j].item()),
                        "slot": int(current_slot_vec[j].item()),
                        "x": int(pred_disc[j, 0].item()),
                        "y": int(pred_disc[j, 1].item()),
                    })

                # teacher forcing batelado
                new_points = pred.clamp(0.0, 1.0).unsqueeze(1)   # [B,1,2]
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

    print(f"💾 Submissão salva em: {output_file}")
    print(f"   Linhas escritas: {total_rows:,} • Usuários: {n_users:,}")
    return {"path": output_file, "rows": total_rows, "users": n_users}



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
    if results:
        print("\n📄 Gerando submissão HuMob...")
        submission_df = generate_humob_submission(
            parquet_path=parquet_file,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            submission_days=(61, 75),
            sequence_length=12,
            output_file=f"humob_submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        
        print(f"\n🎉 PIPELINE COMPLETO!")
        print(f"✅ Modelo treinado e salvo")
        print(f"✅ Submissão gerada: {len(submission_df):,} predições")
        print(f"✅ Pronto para envio ao HuMob Challenge!")
    
    return results


if __name__ == "__main__":
    results = main()