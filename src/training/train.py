import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.humob_model import HuMobModel, discretize_coordinates
from src.data.dataset import create_humob_loaders, create_test_loader

# MLflow - Import do tracker personalizado
from src.utils.mlflow_tracker import HuMobMLflowTracker

def find_nearest_cluster(target_coords: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
    """Retorna indice do cluster mais proximo para cada ponto alvo.
    Args:
        target_coords: (B, 2) coordenadas normalizadas
        cluster_centers: (C, 2) centros dos clusters
    Returns:
        (B,) indices LongTensor
    """
    dists = torch.cdist(target_coords.float(), cluster_centers.float())  # (B, C)
    return dists.argmin(dim=1)  # (B,)


from src.utils.simple_checkpoint import (
    save_checkpoint,
    load_training_checkpoint,
    get_latest_checkpoint
)

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.should_stop = False

    def update(self, value: float):
        improved = (value < self.best - self.min_delta) if self.mode == 'min' else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True



def compute_cluster_centers(
    parquet_path: str,
    cities: list[str] = ["A"],
    n_clusters: int = 1024,
    sample_size: int = 200_000,
    save_path: str = "cluster_centers.npy",
    chunk_size: int = 50_000,
    coord_columns: list = ['x_norm', 'y_norm']
) -> torch.Tensor:
    """
    Calcula centros de cluster usando K-Means em coordenadas normalizadas.
    CORRIGIDO: agora trabalha com dados já normalizados [0,1].
    """
    
    # Se já existe, carrega
    if os.path.exists(save_path):
        print(f"📂 Carregando centros existentes: {save_path}")
        centers = np.load(save_path)
        return torch.from_numpy(centers.astype(np.float32))
    
    print(f"🔄 Calculando {n_clusters} centros para cidades {cities}...")
    
    # 1) Coleta coordenadas normalizadas
    pf = pq.ParquetFile(parquet_path)
    coords_list = []
    
    city_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    target_city_codes = [city_map[c] for c in cities if c in city_map]
    
    for batch in pf.iter_batches(batch_size=chunk_size):
        table = batch.to_pandas()
        
        # Filtra cidades
        if 'city_encoded' in table.columns:
            mask = table['city_encoded'].isin(target_city_codes)
        elif 'city' in table.columns:
            mask = table['city'].isin(cities)
        else:
            continue
            
        table = table[mask]
        if len(table) == 0:
            continue
        
        # Extrai coordenadas normalizadas
        if all(col in table.columns for col in coord_columns):
            coords = table[coord_columns].values
            # Remove pontos inválidos (ex: 999 normalizado)
            valid_mask = (coords >= 0) & (coords <= 1)
            valid_mask = valid_mask.all(axis=1)
            coords = coords[valid_mask]
            
            if len(coords) > 0:
                coords_list.append(coords)
    
    if not coords_list:
        print("❌ Nenhuma coordenada válida encontrada!")
        # Fallback: grid regular
        x = np.linspace(0, 1, int(np.sqrt(n_clusters)))
        y = np.linspace(0, 1, int(np.sqrt(n_clusters)))
        xx, yy = np.meshgrid(x, y)
        centers = np.stack([xx.flatten(), yy.flatten()], axis=1)[:n_clusters]
        centers = centers.astype(np.float32)
    else:
        coords = np.vstack(coords_list)
        print(f"📊 Coletadas {len(coords):,} coordenadas válidas")
        
        # 2) Amostra se muito grande
        if len(coords) > sample_size:
            idx = np.random.choice(len(coords), sample_size, replace=False)
            coords = coords[idx]
            print(f"🎲 Amostradas {sample_size:,} coordenadas")
        
        # 3) K-Means
        print("⚙️ Executando K-Means...")
        kmeans = KMeans(
            n_clusters=n_clusters, 
            n_init='auto', 
            random_state=42,
            max_iter=300
        ).fit(coords)
        
        centers = kmeans.cluster_centers_.astype(np.float32)
    
    # 4) Salva para reutilizar
    np.save(save_path, centers)
    print(f"💾 Centros salvos em: {save_path}")
    print(f"📏 Shape dos centros: {centers.shape}")
    print(f"📍 Range dos centros: x=[{centers[:,0].min():.3f}, {centers[:,0].max():.3f}], y=[{centers[:,1].min():.3f}, {centers[:,1].max():.3f}]")
    
    return torch.from_numpy(centers)


def _cleanup_old_checkpoints(keep_n: int = 3, checkpoint_dir: str = "outputs/models/checkpoints/"):
    """Remove checkpoints antigos, mantendo apenas os últimos N."""
    from pathlib import Path
    
    checkpoints = sorted(
        Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"),
        key=lambda p: p.stat().st_mtime
    )
    
    # Remove os mais antigos
    for old_checkpoint in checkpoints[:-keep_n]:
        try:
            old_checkpoint.unlink()
            print(f"🗑️ Checkpoint antigo removido: {old_checkpoint.name}")
        except Exception as e:
            print(f"⚠️ Erro removendo {old_checkpoint.name}: {e}")


def train_humob_model(
    parquet_path: str,
    cluster_centers: torch.Tensor,
    device: torch.device,
    cities: list[str] = ["A"],
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    sequence_length: int = 24,
    n_users: int = 100_000,
    save_path: str = "humob_model.pt",
    mlflow_tracker: HuMobMLflowTracker = None,
    resume_from_checkpoint: bool = True,
    checkpoint_every_n_epochs: int = 1,
    keep_last_n_checkpoints: int = 3,
    user_emb_dim: int = 4,
    city_emb_dim: int = 4,
    temporal_dim: int = 8,
    poi_out_dim: int = 4,
    lstm_hidden: int = 4,
    fusion_dim: int = 8,
    loss_alpha: float = 0.1,
    max_sequences_per_user: int = 50,
    max_scheduled_p: float = 0.0,
    max_users: int = 0,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
):
    """Treina o modelo HuMob com dados normalizados e tracking MLflow."""
    print("🏋️ Iniciando treinamento do modelo HuMob...")
    
    # Inicializa run_id antes do if
    run_id = None
    
    # Configuração do experimento
    if mlflow_tracker is not None:
        config = {
            "n_clusters": cluster_centers.shape[0],
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "n_users": n_users,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau", 
            "device": str(device),
            "cities": cities
        }
        run_id = mlflow_tracker.start_base_training_run(config)
        print(f"🔬 MLflow run iniciado: {run_id}")
    
    # 1. Cria loaders
    train_loader, val_loader = create_humob_loaders(
        parquet_path=parquet_path,
        cities=cities,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_sequences_per_user=max_sequences_per_user,
        max_users=max_users,
    )
    
    # 2. Instancia modelo
    model = HuMobModel(
        n_users=n_users,
        n_cities=4,
        cluster_centers=cluster_centers,
        user_emb_dim=user_emb_dim,
        city_emb_dim=city_emb_dim,
        temporal_dim=temporal_dim,
        poi_out_dim=poi_out_dim,
        lstm_hidden=lstm_hidden,
        fusion_dim=fusion_dim,
        sequence_length=sequence_length,
        prediction_steps=1
    ).to(device)
    
    # 3. Setup de treino
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                 betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # loss_alpha controla o peso do MSE auxiliar (CE e a loss principal)

    def grad_norm(model):
        """Calcula norma total dos gradientes"""
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().norm(2).item() ** 2
        return (total ** 0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    fusion_weights_history = []
    
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    stopper = EarlyStopper(patience=3, min_delta=1e-4, mode='min')

    start_epoch = 0
    if resume_from_checkpoint:
        latest = get_latest_checkpoint("outputs/models/checkpoints", "checkpoint_epoch_*.pt")
        if latest:
            try:
                print(f"🔁 Tentando retomar de {latest}")
                start_epoch, _ = load_training_checkpoint(model, optimizer, latest, map_location="cpu", strict=True,
                                                        scheduler=scheduler if 'scheduler' in locals() else None)
                print(f"✅ Checkpoint carregado: época {start_epoch}")
                # Override lr do YAML (ignora lr salvo no checkpoint)
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate
                print(f"🔧 LR forçado para {learning_rate} (do config)")
            except RuntimeError as e:
                print(f"⚠️  Checkpoint incompatível (arquitetura mudou): {e}")
                print("🚀 Ignorando checkpoint e iniciando do zero.")
                start_epoch = 0
        else:
            print("🚀 Nenhum checkpoint encontrado, iniciando do zero")
    else:
        print("🚀 resume_from_checkpoint=False — treinando do zero")
    
    p_scheduled = 0.0  # Fase 4: cresce ate max_scheduled_p ao longo das epocas

    for epoch in range(start_epoch, n_epochs):
        # Fase 4: p_scheduled cresce linearmente de 0 ate max_scheduled_p
        p_scheduled = max_scheduled_p * epoch / max(n_epochs - 1, 1) if n_epochs > 1 else 0.0

        # === TREINO ===
        model.train()
        train_loss_epoch = 0
        train_count = 0
        
        # Inicializa variáveis que podem não ser definidas
        post_clip_norm = 0.0
        
        print(f"\n🔄 Época {epoch+1}/{n_epochs} | p_scheduled={p_scheduled:.2f}")
        train_pbar = tqdm(train_loader, desc=f'Treino {cities}')
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [b.to(device) for b in batch]
                
                # Verificação de dados de entrada
                if not torch.isfinite(coords_seq).all() or not torch.isfinite(target_coords).all():
                    print("⚠️ Dados contêm NaN/Inf, pulando batch")
                    continue
                
                optimizer.zero_grad(set_to_none=True)
                
                target = target_coords.squeeze(1)

                # Fase 4: Scheduled Sampling
                # Com prob. p_scheduled, substitui o ultimo passo de coords_seq
                # pela predicao do modelo, simulando o rollout de avaliacao.
                if p_scheduled > 0 and torch.rand(1).item() < p_scheduled and coords_seq.size(1) > 1:
                    with torch.no_grad():
                        pred_ctx = model.forward_single_step(
                            uid, d_norm, t_sin, t_cos, city, poi_norm,
                            coords_seq[:, :-1, :]  # prefixo sem o ultimo passo real
                        )
                    input_seq = coords_seq.clone()
                    input_seq[:, -1, :] = pred_ctx.detach()
                else:
                    input_seq = coords_seq

                # Forward unico retornando pred e logits (corrige double-pass anterior)
                pred, logits = model.forward_single_step(
                    uid, d_norm, t_sin, t_cos, city, poi_norm, input_seq,
                    return_logits=True
                )

                # Verificação de predição
                if not torch.isfinite(pred).all():
                    print("⚠️ Predição contém NaN/Inf, pulando batch")
                    continue

                # Loss: CE sobre cluster + MSE auxiliar
                target_cluster = find_nearest_cluster(target, model.destination_head.centers)
                loss_ce = F.cross_entropy(logits, target_cluster)
                loss_mse = F.mse_loss(pred, target)
                loss = loss_ce + loss_alpha * loss_mse
                
                if not torch.isfinite(loss):
                    print(f"⚠️ Loss={loss.item()} não finito, pulando batch")
                    continue
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                pre_clip_norm = grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                post_clip_norm = grad_norm(model)
                
                if pre_clip_norm > 10:
                    print(f"⚠️ Gradiente alto: pre={pre_clip_norm:.1f} → post={post_clip_norm:.2f}")

                optimizer.step()
                
                train_loss_epoch += loss.item() * target.size(0)
                train_count += target.size(0)
                
                # Logs
                if batch_idx % 100 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'GradNorm': f'{post_clip_norm:.2f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
                if max_train_batches > 0 and batch_idx + 1 >= max_train_batches:
                    break
            except Exception as e:
                print(f"❌ Erro no batch {batch_idx}: {e}")
                continue
        
        # === VALIDAÇÃO ===
        model.eval()
        val_loss_epoch = 0
        val_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Val')
            for val_batch_idx, batch in enumerate(val_pbar):
                try:
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [b.to(device) for b in batch]
                    
                    pred = model.forward_single_step(uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq)
                    target = target_coords.squeeze(1)
                    
                    loss = criterion(pred, target)
                    
                    val_loss_epoch += loss.item() * target.size(0)
                    val_count += target.size(0)
                    
                    val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                    if max_val_batches > 0 and val_batch_idx + 1 >= max_val_batches:
                        break
                except Exception as e:
                    continue
        
        # Médias
        avg_train_loss = train_loss_epoch / max(train_count, 1)
        avg_val_loss = val_loss_epoch / max(val_count, 1)

        stopper.update(avg_val_loss)
        if stopper.should_stop:
            print(f"⏹️ Early stopping: sem melhoria por {stopper.patience} épocas. Melhor val_loss={stopper.best:.4f}")
            break
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Captura pesos da fusão
        w_r = model.weighted_fusion.w_r.item()
        w_e = model.weighted_fusion.w_e.item()
        fusion_weights = {"w_r": w_r, "w_e": w_e}
        fusion_weights_history.append(fusion_weights)
        
        print(f"Treino: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        print(f"Fusion weights: w_r={w_r:.3f}, w_e={w_e:.3f}")
        
        # centers como tensor (sem numpy)
        centers_tensor = cluster_centers.detach().cpu()

        # 1) salvar checkpoint
        ckpt_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,        # se existir
            epoch=epoch,
            val_loss=avg_val_loss,
            save_dir="outputs/models/checkpoints/",
            filename=f"checkpoint_epoch_{epoch+1:03d}.pt",
            extra_data={
                'centers': centers_tensor,               # <- tensor, seguro
                'config': {
                    'n_users': n_users,
                    'n_cities': 4,
                    'sequence_length': sequence_length,
                    'prediction_steps': 1,
                    'n_clusters': int(centers_tensor.shape[0]),
                    'user_emb_dim': user_emb_dim,
                    'city_emb_dim': city_emb_dim,
                    'temporal_dim': temporal_dim,
                    'poi_out_dim': poi_out_dim,
                    'lstm_hidden': lstm_hidden,
                    'fusion_dim': fusion_dim,
                    'max_scheduled_p': max_scheduled_p,
                },
                'train_loss': float(avg_train_loss),
                'mlflow_run_id': run_id
            }
        )
        _cleanup_old_checkpoints(keep_last_n_checkpoints)

        # 2) depois loga no MLflow (try/except para nunca derrubar treino)
        try:
            if mlflow_tracker is not None:
                mlflow_tracker.log_model_artifact(model, ckpt_path, artifact_path="models")
                mlflow_tracker.log_training_metrics(
                    step=epoch,
                    train_loss=float(avg_train_loss),
                    val_loss=float(avg_val_loss),
                    grad_norm=float(post_clip_norm),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    fusion_w_r=float(w_r),
                    fusion_w_e=float(w_e),
                )
        except Exception as e:
            print(f"[MLflow] logging ignorado: {e}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Salva melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"💾 Novo melhor modelo! Loss: {best_val_loss:.4f}")
            
            torch.save({
                'state_dict': model.state_dict(),
                'centers': cluster_centers.cpu().numpy(),
                'config': {
                    'n_users': n_users,
                    'n_cities': 4,
                    'sequence_length': sequence_length,
                    'prediction_steps': 1,
                    'n_clusters': cluster_centers.shape[0],
                    'user_emb_dim': user_emb_dim,
                    'city_emb_dim': city_emb_dim,
                    'temporal_dim': temporal_dim,
                    'poi_out_dim': poi_out_dim,
                    'lstm_hidden': lstm_hidden,
                    'fusion_dim': fusion_dim,
                    'max_scheduled_p': max_scheduled_p,
                },
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'mlflow_run_id': run_id  # Agora sempre definido
            }, save_path)
            
            if mlflow_tracker is not None:
                mlflow_tracker.log_model_artifact(model, save_path)
    
    # MLflow plots finais
    if mlflow_tracker is not None:
        mlflow_tracker.create_training_plots(
            train_losses=train_losses,
            val_losses=val_losses,
            fusion_weights_history=fusion_weights_history
        )
    
    print(f"\n✅ Treinamento concluído! Modelo salvo em: {save_path}")
    print(f"Melhor loss de validação: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate_model(
    parquet_path: str,
    checkpoint_path: str,
    device: torch.device,
    cities: list[str] = ["D"],
    n_samples: int = 5000,
    sequence_length: int = 24,
    mlflow_tracker: HuMobMLflowTracker | None = None,
    model_type: str = "zero_shot",
):
    """
    Avalia o modelo em cidades específicas com tracking MLflow.
    Suporta checkpoints e 'best models' com diferentes layouts.
    """
    print(f"🎯 Avaliando modelo em cidades {cities}...")

    # --- imports locais para evitar dependências de topo ---
    import numpy as np
    import torch
    from torch import nn
    from tqdm import tqdm
    try:
        # suas funções/classes
        from src.models.humob_model import HuMobModel, discretize_coordinates
    except Exception:
        # se já estiverem importadas no módulo, ignore
        pass

    # 1) Carrega checkpoint (compatível com PyTorch >= 2.6)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    # 1.1) Recupera centers (aceita raiz ou extra.centers; tensor ou numpy)
    centers_obj = ckpt.get("centers", None)
    if centers_obj is None:
        centers_obj = ckpt.get("extra", {}).get("centers", None)
    if centers_obj is None:
        raise RuntimeError(
            "Centers não encontrados no checkpoint (nem em 'centers' nem em 'extra.centers')."
        )

    if isinstance(centers_obj, torch.Tensor):
        centers = centers_obj.to(device)
    else:
        centers = torch.as_tensor(centers_obj, dtype=torch.float32, device=device)

    # 1.2) Config (opcional) para n_users/n_cities/seq_len
    cfg = ckpt.get("config", ckpt.get("extra", {}).get("config", {}))
    n_users = int(cfg.get("n_users", 100_000))
    n_cities = int(cfg.get("n_cities", 4))
    seq_len_eval = int(cfg.get("sequence_length", sequence_length))

    # 1.3) State dict (aceita diferentes chaves)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        # fallback: checkpoint pode ser o próprio state_dict
        state = ckpt

    # 2) Instancia modelo e carrega pesos (lê arch params do config do checkpoint)
    model = HuMobModel(
        n_users=n_users,
        n_cities=n_cities,
        cluster_centers=centers,
        sequence_length=seq_len_eval,
        prediction_steps=1,
        user_emb_dim=int(cfg.get('user_emb_dim', 4)),
        city_emb_dim=int(cfg.get('city_emb_dim', 4)),
        temporal_dim=int(cfg.get('temporal_dim', 8)),
        poi_out_dim=int(cfg.get('poi_out_dim', 4)),
        lstm_hidden=int(cfg.get('lstm_hidden', 4)),
        fusion_dim=int(cfg.get('fusion_dim', 8)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) Loader de teste (use o mesmo seq_len do treino salvo se possível)
    test_loader = create_test_loader(  # presume que já existe essa função no seu código
        parquet_path=parquet_path,
        cities=cities,
        batch_size=32,
        sequence_length=seq_len_eval,
    )

    # 4) Avaliação
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0
    cell_err_sum = 0.0
    cell_err_count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Eval")
        for batch in pbar:
            if total_samples >= n_samples:
                break

            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [
                    b.to(device) for b in batch
                ]

                pred = model.forward_single_step(
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq
                )
                target = target_coords.squeeze(1)

                loss = criterion(pred, target)
                bs = target.size(0)
                total_loss += float(loss.item()) * bs
                total_samples += bs

                # Erro em células (média por amostra; depois média global)
                pred_disc = discretize_coordinates(pred)
                target_disc = discretize_coordinates(target)
                cell_err = (pred_disc - target_disc).abs().float().mean(dim=1)  # [bs]
                cell_err_sum += float(cell_err.sum().item())
                cell_err_count += int(cell_err.numel())

                pbar.set_postfix({"MSE": f"{loss.item():.4f}", "Samples": total_samples})

            except Exception as e:
                # pula batch problemático sem derrubar a avaliação
                continue

    avg_mse = total_loss / max(total_samples, 1)
    avg_cell_error = (cell_err_sum / max(cell_err_count, 1)) if cell_err_count else float("nan")

    print(f"\n📊 Resultados em {cities}:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  Erro médio em células: {avg_cell_error:.2f}")
    print(f"  Amostras avaliadas: {total_samples:,}")

    # 5) MLflow logging (somente escalares)
    if mlflow_tracker is not None:
        try:
            for c in cities:
                mlflow_tracker.log_evaluation_results(
                    city=c,
                    mse=float(avg_mse),
                    cell_error=float(avg_cell_error),
                    n_samples=int(total_samples),
                    model_type=model_type,
                )
        except Exception as e:
            print(f"[MLflow] logging de avaliação ignorado: {e}")

    return float(avg_mse), float(avg_cell_error)
