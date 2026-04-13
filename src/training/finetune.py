import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

from src.models.humob_model import HuMobModel
from src.training.train import find_nearest_cluster
from src.data.dataset import HuMobNormalizedDataset

from src.utils.mlflow_tracker import HuMobMLflowTracker
from src.utils.simple_checkpoint import (
    save_checkpoint,
    load_training_checkpoint,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
)

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
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

def finetune_model(
    parquet_path: str,
    pretrained_checkpoint: str,
    target_city: str,
    device: torch.device,
    n_epochs: int = 3,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    sequence_length: int = 24,
    save_path: str = None,
    data_split: tuple = (0.0, 0.8),
    mlflow_tracker: HuMobMLflowTracker = None,
    base_run_id: str = None,
    resume_from_checkpoint: bool = True,
    checkpoint_every_n_epochs: int = 1,
    loss_alpha: float = 0.1,
    max_scheduled_p: float = 0.0,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
):
    """Fine-tuning do modelo pré-treinado em uma cidade específica."""
    print(f"🎯 FINE-TUNING NA CIDADE {target_city}")
    print("=" * 40)
    
    # Inicializar ft_run_id antes
    ft_run_id = None
    
    if mlflow_tracker is not None:
        config = {
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "data_split": f"{data_split[0]}-{data_split[1]}",
            "target_city": target_city
        }
        ft_run_id = mlflow_tracker.start_finetuning_run(
            base_run_id=base_run_id or "unknown",
            target_city=target_city,
            config=config
        )
        print(f"🔬 MLflow fine-tuning run iniciado: {ft_run_id}")
    
    # 1. Carrega modelo pré-treinado
    print("📂 Carregando modelo pré-treinado...")
    try:
        ckpt = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(pretrained_checkpoint, map_location=device)

    # centers: aceita raiz ou extra.centers; tensor ou numpy
    centers_obj = ckpt.get('centers', None)
    if centers_obj is None:
        centers_obj = ckpt.get('extra', {}).get('centers', None)
    if centers_obj is None:
        raise RuntimeError("Centers não encontrados no checkpoint base.")

    if isinstance(centers_obj, torch.Tensor):
        centers = centers_obj.to(device)
    else:
        centers = torch.as_tensor(centers_obj, dtype=torch.float32, device=device)

    # config (pode estar em raiz ou extra)
    config = ckpt.get('config', ckpt.get('extra', {}).get('config', {}))

    # pesos: aceita state_dict, model_state_dict, ou o próprio dict
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    
    # 2. Instancia modelo
    model = HuMobModel(
        n_users=int(config.get('n_users', 100_000)),
        n_cities=int(config.get('n_cities', 4)),
        cluster_centers=centers,
        sequence_length=sequence_length,
        prediction_steps=int(config.get('prediction_steps', 1)),
        user_emb_dim=int(config.get('user_emb_dim', 4)),
        city_emb_dim=int(config.get('city_emb_dim', 4)),
        temporal_dim=int(config.get('temporal_dim', 8)),
        poi_out_dim=int(config.get('poi_out_dim', 4)),
        lstm_hidden=int(config.get('lstm_hidden', 4)),
        fusion_dim=int(config.get('fusion_dim', 8)),
    ).to(device)

    model.load_state_dict(state, strict=True)
    max_scheduled_p = max_scheduled_p or float(config.get('max_scheduled_p', 0.0))
    print(f"✅ Modelo pré-treinado carregado (val_loss_base: {ckpt.get('val_loss', 'N/A')})")

    
    # 3. Cria loaders
    print(f"📊 Criando datasets para cidade {target_city}...")
    
    from torch.utils.data import DataLoader, IterableDataset
    
    train_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=[target_city],
        mode="train",
        sequence_length=sequence_length,
        train_days=data_split,
        val_days=(0.5, 1.0),
        max_sequences_per_user=30
    )
    
    val_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=[target_city],
        mode="val",
        sequence_length=sequence_length,
        train_days=data_split,
        val_days=(0.5, 1.0),
        max_sequences_per_user=10
    )
    
    # CORREÇÃO A: Loaders com shuffle e pin_memory
    pin = (device.type == "cuda")
    is_iter = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=0,
        shuffle=(False if is_iter else True), pin_memory=pin
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=0,
        shuffle=False, pin_memory=pin
    )
    
    # 4. Setup de fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=learning_rate/10
    )
    
    # loss_alpha controla o peso do MSE auxiliar (CE e a loss principal)
    
    # Checkpoint
    start_epoch = 0
    checkpoint_dir = f"outputs/models/checkpoints/finetune_{target_city}/"

    if resume_from_checkpoint:
        latest = get_latest_checkpoint(checkpoint_dir, pattern="checkpoint_epoch_*.pt")
        if latest:
            try:
                print(f"🔄 Tentando retomar fine-tuning: {latest}")
                start_epoch, _ = load_training_checkpoint(
                    model, optimizer, latest, map_location="cpu", strict=True, scheduler=scheduler
                )
                print(f"✅ Checkpoint carregado: época {start_epoch}")
            except RuntimeError as e:
                print(f"⚠️  Checkpoint incompatível (arquitetura mudou): {e}")
                print("🚀 Ignorando checkpoint e iniciando do zero.")
                start_epoch = 0

    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    fusion_weights_history = []
    
    print(f"🔧 Setup fine-tuning:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {n_epochs}")
    print(f"   Data split: dias {data_split[0]:.1f}-{data_split[1]:.1f}")

    stopper = EarlyStopper(patience=2, min_delta=1e-4, mode='min')
    
    # 5. Loop de fine-tuning
    p_scheduled = 0.0
    for epoch in range(start_epoch, n_epochs):
        # === TREINO ===
        model.train()
        train_loss_epoch = 0.0
        train_count = 0

        p_scheduled = max_scheduled_p * epoch / max(n_epochs - 1, 1) if n_epochs > 1 else 0.0
        print(f"\n🔄 Fine-tune Época {epoch+1}/{n_epochs} - Cidade {target_city}")
        train_pbar = tqdm(train_loader, desc=f'Finetune {target_city}')

        for batch_idx, batch in enumerate(train_pbar):
            try:
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [
                    b.to(device) for b in batch
                ]

                # sanity checks mínimos
                if (not torch.isfinite(coords_seq).all()) or (not torch.isfinite(target_coords).all()):
                    continue

                optimizer.zero_grad(set_to_none=True)

                target = target_coords.squeeze(1)
                input_seq = coords_seq
                if p_scheduled > 0 and torch.rand(1).item() < p_scheduled and coords_seq.size(1) > 1:
                    with torch.no_grad():
                        pred_ctx = model.forward_single_step(
                            uid, d_norm, t_sin, t_cos, city, poi_norm,
                            coords_seq[:, :-1, :]
                        )
                    input_seq = coords_seq.clone()
                    input_seq[:, -1, :] = pred_ctx.detach()

                pred, logits = model.forward_single_step(
                    uid, d_norm, t_sin, t_cos, city, poi_norm, input_seq,
                    return_logits=True
                )

                if not torch.isfinite(pred).all():
                    continue

                # Loss: CE sobre cluster + MSE auxiliar
                target_cluster = find_nearest_cluster(target, model.destination_head.centers)
                loss_ce = F.cross_entropy(logits, target_cluster)
                loss_mse = F.mse_loss(pred, target)
                loss = loss_ce + loss_alpha * loss_mse
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                bs = target.size(0)
                train_loss_epoch += float(loss.item()) * bs
                train_count += bs

                if batch_idx % 50 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.5f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                if max_train_batches > 0 and batch_idx + 1 >= max_train_batches:
                    break
            except Exception as e:
                print(f"⚠️ Erro batch {batch_idx}: {e}")
                continue

        # === VALIDAÇÃO ===
        model.eval()
        val_loss_epoch = 0.0
        val_count = 0

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(tqdm(val_loader, desc='Val')):
                try:
                    uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = [
                        b.to(device) for b in batch
                    ]
                    pred, logits_v = model.forward_single_step(
                        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq,
                        return_logits=True
                    )
                    target = target_coords.squeeze(1)
                    target_cluster_v = find_nearest_cluster(target, model.destination_head.centers)
                    loss_ce_v = F.cross_entropy(logits_v, target_cluster_v)
                    loss_mse_v = F.mse_loss(pred, target)
                    vloss = loss_ce_v + loss_alpha * loss_mse_v

                    bs = target.size(0)
                    val_loss_epoch += float(vloss.item()) * bs
                    val_count += bs
                    if max_val_batches > 0 and val_batch_idx + 1 >= max_val_batches:
                        break
                except Exception:
                    continue

        # === MÉDIAS ===
        avg_train_loss = train_loss_epoch / max(train_count, 1)
        avg_val_loss   = val_loss_epoch   / max(val_count,   1)

        # Early stopping
        stopper.update(avg_val_loss)
        if stopper.should_stop:
            print(f"ℹ️ Early stopping em {target_city}: sem melhoria por {stopper.patience} épocas (best={stopper.best:.5f})")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Pesos de fusão (escalares)
        w_r = float(model.weighted_fusion.w_r.item())
        w_e = float(model.weighted_fusion.w_e.item())
        fusion_weights_history.append({"w_r": w_r, "w_e": w_e})

        print(f"Cidade {target_city} - Treino: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        # CORREÇÃO B: Checkpoint condicional usando checkpoint_every_n_epochs
        if (epoch + 1) % checkpoint_every_n_epochs == 0:
            ckpt_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=avg_val_loss,
                save_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1:03d}.pt",
                extra_data={
                    'centers': centers.detach().cpu(),     # tensor, não numpy
                    'config':  config,
                    'city':    str(target_city),
                    'finetuned_from': str(pretrained_checkpoint),
                },
            )
            cleanup_old_checkpoints(checkpoint_dir, keep_last=3, pattern="checkpoint_epoch_*.pt")

            # === MLflow (não pode derrubar treino) ===
            try:
                if mlflow_tracker is not None:
                    mlflow_tracker.log_model_artifact(
                        model, ckpt_path, artifact_path=f"finetune_{target_city}/checkpoints"
                    )
                    mlflow_tracker.log_training_metrics(
                        step=epoch,
                        train_loss=float(avg_train_loss),
                        val_loss=float(avg_val_loss),
                        lr=float(optimizer.param_groups[0]["lr"]),
                        fusion_w_r=w_r,
                        fusion_w_e=w_e,
                    )
            except Exception as e:
                print(f"[MLflow] logging ignorado: {e}")

        # === Scheduler UMA vez por época ===
        if scheduler is not None:
            scheduler.step()

        # === Best model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_path is None:
                save_path = f"humob_model_finetuned_{target_city}.pt"

            print(f"💾 Novo melhor modelo para {target_city}! Loss: {best_val_loss:.5f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'extra': {
                    'centers': centers.detach().cpu(),
                    'config': {
                        **config,
                        'sequence_length': sequence_length,  # garante consistência
                    },
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(avg_val_loss),
                    'epoch': int(epoch),
                    'city': str(target_city),
                    'finetuned_from': str(pretrained_checkpoint),
                }
            }, save_path)

            try:
                if mlflow_tracker is not None:
                    mlflow_tracker.log_model_artifact(
                        model, save_path, artifact_path=f"finetune_{target_city}/best"
                    )
            except Exception as e:
                print(f"[MLflow] log best ignorado: {e}")

    # CORREÇÃO C: DEDENT - Tudo abaixo sai do loop de épocas
    # --- FIM DO LOOP DE ÉPOCAS ---
    
    # Último checkpoint (se existir)
    last_ckpt_path = get_latest_checkpoint(checkpoint_dir, pattern="checkpoint_epoch_*.pt")

    # Plots/artefatos no MLflow (se seu tracker tiver esse método)
    if mlflow_tracker is not None and hasattr(mlflow_tracker, "create_training_plots"):
        try:
            mlflow_tracker.create_training_plots(
                train_losses=train_losses,
                val_losses=val_losses,
                fusion_weights_history=fusion_weights_history
            )
        except Exception as e:
            print(f"[MLflow] plots ignorados: {e}")

    # Fechar o run do MLflow deste fine-tune
    if mlflow_tracker is not None and hasattr(mlflow_tracker, "end_run"):
        try:
            mlflow_tracker.end_run()
        except Exception as e:
            print(f"[MLflow] end_run ignorado: {e}")

    # Resumo
    print(f"\n✅ Fine-tuning cidade {target_city} concluído!")
    if save_path is not None:
        print(f"💾 Melhor modelo: {save_path}")
    if last_ckpt_path is not None:
        print(f"🧩 Último checkpoint: {last_ckpt_path}")
    print(f"📈 Melhor val loss: {best_val_loss:.5f}")

    return model, train_losses, val_losses




def sequential_finetuning(
    parquet_path: str,
    base_checkpoint: str,
    resume_from_checkpoint: bool = True,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
    cities: list[str] = ["B", "C", "D"],
    device: torch.device = None,
    n_epochs_per_city: int = 3,
    learning_rate: float = 1e-4,
    sequence_length: int = 24,
    # MLflow - Parâmetro para tracker
    mlflow_tracker: HuMobMLflowTracker = None
):
    """
    Fine-tuning sequencial em múltiplas cidades com tracking MLflow.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔄 FINE-TUNING SEQUENCIAL")
    print("=" * 50)
    print(f"Base model: {base_checkpoint}")
    print(f"Cities: {' → '.join(cities)}")
    print(f"Epochs per city: {n_epochs_per_city}")
    
    # MLflow - Extrai run ID do modelo base
    base_run_id = None
    if os.path.exists(base_checkpoint):
        try:
            base_ckpt = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
            base_run_id = base_ckpt.get('mlflow_run_id')
            if base_run_id:
                print(f"🔗 Linked to base run: {base_run_id}")
        except:
            pass
    
    results = {}
    current_checkpoint = base_checkpoint
    
    for i, city in enumerate(cities):
        print(f"\n🎯 Fine-tuning cidade {city} ({i+1}/{len(cities)})")
        
        # Nome do checkpoint desta cidade
        city_checkpoint = f"humob_model_finetuned_{city}.pt"
        
        try:
            # Fine-tuning
            model, train_losses, val_losses = finetune_model(
                parquet_path=parquet_path,
                pretrained_checkpoint=current_checkpoint,
                target_city=city,
                device=device,
                n_epochs=n_epochs_per_city,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                save_path=city_checkpoint,
                resume_from_checkpoint=resume_from_checkpoint,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
                # MLflow - Passa tracker e base run ID
                mlflow_tracker=mlflow_tracker,
                base_run_id=base_run_id,
            )
            
            results[city] = {
                'checkpoint': city_checkpoint,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'status': 'success'
            }
            
            # Próxima cidade usa o modelo desta cidade
            current_checkpoint = city_checkpoint
            
        except Exception as e:
            print(f"❌ Erro no fine-tuning da cidade {city}: {e}")
            results[city] = {
                'checkpoint': None,
                'train_losses': [],
                'val_losses': [],
                'final_val_loss': float('inf'),
                'status': 'failed',
                'error': str(e)
            }
    
    # Relatório final
    print(f"\n📊 RELATÓRIO SEQUENCIAL")
    print("=" * 40)
    
    successful_cities = 0
    for city, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {city}: Loss final = {result['final_val_loss']:.5f}")
            successful_cities += 1
        else:
            print(f"❌ {city}: Falhou ({result.get('error', 'Unknown error')})")
    
    print(f"\n🎉 Conclusão: {successful_cities}/{len(cities)} cidades com sucesso")
    
    if successful_cities == len(cities):
        final_model = f"humob_model_finetuned_{cities[-1]}.pt"
        print(f"🏆 Modelo final disponível: {final_model}")
    
    return results


def compare_models_performance(
    parquet_path: str,
    checkpoints: dict,
    test_cities: list[str] = ["B", "C", "D"],
    device: torch.device = None,
    n_samples: int = 2000,
    # MLflow - Parâmetro para tracker
    mlflow_tracker: HuMobMLflowTracker = None
):
    """
    Compara performance de múltiplos modelos com tracking MLflow.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 40)
    
    # MLflow - Inicia run de comparação
    if mlflow_tracker is not None:
        comp_run_id = mlflow_tracker.start_comparison_run(checkpoints)
        print(f"🔬 MLflow comparison run iniciado: {comp_run_id}")
    
    from src.training.train import evaluate_model
    
    results = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n🔍 Avaliando {model_name}...")
        
        # Determina tipo do modelo baseado no nome
        model_type = "fine_tuned" if any(k in model_name.upper() for k in ["FT", "FINE"]) else "zero_shot"
                
        results[model_name] = {}

        import mlflow as _mlflow
        _run_name = f"eval_{model_name.replace(' ', '_')}"
        _run_ctx = _mlflow.start_run(run_name=_run_name, nested=True) if mlflow_tracker is not None else None
        if _run_ctx is not None:
            _mlflow.set_tags({"stage": "evaluation", "model_name": model_name, "model_type": model_type})

        for city in test_cities:
            try:
                mse, cell_error = evaluate_model(
                    parquet_path=parquet_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    cities=[city],
                    n_samples=n_samples,
                    mlflow_tracker=mlflow_tracker,
                    model_type=model_type
                )
                results[model_name][city] = {'mse': mse, 'cell_error': cell_error}
                if _run_ctx is not None:
                    _mlflow.log_metrics({f"mse_{city}": mse, f"cell_error_{city}": cell_error})
                print(f"   {city}: MSE={mse:.4f}, Erro células={cell_error:.2f}")

            except Exception as e:
                print(f"   ❌ {city}: Erro - {e}")
                results[model_name][city] = {'mse': float('inf'), 'cell_error': float('inf')}

        if _run_ctx is not None:
            _mlflow.end_run()
    
    # MLflow - Log comparação completa
    if mlflow_tracker is not None:
        mlflow_tracker.log_model_comparison(results)
        mlflow_tracker.create_results_comparison_plot(results, "cell_error")
    
    # Relatório comparativo
    print(f"\n📋 COMPARAÇÃO DETALHADA")
    print("=" * 50)
    
    for city in test_cities:
        print(f"\n🏙️ Cidade {city}:")
        city_results = []
        for model_name in checkpoints.keys():
            if city in results[model_name]:
                mse = results[model_name][city]['mse']
                cell_err = results[model_name][city]['cell_error']
                city_results.append((model_name, mse, cell_err))
        
        # Ordena por MSE
        city_results.sort(key=lambda x: x[1])
        
        for i, (model_name, mse, cell_err) in enumerate(city_results):
            ranking = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}º"
            print(f"  {ranking} {model_name:20s}: MSE={mse:.4f}, Células={cell_err:.2f}")
    
    # MLflow - Log resumo final para o paper
    if mlflow_tracker is not None:
        # Calcula melhoria do fine-tuning
        zero_shot_results = {k: v for k, v in results.items() if "zero" in k.lower()}
        ft_results = {k: v for k, v in results.items() if "fine" in k.lower()}
        
        if zero_shot_results and ft_results:
            # Pega melhor resultado de cada tipo
            zero_shot_avg = np.mean([
                np.mean([city_data['cell_error'] for city_data in model_data.values() 
                        if city_data['cell_error'] != float('inf')])
                for model_data in zero_shot_results.values()
            ])
            
            ft_avg = np.mean([
                np.mean([city_data['cell_error'] for city_data in model_data.values()
                        if city_data['cell_error'] != float('inf')])  
                for model_data in ft_results.values()
            ])
            
            improvement_pct = ((zero_shot_avg - ft_avg) / zero_shot_avg) * 100
            
            mlflow_tracker.log_paper_summary({
                "final_avg_error_km": ft_avg * 0.5,
                "zero_shot_error_km": zero_shot_avg * 0.5,
                "improvement_pct": improvement_pct,
                "n_experiments": len(checkpoints)
            })
    
    return results


# Função principal para execução completa
def main():
    """Execução do fine-tuning completo com MLflow."""
    
    # MLflow - Configura tracker
    mlflow_tracker = HuMobMLflowTracker(experiment_name="HuMob_Challenge_Paper")
    
    # Configurações - AJUSTE AQUI
    parquet_file = "humob_all_cities_v2_normalized.parquet"
    base_model = "humob_model_A.pt"  # Modelo treinado apenas em A
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 FINE-TUNING HUMOB CHALLENGE COM MLFLOW")
    print("=" * 60)
    
    # Verifica arquivos
    import os
    if not os.path.exists(parquet_file):
        print(f"❌ Dados não encontrados: {parquet_file}")
        return
    
    if not os.path.exists(base_model):
        print(f"❌ Modelo base não encontrado: {base_model}")
        print("Execute primeiro o treinamento com run_humob.py")
        return
    
    # Fine-tuning sequencial
    results = sequential_finetuning(
        parquet_path=parquet_file,
        base_checkpoint=base_model,
        cities=["B", "C", "D"],
        device=device,
        n_epochs_per_city=3,
        learning_rate=5e-5,
        sequence_length=24,
        # MLflow - Passa tracker
        mlflow_tracker=mlflow_tracker
    )
    
    # Comparação de performance se tudo deu certo
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    if successful > 0:
        print(f"\n🔍 COMPARANDO PERFORMANCE...")
        
        # Monta lista de checkpoints para comparar
        checkpoints = {'Zero-shot (A apenas)': base_model}
        
        for city, result in results.items():
            if result['status'] == 'success':
                checkpoints[f'Fine-tuned {city}'] = result['checkpoint']
        
        comparison = compare_models_performance(
            parquet_path=parquet_file,
            checkpoints=checkpoints,
            device=device,
            n_samples=3000,
            # MLflow - Passa tracker
            mlflow_tracker=mlflow_tracker
        )
        
        print("\n🎉 FINE-TUNING COMPLETO!")
        print("Agora você pode usar os modelos fine-tuned para submissão.")
        print("🔬 Dados salvos no MLflow - execute 'mlflow ui' para visualizar")
    
    return results


if __name__ == "__main__":
    main()