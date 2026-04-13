# run.py — runner genérico para YAML
from __future__ import annotations
import argparse, inspect, os
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch

# --- imports do seu projeto ---
from src.training.train import compute_cluster_centers, train_humob_model
from src.training.finetune import sequential_finetuning, compare_models_performance
from src.training.pipeline import generate_humob_submission, generate_pred_gt_parquet

# MLflow helper: suporta tanto helpers em nível de módulo quanto a classe
try:
    from src.utils.mlflow_tracker import setup_mlflow_for_humob  # helper (recomendado)
except Exception:
    from src.utils.mlflow_tracker import HuMobMLflowTracker  # fallback
    def setup_mlflow_for_humob(experiment_name: str, tracking_uri: str = "file:./mlruns"):
        return HuMobMLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)


def _get(cfg: Dict[str, Any], path: str, default=None):
    """Acesso seguro com 'a.b.c'."""
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def run_from_yaml(cfg_path: str) -> None:
    # ---- carregar config ----
    cfg = yaml.safe_load(open(cfg_path))
    print(f"🧩 Config: {cfg_path}")

    # ---- resolver caminhos ----
    # data.parquet_file pode vir só com o nome; resolvemos para 'data/<arquivo>' se necessário
    parquet_file = _get(cfg, "data.parquet_file")
    if parquet_file is None:
        raise ValueError("Faltou 'data.parquet_file' no YAML.")
    parquet_path = Path(parquet_file)
    if parquet_path.name == str(parquet_path):  # sem diretório
        parquet_path = Path("data") / parquet_path.name
    parquet_path = parquet_path.as_posix()

    models_dir = Path(_get(cfg, "outputs.models_path", "outputs/models")).as_posix()
    submissions_dir = Path(_get(cfg, "outputs.submissions_path", "outputs/submissions")).as_posix()
    plots_dir = Path(_get(cfg, "outputs.plots_path", "outputs/plots")).as_posix()
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    eval_dir = Path(_get(cfg, "outputs.eval_path", "outputs/eval")).as_posix()
    os.makedirs(eval_dir, exist_ok=True)

    # ---- device/hardware ----
    prefer_device = _get(cfg, "hardware.device", "cuda")
    device = torch.device(prefer_device if (prefer_device == "cpu" or torch.cuda.is_available()) else "cpu")
    num_workers = int(_get(cfg, "hardware.num_workers", 0))
    pin_memory = bool(_get(cfg, "hardware.pin_memory", True))

    # ---- experimento/MLflow ----
    exp_name = _get(cfg, "experiment_name", "humob2")
    tracking_uri = _get(cfg, "logging.tracking_uri", "file:./mlruns")
    tracker = setup_mlflow_for_humob(exp_name, tracking_uri)
    try:
        tracker.log_env_snapshot()
    except Exception as e:
        print(f"⚠️ snapshot do ambiente falhou: {e}")

    # ---- parâmetros do modelo/treino ----
    n_clusters = int(_get(cfg, "model.n_clusters", 32))
    n_users = int(_get(cfg, "model.n_users", 1000))
    seq_len = int(_get(cfg, "model.sequence_length", 12))
    # Fase 1: arch params (com defaults v1 para retrocompatibilidade)
    user_emb_dim = int(_get(cfg, "model.user_emb_dim", 4))
    city_emb_dim = int(_get(cfg, "model.city_emb_dim", 4))
    temporal_dim = int(_get(cfg, "model.temporal_dim", 8))
    poi_out_dim = int(_get(cfg, "model.poi_out_dim", 4))
    lstm_hidden = int(_get(cfg, "model.lstm_hidden", 4))
    fusion_dim = int(_get(cfg, "model.fusion_dim", 8))
    # Fase 2: loss alpha (peso do MSE auxiliar)
    loss_alpha = float(_get(cfg, "model.loss_alpha", 0.1))
    max_seq_per_user = int(_get(cfg, "model.max_sequences_per_user", 50))
    max_scheduled_p = float(_get(cfg, "model.max_scheduled_p", 0.0))
    max_users = int(_get(cfg, "model.max_users", 0))

    base_cfg = _get(cfg, "training.base", {}) or {}
    ft_cfg = _get(cfg, "training.finetune", {}) or {}

    base_epochs = int(base_cfg.get("n_epochs", 1))
    base_lr = float(base_cfg.get("learning_rate", 3e-4))
    base_bs = int(base_cfg.get("batch_size", 16))
    base_mixed = bool(base_cfg.get("mixed_precision", True))
    resume_checkpoint = bool(base_cfg.get("resume_from_checkpoint", True))
    max_train_batches = int(base_cfg.get("max_train_batches", 0))
    max_val_batches = int(base_cfg.get("max_val_batches", 0))

    ft_epochs = int(ft_cfg.get("n_epochs", 1))
    ft_lr = float(ft_cfg.get("learning_rate", 5e-5))
    ft_bs = int(ft_cfg.get("batch_size", max(1, base_bs // 2)))

    # ---- cidades ----
    # Suporte a ambos formatos:
    #   data.base_city + data.finetune_cities
    #   data.cities = {"base": "A", "finetune": ["B","C","D"]}
    base_city = _get(cfg, "data.base_city", None) or _get(cfg, "data.cities.base", "A")
    ft_cities = _get(cfg, "data.finetune_cities", None) or _get(cfg, "data.cities.finetune", ["B", "C", "D"])
    if isinstance(ft_cities, str):
        ft_cities = [ft_cities]

    print("⚙️  Parâmetros resolvidos:")
    print(f"   device={device}, num_workers={num_workers}, pin_memory={pin_memory}")
    print(f"   n_clusters={n_clusters}, n_users={n_users}, sequence_length={seq_len}")
    print(f"   arch: user_emb={user_emb_dim}, city_emb={city_emb_dim}, temporal={temporal_dim}, "
          f"poi_out={poi_out_dim}, lstm_hidden={lstm_hidden}, fusion={fusion_dim}")
    print(f"   loss_alpha={loss_alpha}, max_scheduled_p={max_scheduled_p}")
    print(f"   base: epochs={base_epochs}, bs={base_bs}, lr={base_lr}, amp={base_mixed}, city={base_city}")
    print(f"   ft:   epochs={ft_epochs},  bs={ft_bs},  lr={ft_lr},  cities={ft_cities}")

    # ===========================================================
    # 1) CENTROS (KMeans)
    # ===========================================================
    print("\n🏁 ETAPA 1/4 — Computando centros (KMeans)")
    centers_path = Path(models_dir) / f"centers_{base_city}_{n_clusters}.npy"
    centers = compute_cluster_centers(
        parquet_path=parquet_path,
        cities=[base_city],   # ou ft_cities também, se desejar: [base_city] + ft_cities
        n_clusters=n_clusters,
        save_path=centers_path.as_posix(),
    )
    print(f"   ✅ Centros: {centers.shape} -> {centers_path.name}")

    # ===========================================================
    # 2) TREINO BASE
    # ===========================================================
    print("\n🏋️ ETAPA 2/4 — Treinamento base")
    base_ckpt = Path(models_dir) / f"humob_model_{base_city}.pt"
    model, train_losses, val_losses = train_humob_model(
        parquet_path=parquet_path,
        cluster_centers=centers,
        device=device,
        cities=[base_city],
        n_epochs=base_epochs,
        learning_rate=base_lr,
        batch_size=base_bs,
        sequence_length=seq_len,
        n_users=n_users,
        save_path=base_ckpt.as_posix(),
        mlflow_tracker=tracker,
        user_emb_dim=user_emb_dim,
        city_emb_dim=city_emb_dim,
        temporal_dim=temporal_dim,
        poi_out_dim=poi_out_dim,
        lstm_hidden=lstm_hidden,
        fusion_dim=fusion_dim,
        loss_alpha=loss_alpha,
        max_sequences_per_user=max_seq_per_user,
        max_scheduled_p=max_scheduled_p,
        max_users=max_users,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        resume_from_checkpoint=resume_checkpoint,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    print(f"   ✅ Treino base concluído -> {base_ckpt.name}")

    # ===========================================================
    # 3) FINE-TUNING (sequencial) — genérico
    # ===========================================================
    print("\n🎯 ETAPA 3/4 — Fine-tuning sequencial")
    ft_kwargs = dict(
        parquet_path=parquet_path,
        base_checkpoint=base_ckpt.as_posix(),
        cities=ft_cities,
        device=device,
        n_epochs_per_city=ft_epochs,
        learning_rate=ft_lr,
        sequence_length=seq_len,
        mlflow_tracker=tracker,
        resume_from_checkpoint=resume_checkpoint,
    )
    # Se sequential_finetuning aceitar batch_size, passamos. Se não, ignoramos.
    if "batch_size" in inspect.signature(sequential_finetuning).parameters:
        ft_kwargs["batch_size"] = ft_bs

    results = sequential_finetuning(**ft_kwargs)
    success = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"   ✅ Fine-tuning: {success}/{len(ft_cities)} cidades concluídas")

    # ===========================================================
    # 4) COMPARAÇÃO / SUBMISSÃO (opcionais)
    # ===========================================================
    do_compare = bool(_get(cfg, "eval.compare", True))
    do_submit  = bool(_get(cfg, "submission.enabled", False))
    do_predgt = bool(_get(cfg, "eval.pred_gt.enabled", False))
    predgt_split = _get(cfg, "eval.pred_gt.split", "val")   # "val" é mais seguro (tem GT)
    predgt_day_range = tuple(_get(cfg, "eval.pred_gt.day_range", (61, 75)))
    predgt_bs = int(_get(cfg, "eval.pred_gt.batch_size", 2048))
    predgt_cities = _get(cfg, "eval.pred_gt.target_cities", ft_cities)
    compare_samples = int(_get(cfg, "eval.n_samples", 1000))
    submission_days = tuple(_get(cfg, "submission.days", (61, 75)))
    target_submit_cities = _get(cfg, "submission.target_cities", ft_cities)

    final_ckpt = next(
        (res["checkpoint"] for _, res in reversed(list(results.items()))
        if res.get("status") == "success"),
        base_ckpt.as_posix(),
    )

    if do_compare:
        print("\n📊 ETAPA 4/4 — Comparação de modelos")
        checkpoints = {f"Zero-shot ({base_city})": base_ckpt.as_posix()}
        for city, res in results.items():
            if res.get("status") == "success":
                checkpoints[f"FT {city}"] = res["checkpoint"]

        comparison = compare_models_performance(
            parquet_path=parquet_path,
            checkpoints=checkpoints,
            device=device,
            n_samples=compare_samples,
            mlflow_tracker=tracker,
        )
        print("   ✅ Comparação concluída")
    else:
        comparison = None
        print("\n📊 Comparação desativada pelo YAML")

    if do_submit:
        print("\n📄 Gerando submissão final")

        # pega o último FT bem-sucedido; senão usa o base
        # final_ckpt = next(
        #     (res["checkpoint"] for _, res in reversed(list(results.items()))
        #     if res.get("status") == "success"),
        #     base_ckpt.as_posix(),
        # )

        sub_path = Path(submissions_dir) / "humob_submission_parquet"
        out = generate_humob_submission(
            parquet_path=parquet_path,
            checkpoint_path=final_ckpt,
            device=device,
            target_cities=target_submit_cities,
            submission_days=submission_days,
            sequence_length=seq_len,
            output_file=sub_path.as_posix(),
            # opcionais (se quiser controlar pelo YAML):
            chunk_size=500_000,
            users_batch=512,
        )
        print(f"   ✅ Submissão gerada: {sub_path.name} ({out['rows']:,} linhas, {out['users']:,} usuários)")
    else:
        print("📄 Submissão desativada pelo YAML")

    if do_predgt:
        print("\n📦 Exportando Pred+GT (1 passo) para análise...")
        predgt_out = Path(eval_dir) / f"pred_gt_pairs_{predgt_split}.parquet"
        info = generate_pred_gt_parquet(
            parquet_path=parquet_path,
            checkpoint_path=final_ckpt,
            device=device,
            target_cities=predgt_cities,
            split=predgt_split,             # "val" (seguro) ou "test" (se tiver GT)
            day_range=predgt_day_range,     # p.ex. (61, 75)
            sequence_length=seq_len,
            output_file=predgt_out.as_posix(),
            batch_size=predgt_bs,
            grid_size=200,
        )
        print(f"   ✅ Pred+GT: {info['rows']:,} linhas -> {info['path']}")
    else:
        print("📦 Pred+GT desativado pelo YAML")


    print("\n🎉 FIM — pipeline concluído.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Arquivo YAML com as configurações")
    args = ap.parse_args()
    run_from_yaml(args.config)
