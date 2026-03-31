"""
Re-avaliação dos modelos treinados com logging correto no MLflow.
Roda sem re-treinar — usa modelos já salvos.
"""
import os, torch, mlflow
os.chdir("/workspace")

from src.training.train import evaluate_model

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("humob2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parquet = "data/humob_all_cities_v2_normalized.parquet"

models = {
    "Zero-shot_A": ("outputs/models/humob_model_A.pt",   ["B", "C", "D"], "zero_shot"),
    "FT-B":        ("humob_model_finetuned_B.pt",         ["B"],           "fine_tuned"),
    "FT-C":        ("humob_model_finetuned_C.pt",         ["C"],           "fine_tuned"),
    "FT-D":        ("humob_model_finetuned_D.pt",         ["D"],           "fine_tuned"),
}

all_results = {}

with mlflow.start_run(run_name="comparison_corrected") as parent:
    mlflow.set_tags({"stage": "comparison_corrected"})

    for model_name, (ckpt, cities, model_type) in models.items():
        all_results[model_name] = {}
        with mlflow.start_run(run_name=f"eval_{model_name}", nested=True):
            mlflow.set_tags({"model_name": model_name, "model_type": model_type, "stage": "evaluation"})
            for city in cities:
                mse, cell_err = evaluate_model(parquet, ckpt, device, cities=[city], n_samples=1000)
                mlflow.log_metrics({f"mse_{city}": mse, f"cell_error_{city}": cell_err})
                all_results[model_name][city] = {"mse": mse, "cell_error": cell_err}
                print(f"  {model_name:15s} | {city} | MSE={mse:.5f} | cell_error={cell_err:.2f}")

print("\n=== COMPARAÇÃO ZERO-SHOT vs FINE-TUNED ===")
print(f"{'Cidade':<8} {'Zero-shot':>12} {'Fine-tuned':>12} {'Melhoria':>10}")
print("-" * 46)
for city in ["B", "C", "D"]:
    zs  = all_results["Zero-shot_A"][city]["cell_error"]
    ft  = all_results[f"FT-{city}"][city]["cell_error"]
    imp = (zs - ft) / zs * 100
    print(f"  {city:<6} {zs:>12.2f} {ft:>12.2f} {imp:>+9.1f}%")
