import subprocess, sys, os

_pip_target = "/tmp/pip_libs"
os.makedirs(_pip_target, exist_ok=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--target", _pip_target,
     "git+https://github.com/yahoojapan/geobleu.git", "fastdtw"],
    check=True,
)
if _pip_target not in sys.path:
    sys.path.insert(0, _pip_target)

os.chdir("/workspace")

import mlflow
import pandas as pd
from src.utils.metrics import compute_geobleu, compute_dtw

PARQUET  = "outputs/eval/pred_gt_all_users.parquet"
CITY_MAP = {1: "B", 2: "C", 3: "D"}

print("Carregando predições...")
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} linhas, {df['uid'].nunique():,} usuários únicos\n")

results = {}
for city_code, city_name in CITY_MAP.items():
    city_df = df[df["city"] == city_code].copy()
    n_users = city_df["uid"].nunique()
    print(f"Cidade {city_name} — {n_users} usuários, {len(city_df):,} pontos")

    print(f"  Calculando GEO-BLEU...")
    geo_bleu = compute_geobleu(city_df)

    print(f"  Calculando DTW...")
    dtw = compute_dtw(city_df)

    results[city_name] = {"geo_bleu": geo_bleu, "dtw": dtw, "n_users": n_users}
    print(f"  ✓ GEO-BLEU={geo_bleu:.4f}  DTW={dtw:.2f}\n")

# Resumo final
print("=" * 50)
print(f"{Cidade:<8} {GEO-BLEU:>10} {DTW:>10} {Usuários:>10}")
print("-" * 50)
for city_name, m in results.items():
    print(f"{city_name:<8} {m[geo_bleu]:>10.4f} {m[dtw]:>10.2f} {m[n_users]:>10}")
print("=" * 50)
print("Referência top HuMob 2024: GEO-BLEU=0.319 | DTW=27.15")

# Log to MLflow
mlflow.set_experiment("humob2")
with mlflow.start_run(run_name="official_metrics_FT"):
    for city_name, m in results.items():
        mlflow.log_metric(f"geo_bleu_{city_name}", m["geo_bleu"])
        mlflow.log_metric(f"dtw_{city_name}",      m["dtw"])
        mlflow.log_metric(f"n_users_{city_name}",  m["n_users"])
print("\nResultados registrados no MLflow (experimento humob2).")
