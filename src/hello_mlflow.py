# src/hello_mlflow.py
import mlflow, time
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("humob2")
run = mlflow.start_run(run_name="hello")
try:
    mlflow.log_param("who", "nitro")
    for step in range(10):
        mlflow.log_metric("dummy_loss", 1.0/(step+1), step=step)
        time.sleep(0.05)
    # exemplo de nested run (evita “already active” quando iterar cidades)
    with mlflow.start_run(run_name="nested_example", nested=True):
        mlflow.log_metric("inner_metric", 0.123)
finally:
    mlflow.end_run()
print("OK")
