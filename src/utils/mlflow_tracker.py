from __future__ import annotations

import atexit
import json
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _flatten_params(d: Dict[str, Any], prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Achata dicionários aninhados para logar como params no MLflow.
    Converte apenas tipos simples; o restante vai para log_dict.
    """
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_params(v, key, out)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


class HuMobMLflowTracker:
    """
    Rastreador de experimentos “à prova de falhas”:
    - Garante que um run ativo exista apenas quando você quiser.
    - Usa nested=True para execuções filhas (e.g., fine-tuning, comparações).
    - Oferece utilitários para logar ambiente/requirements.
    """

    def __init__(self, experiment_name: str = "HuMob", tracking_uri: Optional[str] = None) -> None:
        self.experiment_name = experiment_name
        self.run_id: Optional[str] = None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        atexit.register(self._safe_end)

    # ------------- ciclo de vida -----------------

    def _safe_end(self) -> None:
        """Fecha um run ativo silenciosamente (usado no atexit)."""
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def end_run(self) -> None:
        """Finalize explicitamente o run atual."""
        if mlflow.active_run() is not None:
            mlflow.end_run()
        self.run_id = None

    # ------------- criação de runs --------------

    def start_base_training_run(self, config: Dict[str, Any], tags: Optional[Dict[str, str]] = None) -> str:
        """
        Inicia o run 'pai' de treinamento base.
        Fecha qualquer run pendente antes de abrir um novo (evita 'already active').
        """
        if mlflow.active_run() is not None:
            mlflow.end_run()

        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=f"base-{_ts()}")
        self.run_id = run.info.run_id

        base_tags = {
            "stage": "base_training",
            "experiment": self.experiment_name,
        }
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)

        # Log params (achatados) + config completa como artefato
        flat = _flatten_params(config)
        if flat:
            mlflow.log_params(flat)
        try:
            mlflow.log_dict(config, "config/config.json")
        except Exception:
            pass

        return self.run_id

    def start_finetuning_run(
        self,
        base_run_id: str,
        target_city: str,
        config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Inicia um run de fine-tuning como 'nested' sob o run atual (pai).
        """
        run = mlflow.start_run(run_name=f"ft-{target_city}-{_ts()}", nested=True)
        self.run_id = run.info.run_id

        base_tags = {
            "stage": "fine_tuning",
            "target_city": target_city,
            "base_run_id": base_run_id,
            "experiment": self.experiment_name,
        }
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)

        flat = _flatten_params(config)
        if flat:
            mlflow.log_params(flat)
        try:
            mlflow.log_dict(config, f"config/ft_{target_city}.json")
        except Exception:
            pass

        return self.run_id

    def start_comparison_run(self, checkpoints: Dict[str, str], tags: Optional[Dict[str, str]] = None) -> str:
        """
        Inicia um run de comparação/avaliação como 'nested'.
        """
        run = mlflow.start_run(run_name=f"compare-{_ts()}", nested=True)
        self.run_id = run.info.run_id

        base_tags = {
            "stage": "comparison",
            "experiment": self.experiment_name,
        }
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)

        mlflow.log_dict(checkpoints, "inputs/checkpoints.json")
        return self.run_id

    # ------------- logging utilitário -----------

    def log_training_metrics(self, step: Optional[int] = None, **metrics: float) -> None:
        """Log de métricas de treino/val. Use 'step=epoch' para gráficos por época."""
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)

    def log_evaluation_results(self, **values: float) -> None:
        """Log de métricas de avaliação finais."""
        mlflow.log_metrics(values)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_model_file(self, path: str, artifact_subdir: str = "models") -> None:
        mlflow.log_artifact(path, artifact_path=artifact_subdir)

    def log_env_snapshot(self) -> None:
        """
        Faz um snapshot leve do ambiente: versões, CUDA, pip freeze (se disponível).
        """
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        # torch (opcional)
        try:
            import torch  # type: ignore

            info.update(
                {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
            )
        except Exception:
            pass

        mlflow.log_dict(info, "env/system.json")

        # pip freeze (opcional)
        try:
            req = subprocess.check_output(["pip", "freeze"], text=True)
            # Evite forçar torch+cu aqui; deixe o ambiente/container resolver
            mlflow.log_text(req, "env/requirements_freeze.txt")
        except Exception:
            pass

    # dentro da classe HuMobMLflowTracker
    def log_model_artifact(self, model, file_path: str, artifact_path: str = "models", also_log_mlflow_model: bool = False):
        import os, mlflow
        if not os.path.exists(file_path):
            print(f"[MLflow] ⚠️ Arquivo não encontrado: {file_path} (ignorando)")
            return
        try:
            # garante um run ativo
            if mlflow.active_run() is None:
                if self.run_id:
                    mlflow.start_run(run_id=self.run_id)
                else:
                    run = mlflow.start_run(run_name=f"artifact-{_ts()}", nested=True)
                    self.run_id = run.info.run_id

            mlflow.log_artifact(file_path, artifact_path=artifact_path)

            if also_log_mlflow_model:
                try:
                    import mlflow.pytorch as mpt
                    mpt.log_model(model, artifact_path=f"{artifact_path}/mlflow_model")
                except Exception as e:
                    print(f"[MLflow] (opcional) log_model falhou: {e}")
        except Exception as e:
            print(f"[MLflow] ⚠️ Falha ao logar artefato '{file_path}': {e}")



    # ------------- plots (placeholders seguros) -

    def create_training_plots(self, **_: Any) -> None:
        """Placeholder: implemente quando quiser salvar figuras específicas."""
        return

    def log_model_comparison(self, *_: Any) -> None:
        """Placeholder: implemente quando quiser comparar múltiplos modelos."""
        return

    def create_results_comparison_plot(self, *_: Any) -> None:
        """Placeholder: implemente quando quiser gerar gráficos de comparação."""
        return

def setup_mlflow_for_humob(experiment_name: str, tracking_uri: str = "file:./mlruns"):
    return HuMobMLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)

def get_experiment_summary_for_paper():
    # Placeholder: retorne um dicionário resumindo experimentos se quiser
    return {}