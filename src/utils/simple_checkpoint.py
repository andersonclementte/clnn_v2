# src/utils/simple_checkpoint.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import os, time
import torch

def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _sanitize_extra(extra_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not extra_data:
        return {}
    safe: Dict[str, Any] = {}
    np = None
    try:
        import numpy as _np
        np = _np
    except Exception:
        pass

    for k, v in extra_data.items():
        # tensores sempre em CPU
        if isinstance(v, torch.Tensor):
            safe[k] = v.detach().cpu()
            continue
        # numpy -> tensor
        if np is not None and isinstance(v, np.ndarray):
            safe[k] = torch.from_numpy(v).cpu()
            continue
        # escalares primitivos e None
        if isinstance(v, (int, float, bool, str)) or v is None:
            safe[k] = v
            continue
        # listas/tuplas/dicts simples -> mantém se parecerem primitivos
        if isinstance(v, (list, tuple)):
            safe[k] = v  # assume primitivos
            continue
        if isinstance(v, dict):
            safe[k] = {kk: vv if isinstance(vv, (int, float, bool, str, type(None))) else str(vv)
                       for kk, vv in v.items()}
            continue
        # qualquer outra coisa: stringify
        safe[k] = str(v)
    return safe

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: Optional[float] = None,
    save_dir: str = "outputs/models/checkpoints",
    filename: Optional[str] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
    path = str(Path(save_dir) / filename)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
        "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
        "extra": _sanitize_extra(extra_data),
        "torch_version": torch.__version__,
        "saved_at": time.time(),
    }
    _atomic_save(payload, path)

    # also write/update symlink/copy for "last.pt"
    last = Path(save_dir) / "last.pt"
    try:
        if last.is_symlink() or last.exists():
            last.unlink()
        last.symlink_to(Path(filename).name)
    except Exception:
        # se symlink não for possível, grava uma cópia leve
        torch.save(payload, str(last))
    return path

def load_training_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Retoma treino (model+optimizer+scheduler+scaler).
    Usa weights_only=False pois precisamos de objetos não-tensor nos states.
    """
    try:
        ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch, ckpt

def load_model_weights_for_inference(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Para avaliação/inferência segura (não carrega optimizer etc).
    Pode usar weights_only=True desde que o payload só tenha tensores/escalares.
    """
    try:
        ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)  # fallback se salvar só o state_dict
    model.load_state_dict(state, strict=strict)
    return ckpt

def get_latest_checkpoint(dir_path: str, pattern: str = "checkpoint_epoch_*.pt") -> Optional[str]:
    p = Path(dir_path)
    if not p.exists():
        return None
    cks = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cks[0]) if cks else None

def cleanup_old_checkpoints(dir_path: str, keep_last: int = 3, pattern: str = "checkpoint_epoch_*.pt") -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        return []
    cks = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    to_remove = cks[keep_last:]
    removed = []
    for f in to_remove:
        try:
            f.unlink()
            removed.append(str(f))
        except Exception:
            pass
    return removed
