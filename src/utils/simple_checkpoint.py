from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    save_dir: str,
    filename: str,
    extra_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Salva um checkpoint simples e portátil (carregável em CPU/GPU).
    Retorna o caminho salvo.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_data:
        payload.update(extra_data)

    path = os.path.join(save_dir, filename)
    torch.save(payload, path)
    return path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    strict: bool = True,
    map_location: str = "cpu",
) -> Tuple[int, Dict[str, Any]]:
    """
    Carrega um checkpoint de forma portátil.
    Retorna (epoch, dict_completo).
    """
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    return epoch, ckpt


def get_latest_checkpoint(dir_path: str = "outputs/models/checkpoints/", pattern: str = "checkpoint_epoch_*.pt") -> Optional[str]:
    """
    Retorna o caminho do checkpoint mais recente (ou None).
    """
    p = Path(dir_path)
    if not p.exists():
        return None
    cks = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cks[0]) if cks else None
