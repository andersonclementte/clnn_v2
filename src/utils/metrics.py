"""Métricas oficiais do HuMob Challenge: GEO-BLEU e DTW."""
import numpy as np
import pandas as pd
from geobleu import calc_geobleu_bulk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_geobleu(df: pd.DataFrame, processes: int = 4) -> float:
    """
    Calcula GEO-BLEU para um DataFrame de pred+GT.
    df deve ter colunas: uid, day, slot, x_pred_cell, y_pred_cell, x_gt_cell, y_gt_cell
    Retorna score médio (0-1, maior = melhor).
    """
    generated = list(zip(df.uid, df.day, df.slot, df.x_pred_cell, df.y_pred_cell))
    reference = list(zip(df.uid, df.day, df.slot, df.x_gt_cell, df.y_gt_cell))
    return calc_geobleu_bulk(generated, reference, processes=processes)


def compute_dtw(df: pd.DataFrame) -> float:
    """
    Calcula DTW médio por usuário para um DataFrame de pred+GT.
    Retorna distância média (menor = melhor).
    """
    scores = []
    for _, user_df in df.groupby("uid"):
        pred = user_df[["x_pred_cell", "y_pred_cell"]].values.astype(float)
        gt   = user_df[["x_gt_cell",   "y_gt_cell"]].values.astype(float)
        dist, _ = fastdtw(pred, gt, dist=euclidean)
        scores.append(dist)
    return float(np.mean(scores))
