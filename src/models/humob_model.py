import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.external_info import ExternalInformationFusionNormalized, ExternalInformationDense
from src.models.partial_info import CoordLSTM


class WeightedFusion(nn.Module):
    """
    Funde dois vetores de mesmo tamanho por uma soma ponderada aprendível.
    """
    def __init__(self, dim: int = 20, init_w_r: float = 0.5, init_w_e: float = 0.5):
        super().__init__()
        self.w_r = nn.Parameter(torch.tensor(init_w_r, dtype=torch.float32))
        self.w_e = nn.Parameter(torch.tensor(init_w_e, dtype=torch.float32))
        self.dim = dim

    def forward(self, static_red: torch.Tensor, dyn_emb: torch.Tensor) -> torch.Tensor:
        assert static_red.shape == dyn_emb.shape and static_red.size(1) == self.dim
        fused = self.w_r * static_red + self.w_e * dyn_emb # W_r * static + W_e * dynamic
        return fused


class MLPSmall(nn.Module):
    """MLP menor e mais estável"""
    def __init__(self, in_dim: int, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, n_clusters)
        
        # Inicialização Xavier (mais estável)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DestinationHead(nn.Module):
    """Combina MLP + softmax + weighted sum pelos cluster centers."""
    def __init__(self, in_dim: int, hidden_dim: int, cluster_centers: torch.Tensor):
        super().__init__()
        C, coord_dim = cluster_centers.shape
        assert coord_dim == 2
        self.mlp = MLPSmall(in_dim, hidden_dim, C)
        self.register_buffer("centers", cluster_centers)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(fused)
        P = F.softmax(logits, dim=1)
        coords = P @ self.centers
        return coords


def discretize_coordinates(coords_pred: torch.Tensor, grid_size: int = 200):
    """Converte coordenadas contínuas [0,1] para grid discreto [0, grid_size-1]."""
    # Desnormaliza: [0,1] -> [0, grid_size-1]
    coords_discrete = coords_pred * (grid_size - 1)
    coords_discrete = torch.round(coords_discrete).long()
    coords_discrete = torch.clamp(coords_discrete, 0, grid_size - 1)
    return coords_discrete


class HuMobModel(nn.Module):
    """
    Modelo completo que combina todas as partes e faz rollout para múltiplos passos.
    """
    def __init__(
        self,
        n_users: int,
        n_cities: int,
        cluster_centers: torch.Tensor,
        user_emb_dim: int = 4,
        city_emb_dim: int = 4, 
        temporal_dim: int = 8,
        poi_out_dim: int = 4,
        lstm_hidden: int = 4,
        fusion_dim: int = 8,
        sequence_length: int = 48,
        prediction_steps: int = 1,
        disable_poi: bool = False
    ):
        super().__init__()
        
        # Salva configurações
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        
        # Componentes - CORRIGIDO: usando ExternalInformationFusionNormalized
        self.fusion = ExternalInformationFusionNormalized(
            n_users=n_users,
            n_cities=n_cities,
            user_emb_dim=user_emb_dim,
            city_emb_dim=city_emb_dim,
            temporal_dim=temporal_dim,
            poi_out_dim=poi_out_dim,
            disable_poi=disable_poi
        )
        self.dense = ExternalInformationDense(
            in_dim=self.fusion.out_dim, 
            out_dim=fusion_dim
        )
        self.lstm = CoordLSTM(
            input_size=2, 
            hidden_size=lstm_hidden, 
            bidirectional=True
        )
        self.weighted_fusion = WeightedFusion(dim=fusion_dim)
        self.destination_head = DestinationHead(
            in_dim=fusion_dim,
            hidden_dim=500,
            cluster_centers=cluster_centers
        )
        self._stable_init()
        
    def forward_single_step(self, uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq):
        """Faz uma predição para um único passo."""
        # Informação estática (contexto) - CORRIGIDO: parâmetros normalizados
        static_emb = self.fusion(uid, d_norm, t_sin, t_cos, city, poi_norm)
        static_red = self.dense(static_emb)
        
        # Informação dinâmica (padrão de movimento)
        dyn_emb = self.lstm(coords_seq)
        
        # Fusão inteligente
        fused = self.weighted_fusion(static_red, dyn_emb)
        
        # Predição final
        pred_coords = self.destination_head(fused)
        
        return pred_coords
    
    def _stable_init(self):
        """Inicialização mais estável"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        self.apply(init_weights)
    
    def rollout_predictions(
        self, 
        uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, 
        n_steps: int,
        use_predictions: bool = True
    ):
        """
        Faz predições para múltiplos passos futuros.
        CORRIGIDO: agora funciona com dados normalizados e ciclos temporais corretos.
        """
        predictions = []
        current_seq = coords_seq.clone()
        current_t_sin, current_t_cos = t_sin.clone(), t_cos.clone()
        current_d_norm = d_norm.clone()
        
        for step in range(n_steps):
            # Prediz próximo passo
            pred = self.forward_single_step(
                uid, current_d_norm, current_t_sin, current_t_cos, 
                city, poi_norm, current_seq
            )
            predictions.append(pred)
            
            if use_predictions and step < n_steps - 1:
                # Atualiza sequência: remove primeiro ponto, adiciona predição
                new_point = pred.unsqueeze(1)  # (batch, 1, 2)
                current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)
            
            # CORRIGIDO: incrementa tempo com codificação circular
            # Avança um timeslot (1/48 do dia)
            current_t_raw = torch.atan2(current_t_sin, current_t_cos) / (2 * np.pi) * 48
            current_t_raw = (current_t_raw + 1) % 48
            
            # Atualiza sin/cos
            current_t_sin = torch.sin(2 * np.pi * current_t_raw / 48)
            current_t_cos = torch.cos(2 * np.pi * current_t_raw / 48)
            
            # Se completou um dia (48 slots), incrementa dia
            mask_new_day = (current_t_raw < 1).float()  # Voltou para ~0
            current_d_norm = current_d_norm + mask_new_day / 74.0  # Incrementa d_norm
            
        return torch.stack(predictions, dim=1)  # (batch, n_steps, 2)
    
    def forward(self, uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, n_steps=1):
        """Forward principal - pode ser usado tanto para treino quanto inferência."""
        if n_steps == 1:
            return self.forward_single_step(uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq)
        else:
            return self.rollout_predictions(
                uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, n_steps
            )