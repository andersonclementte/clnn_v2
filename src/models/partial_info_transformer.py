import torch
import torch.nn as nn


class CoordTransformer(nn.Module):
    """
    Transformer Encoder sobre sequências de coordenadas (x, y) com mean pooling.

    Substitui o CoordLSTM da Fase 3 eliminando o BPTT estrutural: a self-attention
    processa todos os timesteps em paralelo, sem gradiente recursivo. Isso
    remove a causa principal das explosões de gradiente observadas na v2 BiLSTM.

    Args:
        input_size (int): features de entrada (2 para x, y).
        d_model (int): dimensão do modelo. Deve ser igual a fusion_dim do
            WeightedFusion para preservar a invariante da arquitetura.
        nhead (int): número de attention heads. d_model deve ser divisível por nhead.
        num_layers (int): camadas empilhadas do encoder.
        dim_feedforward (int): dimensão do MLP interno de cada camada.
        max_len (int): seq_len máxima suportada (para o positional encoding).
        dropout (float): dropout dentro do encoder.
    """
    def __init__(
        self,
        input_size: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_len: int = 720,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) deve ser divisível por nhead ({nhead})"

        self.d_model = d_model
        self.output_dim = d_model  # mesma interface de CoordLSTM
        self.max_len = max_len

        # Projeção de entrada 2-D → d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding aprendível
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Stack de TransformerEncoderLayer com pre-norm (mais estável)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2) — sequência de coordenadas normalizadas.

        Returns:
            (batch, d_model) — mean pooling sobre todos os outputs encoded.
        """
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(
                f"seq_len={T} excede max_len={self.max_len} do positional encoding"
            )

        # Projeção + positional encoding
        h = self.input_proj(x) + self.pos_embedding[:, :T]
        # Encoder: (B, T, d_model)
        h = self.encoder(h)
        # Mean pooling: (B, d_model)
        return h.mean(dim=1)
