import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordLSTM(nn.Module):
    """
    BiLSTM sobre sequências de coordenadas (x, y) com attention pooling.

    Em vez de usar apenas o último hidden state, aplica atenção aprendível
    sobre todos os outputs do LSTM, permitindo ao modelo focar nos momentos
    mais relevantes da trajetória (ex: padrões do dia anterior, horários de pico).

    Args:
        input_size (int): features de entrada (2 para x, y).
        hidden_size (int): dimensão do hidden state do LSTM.
        num_layers (int): camadas LSTM empilhadas.
        bidirectional (bool): se True, usa BiLSTM (output_dim = hidden_size * 2).
        dropout (float): dropout entre camadas (só ativo se num_layers > 1).
    """
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 20,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Camada de atenção: projeta cada hidden state para um score escalar
        self.attention = nn.Linear(self.output_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2) — sequência de coordenadas normalizadas.

        Returns:
            (batch, hidden_size * num_directions) — vetor de contexto ponderado
            por atenção sobre toda a sequência.
        """
        # output: (batch, seq_len, hidden_size * num_directions)
        output, _ = self.lstm(x)

        # Scores de atenção: (batch, seq_len, 1) → (batch, seq_len)
        scores = self.attention(output).squeeze(-1)
        weights = F.softmax(scores, dim=1)          # (batch, seq_len)

        # Vetor de contexto: média ponderada dos hidden states
        context = (weights.unsqueeze(-1) * output).sum(dim=1)  # (batch, output_dim)

        return context
