from __future__ import annotations

import torch
import torch.nn as nn

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel


class TempInputCausalTransformerModel(nn.Module):
    """Causal Transformer without AdaLN; normalized temperature is appended to each token."""

    def __init__(self, d_model=96, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(6, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x_dyn, t_mean, h_prev=None):
        t_seq = t_mean.unsqueeze(1).expand(-1, x_dyn.size(1), -1)
        x = self.embedding(torch.cat([x_dyn, t_seq], dim=-1))
        x = self.transformer(x, mask=self._causal_mask(x.size(1), x.device))

        if h_prev is None:
            h_prev = torch.zeros(1, x_dyn.size(0), self.d_model, device=x_dyn.device, dtype=x.dtype)
        h_fusion, h_current = self.gru(x, h_prev)
        return self.mlp(h_fusion), h_current


class FiLMCausalTransformerModel(nn.Module):
    """Causal Transformer with post-encoder FiLM temperature modulation."""

    def __init__(self, d_model=96, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temp_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, d_model * 2),
        )
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x_dyn, t_mean, h_prev=None):
        x = self.embedding(x_dyn)
        h_seq = self.transformer(x, mask=self._causal_mask(x.size(1), x.device))
        gamma, beta = self.temp_net(t_mean).chunk(2, dim=-1)
        h_modulated = gamma.unsqueeze(1) * h_seq + beta.unsqueeze(1)

        if h_prev is None:
            h_prev = torch.zeros(1, x_dyn.size(0), self.d_model, device=x_dyn.device, dtype=x.dtype)
        h_fusion, h_current = self.gru(h_modulated, h_prev)
        return self.mlp(h_fusion), h_current


class TempInputAdaLNCausalTransformerModel(nn.Module):
    """
    Causal Transformer with both explicit temperature input and AdaLN modulation.

    The temperature is appended to every token as an input feature, while the
    same temperature also conditions the AdaLN residual modulation.
    """

    def __init__(self, d_model=96, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x_dyn, t_mean, h_prev=None):
        t_seq = t_mean.unsqueeze(1).expand(-1, x_dyn.size(1), -1)
        x = self.embedding(torch.cat([x_dyn, t_seq], dim=-1))
        attn_mask = self._causal_mask(x.size(1), x.device)

        for block in self.blocks:
            x = block(x, t_mean, attn_mask=attn_mask)

        if h_prev is None:
            h_prev = torch.zeros(1, x_dyn.size(0), self.d_model, device=x_dyn.device, dtype=x.dtype)
        h_fusion, h_current = self.gru(x, h_prev)
        return self.mlp(h_fusion), h_current


class AdaLNNoStateTransferModel(AdaLNBatteryTDGCMModel):
    """AdaLN model that resets GRU hidden state for every window."""

    def forward(self, x_dyn, t_mean, h_prev=None):
        return super().forward(x_dyn, t_mean, h_prev=None)
