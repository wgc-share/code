import torch
import torch.nn as nn


class BatteryTDGCMModel(nn.Module):
    """Non-causal Transformer with FiLM temperature modulation and a GRU."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
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

    def forward(self, x_dyn, t_mean, h_prev=None):
        x_emb = self.embedding(x_dyn)
        h_seq = self.transformer(x_emb)

        gamma, beta = self.temp_net(t_mean).chunk(2, dim=-1)
        h_modulated = gamma.unsqueeze(1) * h_seq + beta.unsqueeze(1)

        if h_prev is None:
            h_prev = torch.zeros(
                1,
                x_dyn.size(0),
                self.d_model,
                device=x_dyn.device,
                dtype=x_emb.dtype,
            )

        h_fusion, h_current = self.gru(h_modulated, h_prev)
        soc_pred = self.mlp(h_fusion)
        return soc_pred, h_current
