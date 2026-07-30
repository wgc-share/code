import torch
import torch.nn as nn


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.temp_mod = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, d_model * 6),
        )

        nn.init.zeros_(self.temp_mod[-1].weight)
        nn.init.zeros_(self.temp_mod[-1].bias)

    def forward(self, x, t_mean, attn_mask=None):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.temp_mod(t_mean).chunk(6, dim=-1)

        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        h1 = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attn(h1, h1, h1, attn_mask=attn_mask, need_weights=False)
        x = x + gate1 * attn_out

        h2 = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.ff(h2)
        return x


class BatteryTDGCMModel(nn.Module):
    """
    Temperature-conditioned PITD-Net variant with AdaLN blocks.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1, use_causal=False):
        super().__init__()
        self.d_model = d_model
        self.use_causal = use_causal

        self.embedding = nn.Linear(5, d_model)
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
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x_dyn, t_mean, h_prev=None):
        x = self.embedding(x_dyn)
        attn_mask = self._causal_mask(x.size(1), x.device) if self.use_causal else None

        for block in self.blocks:
            x = block(x, t_mean, attn_mask=attn_mask)

        if h_prev is None:
            h_prev = torch.zeros(1, x_dyn.size(0), self.d_model, device=x_dyn.device, dtype=x.dtype)

        h_fusion, h_current = self.gru(x, h_prev)
        soc_pred = self.mlp(h_fusion)
        return soc_pred, h_current
