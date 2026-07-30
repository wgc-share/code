import torch
import torch.nn as nn


class BatteryTDGCMModel(nn.Module):
    """
    PITD-Net causal variant.
    Keeps the same architecture as the original model, but adds a causal
    attention mask inside the TransformerEncoder so each position can only
    attend to past and current tokens.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super(BatteryTDGCMModel, self).__init__()

        self.embedding = nn.Linear(5, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
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
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x_dyn, t_mean, h_prev=None):
        """
        x_dyn:  [Batch, Seq_Len, 5]
        t_mean: [Batch, 1]
        h_prev: [1, Batch, d_model]
        """
        x_emb = self.embedding(x_dyn)
        seq_len = x_emb.size(1)
        causal_mask = self._causal_mask(seq_len, x_emb.device)
        h_seq = self.transformer(x_emb, mask=causal_mask)

        temp_params = self.temp_net(t_mean)
        gamma = temp_params[:, : x_emb.size(-1)].unsqueeze(1)
        beta = temp_params[:, x_emb.size(-1) :].unsqueeze(1)
        h_modulated = gamma * h_seq + beta

        if h_prev is None:
            h_prev = torch.zeros(
                1,
                x_dyn.size(0),
                x_emb.size(-1),
                device=x_dyn.device,
                dtype=x_emb.dtype,
            )

        h_fusion, h_current = self.gru(h_modulated, h_prev)
        soc_pred = self.mlp(h_fusion)
        return soc_pred, h_current
