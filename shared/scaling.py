from __future__ import annotations

import torch


class PITDScaler:
    """Feature standardizer fitted only on the training dataset."""

    def __init__(self):
        self.stats = {}

    def fit(self, dataset):
        if len(dataset) == 0:
            raise ValueError("Cannot fit PITDScaler on an empty dataset")

        x_sum = torch.zeros(5, dtype=torch.float64)
        x_square_sum = torch.zeros(5, dtype=torch.float64)
        x_count = 0
        t_sum = torch.tensor(0.0, dtype=torch.float64)
        t_square_sum = torch.tensor(0.0, dtype=torch.float64)
        t_count = 0

        for sample in dataset.samples:
            x = sample["x_dyn"].to(dtype=torch.float64)
            t = sample["t_mean"].to(dtype=torch.float64).reshape(-1)
            x_sum += x.sum(dim=0)
            x_square_sum += x.square().sum(dim=0)
            x_count += x.size(0)
            t_sum += t.sum()
            t_square_sum += t.square().sum()
            t_count += t.numel()

        if x_count == 0 or t_count == 0:
            raise ValueError("Training dataset contains no usable scaler values")

        x_mean = x_sum / x_count
        x_var = torch.clamp(x_square_sum / x_count - x_mean.square(), min=0.0)
        t_mean = t_sum / t_count
        t_var = torch.clamp(t_square_sum / t_count - t_mean.square(), min=0.0)
        self.stats = {
            "x_mean": x_mean.to(dtype=torch.float32),
            "x_std": (torch.sqrt(x_var) + 1e-6).to(dtype=torch.float32),
            "t_mean": t_mean.to(dtype=torch.float32),
            "t_std": (torch.sqrt(t_var) + 1e-6).to(dtype=torch.float32),
        }

    def transform(self, x_dyn, t_mean):
        required = {"x_mean", "x_std", "t_mean", "t_std"}
        missing = required.difference(self.stats)
        if missing:
            raise RuntimeError(f"PITDScaler is not fitted; missing statistics: {sorted(missing)}")

        x_mean = self.stats["x_mean"].to(device=x_dyn.device, dtype=x_dyn.dtype)
        x_std = self.stats["x_std"].to(device=x_dyn.device, dtype=x_dyn.dtype)
        temp_mean = self.stats["t_mean"].to(device=t_mean.device, dtype=t_mean.dtype)
        temp_std = self.stats["t_std"].to(device=t_mean.device, dtype=t_mean.dtype)
        return (x_dyn - x_mean) / x_std, (t_mean - temp_mean) / temp_std
