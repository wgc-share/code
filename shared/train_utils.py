from __future__ import annotations

import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedSchemeBManager:
    def __init__(self, dataset, batch_size, shuffle_files=True):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size

        file_to_indices = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            file_to_indices[s["filenames"]].append(i)

        fnames = list(file_to_indices.keys())
        if shuffle_files:
            np.random.shuffle(fnames)

        self.lanes = [[] for _ in range(batch_size)]
        lane_lengths = [0] * batch_size
        for fn in fnames:
            lane_idx = int(np.argmin(lane_lengths))
            self.lanes[lane_idx].append(fn)
            lane_lengths[lane_idx] += len(file_to_indices[fn])

        self.file_to_indices = file_to_indices
        self.pointers = [0] * batch_size
        self.sub_pointers = [0] * batch_size
        self.total_steps = max(lane_lengths) if lane_lengths else 0
        self.current_step = 0

    def get_next_batch(self):
        if self.current_step >= self.total_steps:
            return None, None, None, True

        indices, masks, is_first = [], [], []
        for lane_idx in range(self.batch_size):
            lane_files = self.lanes[lane_idx]
            file_ptr = self.pointers[lane_idx]
            if file_ptr < len(lane_files):
                fname = lane_files[file_ptr]
                f_indices = self.file_to_indices[fname]
                indices.append(f_indices[self.sub_pointers[lane_idx]])
                masks.append(1.0)
                is_first.append(self.sub_pointers[lane_idx] == 0)
                self.sub_pointers[lane_idx] += 1
                if self.sub_pointers[lane_idx] >= len(f_indices):
                    self.pointers[lane_idx] += 1
                    self.sub_pointers[lane_idx] = 0
            else:
                indices.append(0)
                masks.append(0.0)
                is_first.append(False)

        self.current_step += 1
        return indices, masks, is_first, False


class PITDPhysicsLoss(nn.Module):
    def __init__(self, lambda_range: float = 1.0):
        super().__init__()
        self.lambda_range = lambda_range
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.01)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, soc_pred, soc_true, current_raw, Q_as, is_first, lambda_ah, mask):
        m = mask.view(-1, 1, 1)
        l_data_raw = self.mse(soc_pred, soc_true)
        if is_first.any():
            l_init = self.mse(soc_pred[:, 0, :], soc_true[:, 0, :])
            first_mask = is_first.to(dtype=l_init.dtype).view(-1, 1)
            l_data_raw[:, 0, :] += 5.0 * l_init * first_mask
        l_data = (l_data_raw * m).sum() / (m.sum() * soc_pred.size(1) + 1e-9)

        delta_soc_pred = soc_pred[:, 1:, 0] - soc_pred[:, :-1, 0]
        delta_soc_phys = -(current_raw[:, :-1] * 1.0) / Q_as.unsqueeze(1)
        l_ah_raw = self.smooth_l1(delta_soc_pred, delta_soc_phys)
        m_b = mask.view(-1, 1)
        l_ah = (l_ah_raw * m_b).sum() / (m_b.sum() * delta_soc_pred.size(1) + 1e-9)

        l_range_raw = torch.relu(soc_pred - 1.0) + torch.relu(-soc_pred)
        l_range = (l_range_raw * m).sum() / (m.sum() * soc_pred.size(1) + 1e-9)
        total_loss = l_data + lambda_ah * l_ah + self.lambda_range * l_range
        return total_loss, l_data, l_ah, l_range


def _safe_metric_name(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _save_eval_metrics(label: str, avg_mae: float, file_maes: dict, batch_size: int, output_dir, run_id: str | None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_name = _safe_metric_name(label)
    suffix = f"_{run_id}" if run_id else ""

    summary_path = output_path / f"{label_name}_summary{suffix}.csv"
    files_path = output_path / f"{label_name}_file_mae{suffix}.csv"

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "avg_mae", "files", "batch_size", "saved_at"])
        writer.writeheader()
        writer.writerow(
            {
                "label": label,
                "avg_mae": avg_mae,
                "files": len(file_maes),
                "batch_size": batch_size,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    with files_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "mae"])
        writer.writeheader()
        for filename, mae in sorted(file_maes.items()):
            writer.writerow({"filename": filename, "mae": mae})

    print(f"[{label}] metrics saved: {summary_path}")
    print(f"[{label}] file MAE saved: {files_path}")


def evaluate_dataset(model, loader: DataLoader, scaler, config, label: str, output_dir=None, run_id: str | None = None):
    if len(loader.dataset) == 0:
        raise ValueError(f"Cannot evaluate empty dataset: {label}")

    batch_size = loader.batch_size or 1
    manager = BalancedSchemeBManager(loader.dataset, batch_size, shuffle_files=False)
    model.eval()
    file_errors = defaultdict(list)
    h_state = torch.zeros(1, batch_size, config["D_MODEL"], device=config["DEVICE"])

    with torch.no_grad():
        pbar = tqdm(
            total=manager.total_steps,
            desc=f"{label} eval",
            leave=False,
            colour="green",
            disable=config.get("DISABLE_TQDM", False),
        )
        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [loader.dataset[i] for i in indices]
            x = torch.stack([sample["x_dyn"] for sample in samples]).to(config["DEVICE"])
            t = torch.stack([sample["t_mean"] for sample in samples]).to(config["DEVICE"])
            y = torch.stack([sample["soc"] for sample in samples]).to(config["DEVICE"])
            active = torch.tensor(masks, dtype=torch.bool, device=config["DEVICE"])
            first = torch.tensor(is_first, dtype=torch.bool, device=config["DEVICE"])

            h_state = h_state.detach()
            if first.any():
                h_state[:, first, :] = 0.0
            x_n, t_n = scaler.transform(x, t)
            y_p, h_state = model(x_n, t_n, h_state)
            sample_maes = torch.abs(y_p - y).mean(dim=(1, 2))
            for lane_idx, sample in enumerate(samples):
                if active[lane_idx]:
                    file_errors[sample["filenames"]].append(sample_maes[lane_idx].item())
            pbar.update(1)
        pbar.close()

    file_maes = {fn: float(np.mean(errs)) for fn, errs in file_errors.items()}
    avg_mae = float(np.mean(list(file_maes.values()))) if file_maes else float("nan")
    worst = sorted(file_maes.items(), key=lambda kv: kv[1], reverse=True)[:3]
    # print(f"[{label}] files={len(file_maes)} batch={batch_size} avg_mae={avg_mae:.4f}")
    # if worst:
    #     print(f"[{label}] worst files: " + ", ".join(f"{fn}:{mae:.4f}" for fn, mae in worst))
    if output_dir is not None:
        _save_eval_metrics(label, avg_mae, file_maes, batch_size, output_dir, run_id)
    return avg_mae, file_maes
