from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CODE_ROOT = BASE_DIR.parent
SHARED_DIR = CODE_ROOT / "shared"

shared_path = str(SHARED_DIR)
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from eval_soc_start_causal_adaln_dropout import filter_from_soc_start
from project_paths import baseline_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, parse_segment_filename, split_soccc_by_cells
from soc_interval_eval import evaluate_soc_intervals, parse_soc_intervals, prompt_soc_intervals, write_soc_interval_metrics
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed
from torch_io import load_torch_file, save_torch_file
from tune_preset_soc_start_causal_adaln_dropout import aggregate_segment_rows


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_SOC_START_AUG_FILTERED_TEMPS"
TEMP_TARGETS = [10.0, 25.0, 40.0]
TEMP_TOLERANCE = 2.0
SOC_STARTS = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
AUG_DIVISIONS = [8, 9, 10, 11, 12]


def temp_cache_suffix(temps: list[float], tolerance: float) -> str:
    labels = []
    for temp in temps:
        labels.append(str(int(temp)) if float(temp).is_integer() else str(temp).replace(".", "p"))
    tol_label = str(int(tolerance)) if float(tolerance).is_integer() else str(tolerance).replace(".", "p")
    return "T" + "_".join(labels) + f"_pm{tol_label}"


def filter_files_by_temperature(files: list[str], temps: list[float], tolerance: float) -> list[str]:
    targets = [float(temp) for temp in temps]
    kept = []
    for filename in files:
        temp_c = float(parse_segment_filename(filename).get("temp_C", float("nan")))
        if any(abs(temp_c - target) <= tolerance for target in targets):
            kept.append(filename)
    return kept


class SOCStartAugmentedDataset:
    def __init__(self, base_dataset: BatteryTDGCMDataset, divisions: list[int], seed: int):
        self.base_dataset = base_dataset
        self.samples, self.assignment_rows = self._build_samples(base_dataset, divisions, seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def _build_samples(base_dataset: BatteryTDGCMDataset, divisions: list[int], seed: int):
        rng = np.random.default_rng(seed)
        file_to_samples = defaultdict(list)
        for sample in base_dataset.samples:
            file_to_samples[sample["filenames"]].append(sample)

        augmented = []
        assignment_rows = []
        for filename in sorted(file_to_samples):
            samples = file_to_samples[filename]
            n_div = int(rng.choice(divisions))
            starts = [100.0 - idx * (100.0 / n_div) for idx in range(n_div)]
            for start_soc in starts:
                threshold = start_soc / 100.0
                start_idx = None
                for idx, sample in enumerate(samples):
                    window_start_soc = float(sample["soc"][0, 0])
                    if window_start_soc <= threshold:
                        start_idx = idx
                        break
                if start_idx is None:
                    continue

                subsegment_name = f"{filename}__socstart_{start_soc:.4f}".replace(".", "p")
                for sample in samples[start_idx:]:
                    copied = dict(sample)
                    copied["filenames"] = subsegment_name
                    copied["source_filename"] = filename
                    copied["soc_start_percent"] = float(start_soc)
                    copied["aug_divisions"] = n_div
                    augmented.append(copied)
                assignment_rows.append(
                    {
                        "source_filename": filename,
                        "aug_divisions": n_div,
                        "soc_start_percent": float(start_soc),
                        "start_window_index": start_idx,
                        "windows": len(samples) - start_idx,
                    }
                )
        return augmented, assignment_rows

    @classmethod
    def from_base(cls, base_dataset: BatteryTDGCMDataset, divisions: list[int], seed: int):
        obj = cls.__new__(cls)
        obj.base_dataset = base_dataset
        obj.samples, obj.assignment_rows = cls._build_samples(base_dataset, divisions, seed)
        return obj


def _make_results_dirs():
    base = baseline_results_dir() / "soc_start_aug_filtered_temps"
    cache_dir = base / "pt"
    pth_dir = base / "pth_save"
    csv_dir = base / "csv_save"
    meta_dir = base / "metadata"
    for d in (cache_dir, pth_dir, csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def latest_checkpoint(pth_dir: Path):
    ckpts = list(pth_dir.glob(f"best_model_{PROJECT}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {pth_dir}. Run training first.")
    return max(ckpts, key=lambda path: path.stat().st_mtime)


def make_config():
    cache_dir, pth_dir, csv_dir, meta_dir = _make_results_dirs()
    return {
        "PROJECT": PROJECT,
        "DATA_DIR": str(processed_segments_dir()),
        "SPLIT_FILE": str(split_file_path()),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "BATCH_SIZE": 256,
        "VAL_BATCH_SIZE": 64,
        "SOC_BATCH_SIZE": 64,
        "LR": 2e-4,
        "EPOCHS": 50,
        "D_MODEL": 64,
        "NHEAD": 4,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10.0,
        "LAMBDA_AH_STEP": 20.0,
        "GRAD_CLIP": 1.0,
        "P_RESET": math.nan,
        "AUG_DIVISIONS": AUG_DIVISIONS,
        "AUG_SEED": 42,
        "TEMPS": TEMP_TARGETS,
        "TEMP_TOLERANCE": TEMP_TOLERANCE,
        "SOC_STARTS": SOC_STARTS,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
    }


def print_config(config: dict):
    print("=== SchemeB causal + AdaLN + SOC-start augmented run ===")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR  : {config['CACHE_DIR']}")
    print(f"PTH_DIR    : {config['PTH_DIR']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"BATCH_SIZE : {config['BATCH_SIZE']}")
    print(f"VAL_BATCH  : {config['VAL_BATCH_SIZE']} (deterministic stateful lanes)")
    print("P_RESET    : disabled")
    print(f"AUG_DIVS   : {config['AUG_DIVISIONS']}")
    print(f"TEMPS      : {config['TEMPS']}")
    print(f"TEMP_TOL   : +/-{config['TEMP_TOLERANCE']} C")
    print(f"WINDOW     : {config['WINDOW_SIZE']}")
    print(f"STRIDE     : {config['STRIDE']}")
    print("TRANSFORMER: causal mask enabled")
    print("TEMP MOD   : AdaLN")


def load_all_datasets(config: dict):
    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    raw_counts = {
        "train": len(train_f),
        "val": len(val_f),
        "test_random": len(test_random_f),
        "test_fixed": len(test_fixed_f),
    }
    train_f = filter_files_by_temperature(train_f, config["TEMPS"], config["TEMP_TOLERANCE"])
    val_f = filter_files_by_temperature(val_f, config["TEMPS"], config["TEMP_TOLERANCE"])
    test_random_f = filter_files_by_temperature(test_random_f, config["TEMPS"], config["TEMP_TOLERANCE"])
    test_fixed_f = filter_files_by_temperature(test_fixed_f, config["TEMPS"], config["TEMP_TOLERANCE"])
    print(
        f"Temperature filter: targets={config['TEMPS']} C, tolerance=+/-{config['TEMP_TOLERANCE']} C | "
        f"train={raw_counts['train']}->{len(train_f)} | "
        f"val={raw_counts['val']}->{len(val_f)} | "
        f"test_random={raw_counts['test_random']}->{len(test_random_f)} | "
        f"test_fixed={raw_counts['test_fixed']}->{len(test_fixed_f)}"
    )
    split_lists = {
        "train": train_f,
        "val": val_f,
        "test_random": test_random_f,
        "test_fixed": test_fixed_f,
    }
    empty_splits = [name for name, files in split_lists.items() if not files]
    if empty_splits:
        raise RuntimeError(f"Empty split(s) after temperature filtering: {', '.join(empty_splits)}")
    suffix = temp_cache_suffix(config["TEMPS"], config["TEMP_TOLERANCE"])

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], train_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"train_cache_causal_adaln_soc_start_aug_{suffix}.pt"),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], val_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"val_cache_causal_adaln_soc_start_aug_{suffix}.pt"),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], test_random_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"test_random_cache_causal_adaln_soc_start_aug_{suffix}.pt"),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], test_fixed_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"test_fixed_cache_causal_adaln_soc_start_aug_{suffix}.pt"),
    )
    return train_ds, val_ds, test_random_ds, test_fixed_ds


def build_model(config: dict):
    return AdaLNBatteryTDGCMModel(
        d_model=config["D_MODEL"],
        nhead=config["NHEAD"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        use_causal=True,
    ).to(config["DEVICE"])


def write_test_segment_metrics(rows: list[dict], csv_dir: str, run_id: str):
    df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ("split_order", "metric_order", "segment_order")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    out_path = Path(csv_dir) / f"test_segment_metrics_{run_id}.csv"
    total_path = Path(csv_dir) / f"test_total_summary_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    df[df["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"Test segment metrics saved: {out_path}")
    print(f"Test total summary saved: {total_path}")
    return df


def evaluate_test_splits(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str):
    _, _, test_random_ds, test_fixed_ds = datasets
    rows = []
    for split_order, (split_name, dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        loader = DataLoader(dataset, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
        label = "Test random" if split_name == "test_random" else "Test fixed"
        avg_mae, file_maes = evaluate_dataset(
            model,
            loader,
            scaler,
            {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]},
            label=label,
            output_dir=config["CSV_DIR"],
            run_id=run_id,
        )
        rows.append(
            {
                "trial_id": run_id,
                "trial_index": 1,
                "p_reset": config["P_RESET"],
                "split_order": split_order,
                "split": split_name,
                "soc_start_order": 0,
                "soc_start_percent": math.nan,
                "metric_order": 1,
                "metric_level": "total",
                "segment_order": 0,
                "segment_id": "total",
                "condition": "total",
                "cell_id": "",
                "segment_index": "",
                "avg_mae": avg_mae,
                "files": len(file_maes),
                "windows": len(dataset),
                "batch_size": config["VAL_BATCH_SIZE"],
            }
        )
        segment_rows = aggregate_segment_rows(
            {
                **config,
                "TRIAL_ID": run_id,
                "TRIAL_INDEX": 1,
                "SOC_BATCH_SIZE": config["VAL_BATCH_SIZE"],
            },
            split_name,
            split_order,
            math.nan,
            0,
            file_maes,
            dataset,
        )
        rows.extend(segment_rows)

    metrics_df = write_test_segment_metrics(rows, config["CSV_DIR"], run_id)
    print("\n=== Test total metrics ===")
    print(metrics_df[metrics_df["metric_level"].eq("total")][["split", "avg_mae", "files", "windows"]].to_string(index=False))
    print("\n=== Test segment metrics ===")
    print(
        metrics_df[metrics_df["metric_level"].eq("segment")][
            ["split", "segment_id", "condition", "avg_mae", "files", "windows"]
        ].to_string(index=False)
    )
    return metrics_df


def write_soc_start_metrics(rows: list[dict], csv_dir: str, run_id: str):
    df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ("split_order", "soc_start_order", "metric_order", "segment_order")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    out_path = Path(csv_dir) / f"soc_start_metrics_{run_id}.csv"
    total_path = Path(csv_dir) / f"soc_start_total_summary_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    df[df["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"SOC-start detailed metrics saved: {out_path}")
    print(f"SOC-start total metrics saved: {total_path}")
    return df


def evaluate_soc_start_splits(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str):
    _, _, test_random_ds, test_fixed_ds = datasets
    rows = []
    for split_order, (split_name, base_dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        for soc_order, soc_start in enumerate(config["SOC_STARTS"], start=1):
            filtered_ds = filter_from_soc_start(base_dataset, soc_start)
            if len(filtered_ds) == 0:
                rows.append(
                    {
                        "trial_id": run_id,
                        "trial_index": 1,
                        "p_reset": config["P_RESET"],
                        "split_order": split_order,
                        "split": split_name,
                        "soc_start_order": soc_order,
                        "soc_start_percent": soc_start,
                        "metric_order": 1,
                        "metric_level": "total",
                        "segment_order": 0,
                        "segment_id": "total",
                        "condition": "total",
                        "cell_id": "",
                        "segment_index": "",
                        "avg_mae": float("nan"),
                        "files": 0,
                        "windows": 0,
                        "batch_size": config["SOC_BATCH_SIZE"],
                    }
                )
                continue

            loader = DataLoader(filtered_ds, batch_size=config["SOC_BATCH_SIZE"], shuffle=False)
            avg_mae, file_maes = evaluate_dataset(
                model,
                loader,
                scaler,
                {
                    "DEVICE": config["DEVICE"],
                    "D_MODEL": config["D_MODEL"],
                    "DISABLE_TQDM": True,
                },
                label=f"{run_id} {split_name} SOC<= {soc_start:g}%",
            )
            rows.append(
                {
                    "trial_id": run_id,
                    "trial_index": 1,
                    "p_reset": config["P_RESET"],
                    "split_order": split_order,
                    "split": split_name,
                    "soc_start_order": soc_order,
                    "soc_start_percent": soc_start,
                    "metric_order": 1,
                    "metric_level": "total",
                    "segment_order": 0,
                    "segment_id": "total",
                    "condition": "total",
                    "cell_id": "",
                    "segment_index": "",
                    "avg_mae": avg_mae,
                    "files": len(file_maes),
                    "windows": len(filtered_ds),
                    "batch_size": config["SOC_BATCH_SIZE"],
                }
            )
            segment_rows = aggregate_segment_rows(
                {
                    **config,
                    "TRIAL_ID": run_id,
                    "TRIAL_INDEX": 1,
                },
                split_name,
                split_order,
                soc_start,
                soc_order,
                file_maes,
                filtered_ds,
            )
            rows.extend(segment_rows)

    metrics_df = write_soc_start_metrics(rows, config["CSV_DIR"], run_id)
    total_df = metrics_df[metrics_df["metric_level"].eq("total")]
    print("\n=== SOC-start total metrics ===")
    print(total_df[["split", "soc_start_percent", "avg_mae", "files", "windows"]].to_string(index=False))
    return metrics_df


def train_causal_adaln_dropout():
    set_seed(42)
    config = make_config()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print_config(config)
    datasets = load_all_datasets(config)
    train_base_ds, val_ds, _, _ = datasets
    train_ds = SOCStartAugmentedDataset.from_base(
        train_base_ds,
        config["AUG_DIVISIONS"],
        config["AUG_SEED"],
    )
    pd.DataFrame(train_ds.assignment_rows).to_csv(
        os.path.join(config["META_DIR"], f"soc_start_aug_assignment_{run_id}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Training augmentation: base_windows={len(train_base_ds)} -> "
        f"augmented_windows={len(train_ds)} | subsegments={len(train_ds.assignment_rows)}"
    )
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)

    scaler = PITDScaler()
    scaler.fit(train_base_ds)

    model = build_model(config)

    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-5)
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])
    best_overall_mae = float("inf")
    best_model_path = os.path.join(config["PTH_DIR"], f"best_model_{run_id}.pth")
    history = []
    val_mae_matrix = defaultdict(list)

    for epoch in range(config["EPOCHS"]):
        model.train()
        manager = BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
        h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
        pbar = tqdm(total=manager.total_steps, desc=f"Epoch {epoch+1}/{config['EPOCHS']} Training", colour="blue")

        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]
        print(
            f"[Epoch {epoch+1}] start | lambda_ah={curr_lambda:.2f} | "
            f"train_windows={len(train_ds)} | val_windows={len(val_ds)} | steps={manager.total_steps} | "
            "p_reset=disabled"
        )

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_subsegment_starts = 0

        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [train_ds[i] for i in indices]
            x = torch.stack([s["x_dyn"] for s in samples]).to(config["DEVICE"])
            t = torch.stack([s["t_mean"] for s in samples]).to(config["DEVICE"])
            y = torch.stack([s["soc"] for s in samples]).to(config["DEVICE"])
            Q = torch.stack([s["Q"] for s in samples]).to(config["DEVICE"])
            m_t = torch.tensor(masks, dtype=torch.float32, device=config["DEVICE"])
            f_t = torch.tensor(is_first, dtype=torch.bool, device=config["DEVICE"])

            h_state = h_state.detach()
            if f_t.any():
                for i in range(config["BATCH_SIZE"]):
                    if f_t[i]:
                        h_state[:, i, :] = 0.0
                epoch_subsegment_starts += int(f_t.sum().item())

            x_n, t_n = scaler.transform(x, t)
            y_p, h_state = model(x_n, t_n, h_state)
            loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], Q, f_t, curr_lambda, m_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_data_losses.append(l_d.item())
            epoch_ah_losses.append(l_ah.item())
            epoch_range_losses.append(l_range.item())

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Start": int(f_t.sum().item())})

        pbar.close()

        avg_val_mae, file_maes = evaluate_dataset(model, val_loader, scaler, {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]}, label=f"Epoch {epoch+1} Val")
        scheduler.step()
        for fn, mae in file_maes.items():
            val_mae_matrix[fn].append(mae)

        history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
                "Train_Data_Loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan"),
                "Train_Ah_Loss": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
                "Train_Range_Loss": float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan"),
                "Val_MAE_Avg": avg_val_mae,
                "Lambda": curr_lambda,
                "Subsegment_Starts": epoch_subsegment_starts,
            }
        )

        pd.DataFrame(history).to_csv(os.path.join(config["CSV_DIR"], f"log_{run_id}.csv"), index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(os.path.join(config["CSV_DIR"], f"val_matrix_{run_id}.csv"))

        train_total = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        train_data = float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan")
        train_ah = float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan")
        train_range = float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan")
        print(
            f"[Epoch {epoch+1}] summary | "
            f"train_total={train_total:.6f} | "
            f"train_data={train_data:.6f} | "
            f"train_ah={train_ah:.6f} | "
            f"train_range={train_range:.6f} | "
            f"val_mae={avg_val_mae:.4f} | lambda_ah={curr_lambda:.2f} | "
            f"subsegment_starts={epoch_subsegment_starts}"
        )

        if avg_val_mae < best_overall_mae:
            best_overall_mae = avg_val_mae
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                },
                best_model_path,
            )
            print(f"[Epoch {epoch+1}] new best model saved: val_mae={best_overall_mae:.4f}")

    print(f"Training finished. Best val MAE = {best_overall_mae:.4f}")

    if not os.path.exists(best_model_path):
        raise RuntimeError("Training completed without a valid validation checkpoint")
    best_ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best checkpoint for final evaluation.")

    evaluate_soc_start_splits(model, scaler, datasets, config, run_id)


def eval_latest_checkpoint(starts: list[float], soc_batch_size: int):
    config = make_config()
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]))
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    fresh_config = make_config()
    config["CACHE_DIR"] = fresh_config["CACHE_DIR"]
    config["PTH_DIR"] = fresh_config["PTH_DIR"]
    config["CSV_DIR"] = fresh_config["CSV_DIR"]
    config["META_DIR"] = fresh_config["META_DIR"]
    config["VAL_BATCH_SIZE"] = fresh_config["VAL_BATCH_SIZE"]
    config["TEMPS"] = ckpt_config.get("TEMPS", TEMP_TARGETS)
    config["TEMP_TOLERANCE"] = ckpt_config.get("TEMP_TOLERANCE", TEMP_TOLERANCE)
    config["SOC_STARTS"] = starts
    config["SOC_BATCH_SIZE"] = soc_batch_size

    print("=== SOC-start evaluation for latest filtered-temperature augmented checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print_config(config)
    print(f"SOC_STARTS : {config['SOC_STARTS']}")
    print(f"SOC_BATCH  : {config['SOC_BATCH_SIZE']}")

    datasets = load_all_datasets(config)
    train_ds, _, _, _ = datasets
    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"soc_start_eval_{ckpt_path.stem}_{timestamp}"
    evaluate_soc_start_splits(model, scaler, datasets, config, run_id)


def eval_latest_soc_interval_checkpoint(intervals: list[tuple[float, float]], soc_batch_size: int):
    config = make_config()
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]))
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    fresh_config = make_config()
    config["CACHE_DIR"] = fresh_config["CACHE_DIR"]
    config["PTH_DIR"] = fresh_config["PTH_DIR"]
    config["CSV_DIR"] = fresh_config["CSV_DIR"]
    config["META_DIR"] = fresh_config["META_DIR"]
    config["TEMPS"] = ckpt_config.get("TEMPS", TEMP_TARGETS)
    config["TEMP_TOLERANCE"] = ckpt_config.get("TEMP_TOLERANCE", TEMP_TOLERANCE)
    config["SOC_INTERVAL_BATCH_SIZE"] = soc_batch_size
    config["SHOW_PROGRESS"] = False

    print("=== SOC-interval evaluation for latest filtered-temperature SOC-start augmentation checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print_config(config)
    print(f"SOC_INTERVALS: {intervals}")
    print(f"SOC_BATCH    : {config['SOC_INTERVAL_BATCH_SIZE']}")

    datasets = load_all_datasets(config)
    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"soc_interval_eval_{ckpt_path.stem}_{timestamp}"
    rows = evaluate_soc_intervals(model, scaler, datasets, config, run_id, intervals)
    metrics_df = write_soc_interval_metrics(rows, config["CSV_DIR"], run_id)
    total_df = metrics_df[metrics_df["metric_level"].eq("total")]
    print("\n=== SOC-interval total metrics ===")
    print(total_df[["split", "soc_interval", "avg_mae", "files", "windows"]].to_string(index=False))


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate filtered-temperature causal AdaLN SOC-start augmentation baseline.")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "interval-eval"],
        default=None,
        help="Non-interactive mode. train=1, eval=2, interval-eval=SOC interval test.",
    )
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--intervals", default="100-90,90-80,80-70,70-60,60-50,50-40,40-30,30-20,20-10,10-0")
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        config = make_config()
        config["SOC_STARTS"] = parse_float_list(args.starts)
        config["SOC_BATCH_SIZE"] = args.soc_batch_size
        print_config(config)
        print(f"SOC_STARTS : {config['SOC_STARTS']}")
        print(f"SOC_BATCH  : {config['SOC_BATCH_SIZE']}")
        print("Dry run only. No training or evaluation executed.")
        return

    mode = args.mode
    if mode is None:
        print("Select run mode:")
        print("1 - start a new training run")
        print("2 - evaluate the latest checkpoint under the result path with multi-SOC-start testing")
        print("3 - evaluate the latest checkpoint within user-defined SOC intervals")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            mode = "train"
        elif choice == "2":
            mode = "eval"
        elif choice == "3":
            args.intervals = ",".join(f"{high:g}-{low:g}" for high, low in prompt_soc_intervals())
            mode = "interval-eval"
        else:
            raise ValueError(f"Invalid choice: {choice}. Expected 1, 2, or 3.")

    if mode == "train":
        train_causal_adaln_dropout()
    elif mode == "eval":
        eval_latest_checkpoint(parse_float_list(args.starts), args.soc_batch_size)
    elif mode == "interval-eval":
        eval_latest_soc_interval_checkpoint(parse_soc_intervals(args.intervals), args.soc_batch_size)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
