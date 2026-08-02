from __future__ import annotations

import argparse
import json
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
BASELINE_DIR = CODE_ROOT / "baseline"

for path in (str(SHARED_DIR), str(BASELINE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from eval_soc_start_causal_adaln_dropout import filter_from_soc_start
from project_paths import organized_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, split_soccc_by_cells
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed
from tune_preset_soc_start_causal_adaln_dropout import aggregate_segment_rows


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_SOC_GUIDED_RESET"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _make_results_dirs():
    base = organized_results_dir() / "ablation" / "soc_guided_reset"
    cache_dir = base / "pt"
    pth_dir = base / "pth_save"
    csv_dir = base / "csv_save"
    meta_dir = base / "metadata"
    for directory in (cache_dir, pth_dir, csv_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def assign_reset_bins(train_ds: BatteryTDGCMDataset, choices: list[int], seed: int):
    rng = np.random.default_rng(seed)
    filenames = sorted({sample["filenames"] for sample in train_ds.samples})
    return {filename: int(rng.choice(choices)) for filename in filenames}


def soc_progress_bin(sample: dict, n_bins: int) -> int:
    start_soc_percent = float(sample["soc"][0, 0]) * 100.0
    progress_percent = min(100.0, max(0.0, 100.0 - start_soc_percent))
    interval = 100.0 / float(n_bins)
    return min(n_bins, int(progress_percent // interval))


def build_model(config: dict):
    return AdaLNBatteryTDGCMModel(
        d_model=config["D_MODEL"],
        nhead=config["NHEAD"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        use_causal=True,
    ).to(config["DEVICE"])


def evaluate_soc_starts(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str):
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
                        "p_reset": math.nan,
                        "reset_strategy": "soc_guided",
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
                    "DISABLE_TQDM": not config["SHOW_PROGRESS"],
                },
                label=f"{run_id} {split_name} SOC<= {soc_start:g}%",
            )
            rows.append(
                {
                    "trial_id": run_id,
                    "trial_index": 1,
                    "p_reset": math.nan,
                    "reset_strategy": "soc_guided",
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
                    "P_RESET": math.nan,
                },
                split_name,
                split_order,
                soc_start,
                soc_order,
                file_maes,
                filtered_ds,
            )
            for row in segment_rows:
                row["reset_strategy"] = "soc_guided"
            rows.extend(segment_rows)
    return rows


def write_soc_metrics(rows: list[dict], csv_dir: str, run_id: str):
    df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ("split_order", "soc_start_order", "metric_order", "segment_order")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")
    out_path = Path(csv_dir) / f"soc_start_metrics_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    total = df[df["metric_level"].eq("total")].copy()
    summary_path = Path(csv_dir) / f"soc_start_total_summary_{run_id}.csv"
    total.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"SOC-start detailed metrics saved: {out_path}")
    print(f"SOC-start total metrics saved: {summary_path}")
    return df


def train_soc_guided_reset(args: argparse.Namespace):
    set_seed(args.seed)
    data_dir = processed_segments_dir()
    split_file = split_file_path()
    cache_dir, pth_dir, csv_dir, meta_dir = _make_results_dirs()
    n_choices = parse_int_list(args.n_choices)
    soc_starts = parse_float_list(args.starts)

    config = {
        "PROJECT": PROJECT,
        "DATA_DIR": str(data_dir),
        "SPLIT_FILE": str(split_file),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "BATCH_SIZE": args.batch_size,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "SOC_BATCH_SIZE": args.soc_batch_size,
        "LR": 2e-4,
        "EPOCHS": args.epochs,
        "D_MODEL": 64,
        "NHEAD": 4,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10.0,
        "LAMBDA_AH_STEP": 20.0,
        "GRAD_CLIP": 1.0,
        "RESET_STRATEGY": "soc_guided",
        "N_CHOICES": n_choices,
        "SOC_STARTS": soc_starts,
        "SHOW_PROGRESS": args.show_progress,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
    }

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print("=== Ablation: causal + AdaLN + SOC-guided hidden-state reset ===")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR  : {config['CACHE_DIR']}")
    print(f"PTH_DIR    : {config['PTH_DIR']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"BATCH_SIZE : {config['BATCH_SIZE']}")
    print(f"VAL_BATCH  : {config['VAL_BATCH_SIZE']} (deterministic stateful lanes)")
    print(f"SOC_BATCH  : {config['SOC_BATCH_SIZE']}")
    print(f"N_CHOICES  : {config['N_CHOICES']} -> x = 100 / n percent SOC")
    print(f"SOC_STARTS : {config['SOC_STARTS']}")
    print("TRANSFORMER: causal mask enabled")
    print("TEMP MOD   : AdaLN")

    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    print(
        f"Split counts | train={len(train_f)} | val={len(val_f)} | "
        f"test_random={len(test_random_f)} | test_fixed={len(test_fixed_f)}"
    )

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        train_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "train_cache_causal_adaln_soc_guided_reset.pt"),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        val_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "val_cache_causal_adaln_soc_guided_reset.pt"),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_random_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_random_cache_causal_adaln_soc_guided_reset.pt"),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_fixed_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_fixed_cache_causal_adaln_soc_guided_reset.pt"),
    )
    datasets = (train_ds, val_ds, test_random_ds, test_fixed_ds)

    reset_bins_by_file = assign_reset_bins(train_ds, n_choices, args.seed)
    pd.DataFrame(
        [{"filename": filename, "n_bins": n, "x_soc_percent": 100.0 / n} for filename, n in sorted(reset_bins_by_file.items())]
    ).to_csv(os.path.join(config["META_DIR"], f"soc_guided_reset_bins_{run_id}.csv"), index=False, encoding="utf-8-sig")
    Path(config["META_DIR"], f"config_{run_id}.json").write_text(
        json.dumps({**config, "RUN_ID": run_id}, indent=2),
        encoding="utf-8",
    )

    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    scaler = PITDScaler()
    scaler.fit(train_ds)
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
        lane_files = [None] * config["BATCH_SIZE"]
        lane_bins = [None] * config["BATCH_SIZE"]
        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_reset_lanes = 0
        epoch_reset_batches = 0

        pbar = tqdm(total=manager.total_steps, desc=f"Epoch {epoch + 1}/{config['EPOCHS']} Training", colour="blue")
        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [train_ds[index] for index in indices]
            x = torch.stack([sample["x_dyn"] for sample in samples]).to(config["DEVICE"])
            t = torch.stack([sample["t_mean"] for sample in samples]).to(config["DEVICE"])
            y = torch.stack([sample["soc"] for sample in samples]).to(config["DEVICE"])
            q = torch.stack([sample["Q"] for sample in samples]).to(config["DEVICE"])
            m_t = torch.tensor(masks, dtype=torch.float32, device=config["DEVICE"])
            f_t = torch.tensor(is_first, dtype=torch.bool, device=config["DEVICE"])

            h_state = h_state.detach()
            reset_count = 0
            for lane_idx, sample in enumerate(samples):
                if masks[lane_idx] <= 0.5:
                    continue
                filename = sample["filenames"]
                n_bins = reset_bins_by_file[filename]
                current_bin = soc_progress_bin(sample, n_bins)

                if is_first[lane_idx] or lane_files[lane_idx] != filename:
                    h_state[:, lane_idx, :] = 0.0
                    lane_files[lane_idx] = filename
                    lane_bins[lane_idx] = current_bin
                    continue

                if lane_bins[lane_idx] is not None and current_bin > lane_bins[lane_idx]:
                    h_state[:, lane_idx, :] = 0.0
                    reset_count += 1
                lane_bins[lane_idx] = current_bin

            if reset_count > 0:
                epoch_reset_lanes += reset_count
                epoch_reset_batches += 1

            x_n, t_n = scaler.transform(x, t)
            y_p, h_state = model(x_n, t_n, h_state)
            loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], q, f_t, curr_lambda, m_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_data_losses.append(l_d.item())
            epoch_ah_losses.append(l_ah.item())
            epoch_range_losses.append(l_range.item())
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "SOCReset": reset_count})
        pbar.close()

        avg_val_mae, file_maes = evaluate_dataset(
            model,
            val_loader,
            scaler,
            {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]},
            label=f"Epoch {epoch + 1} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        row = {
            "Epoch": epoch + 1,
            "Train_Loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
            "Train_Data_Loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan"),
            "Train_Ah_Loss": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
            "Train_Range_Loss": float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan"),
            "Val_MAE_Avg": avg_val_mae,
            "Lambda": curr_lambda,
            "SOC_Reset_Lanes": epoch_reset_lanes,
            "SOC_Reset_Batches": epoch_reset_batches,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(config["CSV_DIR"], f"log_{run_id}.csv"), index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(os.path.join(config["CSV_DIR"], f"val_matrix_{run_id}.csv"))

        print(
            f"[Epoch {epoch + 1}] summary | "
            f"train_total={row['Train_Loss']:.6f} | "
            f"train_data={row['Train_Data_Loss']:.6f} | "
            f"train_ah={row['Train_Ah_Loss']:.6f} | "
            f"train_range={row['Train_Range_Loss']:.6f} | "
            f"val_mae={avg_val_mae:.6f} | "
            f"soc_reset_lanes={epoch_reset_lanes} | soc_reset_batches={epoch_reset_batches}"
        )

        if avg_val_mae < best_overall_mae:
            best_overall_mae = avg_val_mae
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                    "reset_bins_by_file": reset_bins_by_file,
                },
                best_model_path,
            )
            print(f"[Epoch {epoch + 1}] new best model saved: val_mae={best_overall_mae:.6f}")

    print(f"Training finished. Best val MAE = {best_overall_mae:.6f}")
    best_ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best checkpoint for SOC-start evaluation.")

    rows = evaluate_soc_starts(model, scaler, datasets, config, run_id)
    metrics_df = write_soc_metrics(rows, config["CSV_DIR"], run_id)
    total_df = metrics_df[metrics_df["metric_level"].eq("total")]
    print("\n=== SOC-start total metrics ===")
    print(total_df[["split", "soc_start_percent", "avg_mae", "files", "windows"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Train causal AdaLN model with SOC-guided hidden-state reset.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--n-choices", default="8,9,10,11,12")
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("SOC-guided reset dry run")
        print(f"n choices: {parse_int_list(args.n_choices)}")
        print(f"SOC starts: {parse_float_list(args.starts)}")
        print("Fixed hyperparameters: LR=2e-4, DROPOUT=0.1, D_MODEL=64, NHEAD=4, NUM_LAYERS=2")
        return
    train_soc_guided_reset(args)


if __name__ == "__main__":
    main()
