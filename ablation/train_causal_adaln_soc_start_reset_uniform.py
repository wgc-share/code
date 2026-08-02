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
from soccc_schemeB import BatteryTDGCMDataset, parse_segment_filename, split_soccc_by_cells
from soc_interval_eval import evaluate_soc_intervals, parse_soc_intervals, prompt_soc_intervals, write_soc_interval_metrics
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed
from tune_preset_soc_start_causal_adaln_dropout import aggregate_segment_rows


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_SOC_START_RESET_UNIFORM"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def temp_cache_suffix(temps: list[float], tolerance: float) -> str:
    labels = []
    for temp in temps:
        if float(temp).is_integer():
            labels.append(str(int(temp)))
        else:
            labels.append(str(temp).replace(".", "p"))
    if float(tolerance).is_integer():
        tol_label = str(int(tolerance))
    else:
        tol_label = str(tolerance).replace(".", "p")
    return "T" + "_".join(labels) + f"_pm{tol_label}"


def filter_files_by_temperature(files: list[str], temps: list[float], tolerance: float) -> list[str]:
    targets = [float(temp) for temp in temps]
    kept = []
    for filename in files:
        temp_c = float(parse_segment_filename(filename).get("temp_C", float("nan")))
        if any(abs(temp_c - target) <= tolerance for target in targets):
            kept.append(filename)
    return kept


def _make_results_dirs():
    base = organized_results_dir() / "ablation" / "soc_start_reset_uniform"
    cache_dir = base / "pt"
    pth_dir = base / "pth_save"
    csv_dir = base / "csv_save"
    meta_dir = base / "metadata"
    for directory in (cache_dir, pth_dir, csv_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def _results_dirs_no_create():
    base = organized_results_dir() / "ablation" / "soc_start_reset_uniform"
    return base / "pt", base / "pth_save", base / "csv_save", base / "metadata"


def assign_reset_starts(train_ds: BatteryTDGCMDataset, low: float, high: float, seed: int):
    if high < low:
        low, high = high, low
    if low < 0.0 or high > 100.0:
        raise ValueError(f"reset range must stay within [0, 100], got [{low}, {high}]")
    rng = np.random.default_rng(seed)
    filenames = sorted({sample["filenames"] for sample in train_ds.samples})
    return {filename: float(rng.uniform(low, high)) for filename in filenames}


def latest_checkpoint(pth_dir: Path):
    ckpts = list(pth_dir.glob(f"best_model_{PROJECT}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {pth_dir}. Run training first.")
    return max(ckpts, key=lambda path: path.stat().st_mtime)


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
                        "reset_strategy": "soc_start_reset_uniform",
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
                    "reset_strategy": "soc_start_reset_uniform",
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
                row["reset_strategy"] = "soc_start_reset_uniform"
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
    total_path = Path(csv_dir) / f"soc_start_total_summary_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    df[df["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"SOC-start detailed metrics saved: {out_path}")
    print(f"SOC-start total metrics saved: {total_path}")
    return df


def make_base_config(args: argparse.Namespace, create_dirs: bool = True):
    if create_dirs:
        cache_dir, pth_dir, csv_dir, meta_dir = _make_results_dirs()
    else:
        cache_dir, pth_dir, csv_dir, meta_dir = _results_dirs_no_create()
        csv_dir.mkdir(parents=True, exist_ok=True)

    soc_starts = parse_float_list(args.starts)
    temps = parse_float_list(args.temps)
    temp_tolerance = float(args.temp_tolerance)
    reset_low = float(args.reset_low)
    reset_high = float(args.reset_high)
    return {
        "PROJECT": PROJECT,
        "DATA_DIR": str(processed_segments_dir()),
        "SPLIT_FILE": str(split_file_path()),
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
        "RESET_STRATEGY": "soc_start_reset_uniform",
        "RESET_DISTRIBUTION": "uniform",
        "RESET_LOW": reset_low,
        "RESET_HIGH": reset_high,
        "SOC_STARTS": soc_starts,
        "TEMPS": temps,
        "TEMP_TOLERANCE": temp_tolerance,
        "SHOW_PROGRESS": args.show_progress,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
    }


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
    empty_splits = [
        name
        for name, files in {
            "train": train_f,
            "val": val_f,
            "test_random": test_random_f,
            "test_fixed": test_fixed_f,
        }.items()
        if not files
    ]
    if empty_splits:
        raise RuntimeError(f"Empty split(s) after temperature filtering: {', '.join(empty_splits)}")
    suffix = temp_cache_suffix(config["TEMPS"], config["TEMP_TOLERANCE"])

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        train_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"train_cache_causal_adaln_soc_start_reset_uniform_{suffix}.pt"),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        val_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"val_cache_causal_adaln_soc_start_reset_uniform_{suffix}.pt"),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_random_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"test_random_cache_causal_adaln_soc_start_reset_uniform_{suffix}.pt"),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_fixed_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], f"test_fixed_cache_causal_adaln_soc_start_reset_uniform_{suffix}.pt"),
    )
    return train_ds, val_ds, test_random_ds, test_fixed_ds


def train_soc_start_reset(args: argparse.Namespace):
    set_seed(args.seed)
    config = make_base_config(args, create_dirs=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print("=== Ablation: causal + AdaLN + single SOC-start hidden-state reset ===")
    print(f"DATA_DIR     : {config['DATA_DIR']}")
    print(f"SPLIT_FILE   : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR    : {config['CACHE_DIR']}")
    print(f"PTH_DIR      : {config['PTH_DIR']}")
    print(f"CSV_DIR      : {config['CSV_DIR']}")
    print(f"DEVICE       : {config['DEVICE']}")
    print(f"BATCH_SIZE   : {config['BATCH_SIZE']}")
    print(f"VAL_BATCH    : {config['VAL_BATCH_SIZE']} (deterministic stateful lanes)")
    print(f"SOC_BATCH    : {config['SOC_BATCH_SIZE']}")
    print(f"RESET_DIST   : uniform [{config['RESET_LOW']}, {config['RESET_HIGH']}]")
    print(f"SOC_STARTS   : {config['SOC_STARTS']}")
    print(f"TEMPS        : {config['TEMPS']}")
    print(f"TEMP_TOL     : +/-{config['TEMP_TOLERANCE']} C")
    print("TRANSFORMER  : causal mask enabled")
    print("TEMP MOD     : AdaLN")

    datasets = load_all_datasets(config)
    train_ds, val_ds, _, _ = datasets

    reset_start_by_file = assign_reset_starts(train_ds, config["RESET_LOW"], config["RESET_HIGH"], args.seed)
    pd.DataFrame(
        [{"filename": filename, "reset_start_soc_percent": start} for filename, start in sorted(reset_start_by_file.items())]
    ).to_csv(os.path.join(config["META_DIR"], f"soc_start_reset_uniform_assignment_{run_id}.csv"), index=False, encoding="utf-8-sig")
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
        lane_reset_done = [False] * config["BATCH_SIZE"]
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
                reset_start = reset_start_by_file[filename]
                current_start_soc = float(sample["soc"][0, 0]) * 100.0

                if is_first[lane_idx] or lane_files[lane_idx] != filename:
                    h_state[:, lane_idx, :] = 0.0
                    lane_files[lane_idx] = filename
                    lane_reset_done[lane_idx] = reset_start >= 100.0
                    continue

                if (not lane_reset_done[lane_idx]) and current_start_soc <= reset_start:
                    h_state[:, lane_idx, :] = 0.0
                    lane_reset_done[lane_idx] = True
                    reset_count += 1

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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "SOCStartReset": reset_count})
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
            "SOC_Start_Reset_Lanes": epoch_reset_lanes,
            "SOC_Start_Reset_Batches": epoch_reset_batches,
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
            f"soc_start_reset_uniform_lanes={epoch_reset_lanes} | "
            f"soc_start_reset_uniform_batches={epoch_reset_batches}"
        )

        if avg_val_mae < best_overall_mae:
            best_overall_mae = avg_val_mae
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                    "reset_start_by_file": reset_start_by_file,
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


def eval_latest_checkpoint(args: argparse.Namespace):
    config = make_base_config(args, create_dirs=False)
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]))
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir, pth_dir, csv_dir, meta_dir = _results_dirs_no_create()
    config["CACHE_DIR"] = str(cache_dir)
    config["PTH_DIR"] = str(pth_dir)
    config["CSV_DIR"] = str(csv_dir)
    config["META_DIR"] = str(meta_dir)
    config["SOC_BATCH_SIZE"] = args.soc_batch_size
    config["SOC_STARTS"] = parse_float_list(args.starts)
    config["TEMPS"] = ckpt_config.get("TEMPS", parse_float_list(args.temps))
    config["TEMP_TOLERANCE"] = ckpt_config.get("TEMP_TOLERANCE", float(args.temp_tolerance))
    config["SHOW_PROGRESS"] = args.show_progress

    print("=== Evaluate latest SOC-start-reset checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"SOC_BATCH  : {config['SOC_BATCH_SIZE']}")
    print(f"SOC_STARTS : {config['SOC_STARTS']}")
    print(f"TEMPS      : {config['TEMPS']}")
    print(f"TEMP_TOL   : +/-{config['TEMP_TOLERANCE']} C")

    datasets = load_all_datasets(config)
    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"eval_{ckpt_path.stem}_{timestamp}"
    rows = evaluate_soc_starts(model, scaler, datasets, config, run_id)
    metrics_df = write_soc_metrics(rows, config["CSV_DIR"], run_id)
    total_df = metrics_df[metrics_df["metric_level"].eq("total")]
    print("\n=== SOC-start total metrics ===")
    print(total_df[["split", "soc_start_percent", "avg_mae", "files", "windows"]].to_string(index=False))


def eval_latest_soc_interval_checkpoint(args: argparse.Namespace, intervals: list[tuple[float, float]]):
    config = make_base_config(args, create_dirs=False)
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]))
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir, pth_dir, csv_dir, meta_dir = _results_dirs_no_create()
    config["CACHE_DIR"] = str(cache_dir)
    config["PTH_DIR"] = str(pth_dir)
    config["CSV_DIR"] = str(csv_dir)
    config["META_DIR"] = str(meta_dir)
    config["TEMPS"] = ckpt_config.get("TEMPS", parse_float_list(args.temps))
    config["TEMP_TOLERANCE"] = ckpt_config.get("TEMP_TOLERANCE", float(args.temp_tolerance))
    config["SOC_INTERVAL_BATCH_SIZE"] = args.soc_batch_size
    config["SHOW_PROGRESS"] = args.show_progress

    print("=== SOC-interval evaluation for latest SOC-start-reset checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"SOC_BATCH  : {config['SOC_INTERVAL_BATCH_SIZE']}")
    print(f"INTERVALS  : {intervals}")
    print(f"TEMPS      : {config['TEMPS']}")
    print(f"TEMP_TOL   : +/-{config['TEMP_TOLERANCE']} C")

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


def main():
    parser = argparse.ArgumentParser(description="Train causal AdaLN model with one uniformly sampled SOC-start hidden-state reset per segment.")
    parser.add_argument("--mode", choices=["train", "eval", "interval-eval"], default=None, help="Non-interactive mode. train=1, eval=2, interval-eval=SOC interval test.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--reset-low", type=float, default=0.0)
    parser.add_argument("--reset-high", type=float, default=100.0)
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--intervals", default="100-90,90-80,80-70,70-60,60-50,50-40,40-30,30-20,20-10,10-0")
    parser.add_argument("--temps", default="10,25,40")
    parser.add_argument("--temp-tolerance", type=float, default=2.0)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Uniform SOC-start reset dry run")
        print(f"reset distribution: U({float(args.reset_low)}, {float(args.reset_high)})")
        print(f"test SOC starts: {parse_float_list(args.starts)}")
        print(f"temperatures: {parse_float_list(args.temps)}")
        print(f"temperature tolerance: +/-{float(args.temp_tolerance)} C")
        print("Fixed hyperparameters: LR=2e-4, DROPOUT=0.1, D_MODEL=64, NHEAD=4, NUM_LAYERS=2")
        return

    mode = args.mode
    if mode is None:
        print("Select run mode:")
        print("1 - start a new training run")
        print("2 - evaluate the latest checkpoint under the result path")
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
        train_soc_start_reset(args)
    elif mode == "eval":
        eval_latest_checkpoint(args)
    elif mode == "interval-eval":
        eval_latest_soc_interval_checkpoint(args, parse_soc_intervals(args.intervals))
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
