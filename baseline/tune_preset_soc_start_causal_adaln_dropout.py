from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
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

for path in (str(BASE_DIR), str(SHARED_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from eval_soc_start_causal_adaln_dropout import filter_from_soc_start
from project_paths import baseline_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, split_soccc_by_cells
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_P_RESET_SOC_START"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cache_name(split_name: str, window_size: int, stride: int) -> str:
    base = f"{split_name}_cache_causal_adaln_dropout"
    if window_size == 100 and stride == 100:
        return f"{base}.pt"
    return f"{base}_w{window_size}_s{stride}.pt"


def make_results_dirs(search_id: str):
    base = baseline_results_dir()
    cache_dir = base / "pt"
    run_dir = base / "p_reset_soc_start" / search_id
    pth_dir = run_dir / "pth_save"
    csv_dir = run_dir / "csv_save"
    meta_dir = run_dir / "metadata"
    for directory in (cache_dir, pth_dir, csv_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def load_datasets(config: dict):
    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    split_lists = {
        "train": train_f,
        "val": val_f,
        "test_random": test_random_f,
        "test_fixed": test_fixed_f,
    }
    empty_splits = [name for name, files in split_lists.items() if not files]
    if empty_splits:
        raise RuntimeError(f"Empty data split(s): {', '.join(empty_splits)}")

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        train_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], cache_name("train", config["WINDOW_SIZE"], config["STRIDE"])),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        val_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], cache_name("val", config["WINDOW_SIZE"], config["STRIDE"])),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_random_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], cache_name("test_random", config["WINDOW_SIZE"], config["STRIDE"])),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_fixed_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], cache_name("test_fixed", config["WINDOW_SIZE"], config["STRIDE"])),
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


def train_best_checkpoint(config: dict, train_ds, val_ds, scaler: PITDScaler, csv_dir: Path, pth_dir: Path):
    set_seed(config["SEED"])
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    model = build_model(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_path = pth_dir / f"best_model_{config['TRIAL_ID']}.pth"
    history = []
    val_mae_matrix = defaultdict(list)

    print(f"\n=== Trial {config['TRIAL_ID']} | P_RESET={config['P_RESET']} ===")
    for epoch in range(config["EPOCHS"]):
        model.train()
        manager = BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
        h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_reset_lanes = 0
        epoch_reset_batches = 0

        pbar = tqdm(
            total=manager.total_steps,
            desc=f"{config['TRIAL_ID']} epoch {epoch + 1}/{config['EPOCHS']}",
            colour="blue",
            disable=config.get("DISABLE_TQDM", False),
        )
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
            if f_t.any():
                h_state[:, f_t, :] = 0.0

            active_mask = m_t > 0.5
            eligible_mask = active_mask & (~f_t)
            reset_mask = (torch.rand(config["BATCH_SIZE"], device=config["DEVICE"]) < config["P_RESET"]) & eligible_mask
            reset_count = int(reset_mask.sum().item())
            if reset_count > 0:
                h_state[:, reset_mask, :] = 0.0
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "reset": reset_count})
        pbar.close()

        val_mae, file_maes = evaluate_dataset(
            model,
            val_loader,
            scaler,
            {
                "DEVICE": config["DEVICE"],
                "D_MODEL": config["D_MODEL"],
                "DISABLE_TQDM": config.get("DISABLE_TQDM", False),
            },
            label=f"{config['TRIAL_ID']} epoch {epoch + 1} val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        row = {
            "trial_id": config["TRIAL_ID"],
            "p_reset": config["P_RESET"],
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
            "train_data_loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan"),
            "train_ah_loss": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
            "train_range_loss": float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan"),
            "val_mae": val_mae,
            "lambda_ah": curr_lambda,
            "reset_lanes": epoch_reset_lanes,
            "reset_batches": epoch_reset_batches,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(csv_dir / f"log_{config['TRIAL_ID']}.csv", index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(csv_dir / f"val_matrix_{config['TRIAL_ID']}.csv")

        print(
            f"[{config['TRIAL_ID']}] epoch={epoch + 1} "
            f"train_loss={row['train_loss']:.6f} val_mae={val_mae:.6f} p_reset={config['P_RESET']}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                },
                best_model_path,
            )
            print(f"[{config['TRIAL_ID']}] new best checkpoint: epoch={best_epoch} val_mae={best_val_mae:.6f}")

    return best_model_path, best_epoch, best_val_mae


def evaluate_soc_starts(config: dict, checkpoint_path: Path, scaler: PITDScaler, datasets: tuple, csv_dir: Path):
    _, _, test_random_ds, test_fixed_ds = datasets
    ckpt = load_torch_file(checkpoint_path, map_location=config["DEVICE"])
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    for split_order, (split_name, base_dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        for soc_order, soc_start in enumerate(config["SOC_STARTS"], start=1):
            filtered_ds = filter_from_soc_start(base_dataset, soc_start)
            if len(filtered_ds) == 0:
                rows.append(
                    {
                        "trial_id": config["TRIAL_ID"],
                        "trial_index": config["TRIAL_INDEX"],
                        "p_reset": config["P_RESET"],
                        "split_order": split_order,
                        "split": split_name,
                        "soc_start_order": soc_order,
                        "soc_start_percent": soc_start,
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
                label=f"{config['TRIAL_ID']} {split_name} SOC<= {soc_start:g}%",
            )
            rows.append(
                    {
                        "trial_id": config["TRIAL_ID"],
                        "trial_index": config["TRIAL_INDEX"],
                        "p_reset": config["P_RESET"],
                        "split_order": split_order,
                        "split": split_name,
                        "soc_start_order": soc_order,
                        "soc_start_percent": soc_start,
                        "avg_mae": avg_mae,
                        "files": len(file_maes),
                    "windows": len(filtered_ds),
                    "batch_size": config["SOC_BATCH_SIZE"],
                }
            )

    trial_path = csv_dir / f"soc_start_metrics_{config['TRIAL_ID']}.csv"
    pd.DataFrame(rows).to_csv(trial_path, index=False, encoding="utf-8-sig")
    print(f"[{config['TRIAL_ID']}] SOC-start metrics saved: {trial_path}")
    return rows


def run_trial(config: dict):
    try:
        datasets = load_datasets(config)
        scaler = PITDScaler()
        scaler.fit(datasets[0])
        checkpoint_path, best_epoch, best_val_mae = train_best_checkpoint(
            config,
            datasets[0],
            datasets[1],
            scaler,
            Path(config["CSV_DIR"]),
            Path(config["PTH_DIR"]),
        )
        rows = evaluate_soc_starts(config, checkpoint_path, scaler, datasets, Path(config["CSV_DIR"]))
        for row in rows:
            row.update(
                {
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                    "checkpoint": str(checkpoint_path),
                    "lr": config["LR"],
                    "dropout": config["DROPOUT"],
                    "d_model": config["D_MODEL"],
                    "num_layers": config["NUM_LAYERS"],
                    "batch_size_train": config["BATCH_SIZE"],
                }
            )
        return rows
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [
            {
                "trial_id": config["TRIAL_ID"],
                "trial_index": config["TRIAL_INDEX"],
                "p_reset": config["P_RESET"],
                "split_order": math.nan,
                "split": "",
                "soc_start_order": math.nan,
                "soc_start_percent": float("nan"),
                "avg_mae": float("nan"),
                "files": 0,
                "windows": 0,
                "batch_size": config["SOC_BATCH_SIZE"],
                "best_epoch": math.nan,
                "best_val_mae": math.nan,
                "checkpoint": "",
                "lr": config["LR"],
                "dropout": config["DROPOUT"],
                "d_model": config["D_MODEL"],
                "num_layers": config["NUM_LAYERS"],
                "batch_size_train": config["BATCH_SIZE"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ]


def write_summary(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ("trial_index", "split_order", "soc_start_order") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def main():
    parser = argparse.ArgumentParser(description="Tune P_RESET and evaluate each model on SOC-start test tasks.")
    parser.add_argument("--p-resets", default="0,0.025,0.05,0.75,0.1,0.15,0.2,0.25")
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-trials", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.parallel_trials < 1:
        raise ValueError("--parallel-trials must be >= 1")

    p_resets = parse_float_list(args.p_resets)
    soc_starts = parse_float_list(args.starts)
    search_id = datetime.now().strftime("p_reset_soc_%m%d_%H%M%S")
    run_dir = baseline_results_dir() / "p_reset_soc_start" / search_id
    cache_dir = baseline_results_dir() / "pt"
    pth_dir = run_dir / "pth_save"
    csv_dir = run_dir / "csv_save"
    meta_dir = run_dir / "metadata"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_config = {
        "PROJECT": PROJECT,
        "DATA_DIR": str(processed_segments_dir()),
        "SPLIT_FILE": str(split_file_path()),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "SOC_BATCH_SIZE": args.soc_batch_size,
        "LR": 2e-4,
        "D_MODEL": 64,
        "NHEAD": 4,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10.0,
        "LAMBDA_AH_STEP": 20.0,
        "WEIGHT_DECAY": 1e-5,
        "GRAD_CLIP": 1.0,
        "DEVICE": device,
        "WINDOW_SIZE": args.window_size,
        "STRIDE": args.stride,
        "SOC_STARTS": soc_starts,
        "SHOW_PROGRESS": args.show_progress,
        "DISABLE_TQDM": (args.parallel_trials > 1) or (not args.show_progress),
    }

    trial_configs = []
    for index, p_reset in enumerate(p_resets, start=1):
        p_label = str(p_reset).replace(".", "p")
        trial_id = f"{search_id}_trial{index:03d}_preset{p_label}"
        trial_configs.append(
            {
                **base_config,
                "P_RESET": p_reset,
                "TRIAL_INDEX": index,
                "TRIAL_ID": trial_id,
                "SEED": args.seed + index - 1,
            }
        )

    print("=== P_RESET SOC-start tuning ===")
    print(f"SEARCH_ID       : {search_id}")
    print(f"DEVICE          : {device}")
    print(f"P_RESET values  : {p_resets}")
    print(f"SOC starts      : {soc_starts}")
    print(f"EPOCHS          : {args.epochs}")
    print(f"PARALLEL_TRIALS : {args.parallel_trials}")
    print(f"CSV_DIR         : {csv_dir}")
    print("Fixed mainline hyperparameters: LR=2e-4, DROPOUT=0.1, D_MODEL=64, NHEAD=4, NUM_LAYERS=2")

    if args.dry_run:
        print("Dry run only. Generated trials:")
        for config in trial_configs:
            print(f"{config['TRIAL_ID']}: P_RESET={config['P_RESET']}")
        return

    cache_dir, pth_dir, csv_dir, meta_dir = make_results_dirs(search_id)
    for config in trial_configs:
        config.update(
            {
                "CACHE_DIR": str(cache_dir),
                "PTH_DIR": str(pth_dir),
                "CSV_DIR": str(csv_dir),
                "META_DIR": str(meta_dir),
            }
        )

    metadata = {
        "search_id": search_id,
        "args": vars(args),
        "p_resets": p_resets,
        "soc_starts": soc_starts,
        "fixed_config": base_config,
        "trial_ids": [config["TRIAL_ID"] for config in trial_configs],
    }
    (meta_dir / "search_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Prebuilding shared dataset caches...")
    cache_datasets = load_datasets(trial_configs[0])
    cache_scaler = PITDScaler()
    cache_scaler.fit(cache_datasets[0])
    del cache_scaler
    del cache_datasets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary_rows = []
    summary_path = csv_dir / f"p_reset_soc_start_summary_{search_id}.csv"

    if args.parallel_trials == 1:
        for config in trial_configs:
            rows = run_trial(config)
            summary_rows.extend(rows)
            write_summary(summary_rows, summary_path)
            print(f"[{config['TRIAL_ID']}] summary updated: {summary_path}")
    else:
        worker_count = min(args.parallel_trials, len(trial_configs))
        print(f"Launching {worker_count} parallel trial processes.")
        mp_context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_to_config = {executor.submit(run_trial, config): config for config in trial_configs}
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    rows = future.result()
                except BaseException as exc:
                    rows = [
                        {
                            "trial_id": config["TRIAL_ID"],
                            "trial_index": config["TRIAL_INDEX"],
                            "p_reset": config["P_RESET"],
                            "split_order": math.nan,
                            "split": "",
                            "soc_start_order": math.nan,
                            "soc_start_percent": float("nan"),
                            "avg_mae": float("nan"),
                            "files": 0,
                            "windows": 0,
                            "batch_size": config["SOC_BATCH_SIZE"],
                            "best_epoch": math.nan,
                            "best_val_mae": math.nan,
                            "checkpoint": "",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ]
                    print(f"[{config['TRIAL_ID']}] worker crashed: {type(exc).__name__}: {exc}")
                summary_rows.extend(rows)
                write_summary(summary_rows, summary_path)
                status = "failed" if any(row.get("error") for row in rows) else "finished"
                print(f"[{config['TRIAL_ID']}] {status}; summary updated: {summary_path}")

    final_df = write_summary(summary_rows, summary_path)
    print("\n=== P_RESET SOC-start tuning finished ===")
    print(final_df.to_string(index=False))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
