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

BASE_DIR = Path(__file__).resolve().parent
CODE_ROOT = BASE_DIR.parent
SHARED_DIR = CODE_ROOT / "shared"
ABLATION_DIR = CODE_ROOT / "ablation"
BASELINE_DIR = CODE_ROOT / "baseline"

for path in (str(SHARED_DIR), str(ABLATION_DIR), str(BASELINE_DIR), str(BASE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ablation_filtered_temps_runner as filtered_runner
import baseline_d96head4lay3 as base
from project_paths import organized_results_dir
from scaling import PITDScaler
from train_utils import PITDPhysicsLoss, evaluate_dataset, set_seed
from torch_io import load_torch_file, save_torch_file


METHOD = "temp_input_adapter_no_ah"
SPARSE_STEPS = [2, 5, 10]


def make_results_dirs(sparse_step: int):
    run_root = organized_results_dir() / "baseline_tune" / "sparse_temp_input_adapter_filtered" / f"n{sparse_step}"
    cache_dir = run_root / "pt"
    pth_dir = run_root / "pth_save"
    csv_dir = run_root / "csv_save"
    meta_dir = run_root / "metadata"
    for d in (cache_dir, pth_dir, csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    return run_root, cache_dir, pth_dir, csv_dir, meta_dir


def make_config(args: argparse.Namespace, sparse_step: int) -> dict:
    config = filtered_runner.make_config(METHOD, args)
    run_root, cache_dir, pth_dir, csv_dir, meta_dir = make_results_dirs(sparse_step)
    config.update(
        {
            "PROJECT": f"{config['PROJECT']}_SPARSE_N{sparse_step}",
            "ABLATION_TITLE": f"{config['ABLATION_TITLE']} | sparse n={sparse_step}",
            "CACHE_DIR": str(cache_dir),
            "PTH_DIR": str(pth_dir),
            "CSV_DIR": str(csv_dir),
            "META_DIR": str(meta_dir),
            "SPARSE_STEP": int(sparse_step),
            "RUN_ROOT": str(run_root),
        }
    )
    return config


def print_plan(args: argparse.Namespace):
    print("=== Sparse sampling filtered-temperature baseline ===", flush=True)
    print(f"METHOD      : {METHOD}", flush=True)
    print(f"SPARSE_STEPS: {SPARSE_STEPS}", flush=True)
    print(f"EPOCHS      : {args.epochs}", flush=True)
    print(f"BATCH_SIZE  : {args.batch_size}", flush=True)
    print(f"VAL_BATCH   : {args.val_batch_size}", flush=True)
    print(f"SOC_BATCH   : {args.soc_batch_size}", flush=True)
    print(f"SEED_START  : {args.seed}", flush=True)
    print("LOSS        : no Ah loss", flush=True)
    print("STATE       : GRU state transfer + p_reset=0.05", flush=True)
    print("MODEL       : causal Transformer + temperature as 6th input dim + residual adapter", flush=True)
    print("TEMP SCALAR : window mean t_mean", flush=True)
    print("DATA        : filtered temperatures 10C, 25C, 40C within +/-2C", flush=True)
    print("EVAL        : full SOC + multi-SOC-start on filtered temperatures only", flush=True)
    print("", flush=True)


def summarize_final(test_df: pd.DataFrame, soc_df: pd.DataFrame, config: dict, run_id: str, best_epoch, best_val_mae):
    def mv(df, split):
        rows = df[df["metric_level"].eq("total") & df["split"].eq(split)]
        return float(pd.to_numeric(rows["avg_mae"], errors="coerce").mean()) if not rows.empty else float("nan")

    row = {
        "run_id": run_id,
        "sparse_step": int(config["SPARSE_STEP"]),
        "method": config["ABLATION"],
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_random_mae": mv(test_df, "test_random"),
        "test_fixed_mae": mv(test_df, "test_fixed"),
        "soc_random_mean_mae": mv(soc_df, "test_random"),
        "soc_fixed_mean_mae": mv(soc_df, "test_fixed"),
    }
    row["test_mean_mae"] = float(np.nanmean([row["test_random_mae"], row["test_fixed_mae"]]))
    row["soc_mean_mae"] = float(np.nanmean([row["soc_random_mean_mae"], row["soc_fixed_mean_mae"]]))
    out = Path(config["CSV_DIR"]) / f"final_summary_{run_id}.csv"
    pd.DataFrame([row]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Final summary saved: {out}")
    print(pd.DataFrame([row]).T.to_string(header=False))
    return row


def train_single_sparse_run(sparse_step: int, args: argparse.Namespace, seed: int, round_idx: int) -> dict:
    config = make_config(args, sparse_step)
    set_seed(seed)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"{config['PROJECT']}_r{round_idx:02d}_s{seed}_{timestamp}"

    filtered_runner.print_config(config)
    datasets = base.load_all_datasets(config)
    train_ds, val_ds, _, _ = datasets
    print(
        f"Dataset windows | train={len(train_ds)} | val={len(val_ds)} | "
        f"test_random={len(datasets[2])} | test_fixed={len(datasets[3])}",
        flush=True,
    )

    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    scaler = PITDScaler()
    scaler.fit(train_ds)
    model = filtered_runner.build_model(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-5)
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])

    best_val_mae = float("inf")
    best_epoch = -1
    best_model_path = Path(config["PTH_DIR"]) / f"best_model_{run_id}.pth"
    history = []
    val_mae_matrix = defaultdict(list)

    for epoch in range(1, config["EPOCHS"] + 1):
        losses, curr_lambda, reset_lanes, reset_batches = filtered_runner.train_one_epoch_stateful(
            model, train_ds, scaler, optimizer, criterion, config, epoch
        )

        avg_val_mae, file_maes = evaluate_dataset(
            model,
            val_loader,
            scaler,
            {
                "DEVICE": config["DEVICE"],
                "D_MODEL": config["D_MODEL"],
                "DISABLE_TQDM": config.get("DISABLE_TQDM", False),
            },
            label=f"Epoch {epoch} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            save_torch_file(
                {"model_state_dict": model.state_dict(), "scaler_stats": scaler.stats, "config": config},
                best_model_path,
            )
            print(f"[Epoch {epoch}] new best model saved: val_mae={best_val_mae:.6f}")

        row = {
            "Epoch": epoch,
            "LR": float(optimizer.param_groups[0]["lr"]),
            "Train_Loss": float(np.mean(losses["total"])) if losses["total"] else float("nan"),
            "Train_Data_Loss": float(np.mean(losses["data"])) if losses["data"] else float("nan"),
            "Train_Ah_Loss": float(np.mean(losses["ah"])) if losses["ah"] else float("nan"),
            "Train_SOC_Mono_Loss": float(np.mean(losses["soc_mono"])) if losses["soc_mono"] else float("nan"),
            "Train_Range_Loss": float(np.mean(losses["range"])) if losses["range"] else float("nan"),
            "Val_MAE_Avg": avg_val_mae,
            "Best_Val_MAE_So_Far": best_val_mae,
            "Best_Epoch_So_Far": best_epoch,
            "Lambda": curr_lambda,
            "Reset_Lanes": reset_lanes,
            "Reset_Batches": reset_batches,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(Path(config["CSV_DIR"]) / f"log_{run_id}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_mae_matrix).T.to_csv(Path(config["CSV_DIR"]) / f"val_matrix_{run_id}.csv", encoding="utf-8-sig")
        print(
            f"[Epoch {epoch}] summary | train={row['Train_Loss']:.6f} | data={row['Train_Data_Loss']:.6f} | "
            f"ah={row['Train_Ah_Loss']:.6f} | mono={row['Train_SOC_Mono_Loss']:.6f} | "
            f"val={avg_val_mae:.6f} | best={best_val_mae:.6f} | best_epoch={best_epoch}",
            flush=True,
        )

    ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded best checkpoint for final evaluation: {best_model_path}")

    test_df = base.evaluate_test_splits(model, scaler, datasets, config, run_id)
    print("\nRunning multi-SOC-start evaluation...")
    soc_df = base.evaluate_soc_start_splits(model, scaler, datasets, config, run_id)

    row = summarize_final(test_df, soc_df, config, run_id, best_epoch, best_val_mae)
    row["round_idx"] = round_idx
    row["seed"] = seed
    return row


def main():
    parser = argparse.ArgumentParser(description="Sparse-sampling loop for the filtered-temperature temp-input+adapter baseline.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=str, default="2,5,10", help="Comma-separated sparse sampling factors.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of repeated rounds over the sparse-step set.")
    parser.add_argument("--lambda-soc-mono", type=float, default=None)
    parser.add_argument("--soc-mono-segments", type=int, default=None)
    parser.add_argument("--soc-mono-current-thr", type=float, default=None)
    parser.add_argument("--soc-mono-margin", type=float, default=None)
    args = parser.parse_args()

    step_values = [int(x) for x in args.steps.split(",") if x.strip()]
    if not step_values:
        raise ValueError("--steps cannot be empty")

    print_plan(args)
    print(f"Requested sparse steps: {step_values}", flush=True)
    print(f"Rounds per sparse-step set: {args.rounds}", flush=True)
    if args.dry_run:
        print("Dry run only. No training executed.", flush=True)
        return

    rows = []
    if args.rounds <= 0:
        raise ValueError(f"--rounds must be positive, got {args.rounds}")

    for round_idx in range(1, args.rounds + 1):
        seed = args.seed + round_idx - 1
        print(f"=== ROUND {round_idx}/{args.rounds} | seed={seed} ===", flush=True)
        for step_idx, sparse_step in enumerate(step_values, start=1):
            print(
                f"--- Round {round_idx}/{args.rounds} | run {step_idx}/{len(step_values)} | sparse_step={sparse_step} ---",
                flush=True,
            )
            result = train_single_sparse_run(sparse_step, args, seed=seed, round_idx=round_idx)
            rows.append(result)

    out_dir = organized_results_dir() / "baseline_tune" / "sparse_temp_input_adapter_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sparse_step_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"Comparison summary saved: {out_dir / 'sparse_step_comparison.csv'}")
    print(summary.to_string(index=False))

    numeric_cols = [
        "best_val_mae",
        "test_random_mae",
        "test_fixed_mae",
        "soc_random_mean_mae",
        "soc_fixed_mean_mae",
        "test_mean_mae",
        "soc_mean_mae",
    ]
    grouped = summary.groupby("sparse_step")[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(out_dir / "sparse_step_comparison_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Aggregate summary saved: {out_dir / 'sparse_step_comparison_summary.csv'}")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
