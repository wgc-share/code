from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
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

shared_path = str(SHARED_DIR)
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from project_paths import baseline_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, split_soccc_by_cells
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_DROPOUT_HPARAM"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def cache_name(split_name: str, window_size: int, stride: int) -> str:
    base = f"{split_name}_cache_causal_adaln_dropout"
    if window_size == 100 and stride == 100:
        return f"{base}.pt"
    return f"{base}_w{window_size}_s{stride}.pt"


def make_results_dirs(search_id: str):
    base = baseline_results_dir()
    cache_dir = base / "pt"
    run_dir = base / "hparam_search" / search_id
    pth_dir = run_dir / "pth_save"
    csv_dir = run_dir / "csv_save"
    meta_dir = run_dir / "metadata"
    for directory in (cache_dir, pth_dir, csv_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def build_trials(args: argparse.Namespace) -> list[dict]:
    grid = {
        "LR": parse_float_list(args.lrs),
        "P_RESET": parse_float_list(args.p_resets),
        "DROPOUT": parse_float_list(args.dropouts),
        "D_MODEL": parse_int_list(args.d_models),
        "NHEAD": parse_int_list(args.nheads),
        "NUM_LAYERS": parse_int_list(args.num_layers),
        "BATCH_SIZE": parse_int_list(args.batch_sizes),
        "LAMBDA_AH_START": parse_float_list(args.lambda_ah_starts),
        "LAMBDA_AH_STEP": parse_float_list(args.lambda_ah_steps),
        "WEIGHT_DECAY": parse_float_list(args.weight_decays),
        "GRAD_CLIP": parse_float_list(args.grad_clips),
    }
    keys = list(grid.keys())
    trials = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]

    rng = random.Random(args.seed)
    if args.search == "random":
        rng.shuffle(trials)
        trials = trials[: args.max_trials]
    elif args.max_trials is not None:
        trials = trials[: args.max_trials]

    return trials


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


def train_one_trial(trial_config: dict, datasets: tuple, scaler: PITDScaler, csv_dir: Path, pth_dir: Path, trial_id: str):
    train_ds, val_ds, test_random_ds, test_fixed_ds = datasets
    set_seed(trial_config["SEED"])

    val_loader = DataLoader(val_ds, batch_size=trial_config["VAL_BATCH_SIZE"], shuffle=False)
    test_random_loader = DataLoader(test_random_ds, batch_size=trial_config["TEST_BATCH_SIZE"], shuffle=False)
    test_fixed_loader = DataLoader(test_fixed_ds, batch_size=trial_config["TEST_BATCH_SIZE"], shuffle=False)

    model = AdaLNBatteryTDGCMModel(
        d_model=trial_config["D_MODEL"],
        nhead=trial_config["NHEAD"],
        num_layers=trial_config["NUM_LAYERS"],
        dropout=trial_config["DROPOUT"],
        use_causal=True,
    ).to(trial_config["DEVICE"])

    optimizer = optim.AdamW(model.parameters(), lr=trial_config["LR"], weight_decay=trial_config["WEIGHT_DECAY"])
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial_config["EPOCHS"])

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_path = pth_dir / f"best_model_{trial_id}.pth"
    history = []
    val_mae_matrix = defaultdict(list)

    print(f"\n=== Trial {trial_id} ===")
    print(json.dumps({k: trial_config[k] for k in trial_config["TUNED_KEYS"]}, indent=2))

    for epoch in range(trial_config["EPOCHS"]):
        model.train()
        manager = BalancedSchemeBManager(train_ds, trial_config["BATCH_SIZE"])
        h_state = torch.zeros(1, trial_config["BATCH_SIZE"], trial_config["D_MODEL"], device=trial_config["DEVICE"])
        curr_lambda = trial_config["LAMBDA_AH_START"] + epoch * trial_config["LAMBDA_AH_STEP"]

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_reset_lanes = 0
        epoch_reset_batches = 0

        pbar = tqdm(
            total=manager.total_steps,
            desc=f"{trial_id} epoch {epoch + 1}/{trial_config['EPOCHS']}",
            colour="blue",
            disable=trial_config.get("DISABLE_TQDM", False),
        )
        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [train_ds[index] for index in indices]
            x = torch.stack([sample["x_dyn"] for sample in samples]).to(trial_config["DEVICE"])
            t = torch.stack([sample["t_mean"] for sample in samples]).to(trial_config["DEVICE"])
            y = torch.stack([sample["soc"] for sample in samples]).to(trial_config["DEVICE"])
            q = torch.stack([sample["Q"] for sample in samples]).to(trial_config["DEVICE"])
            m_t = torch.tensor(masks, dtype=torch.float32, device=trial_config["DEVICE"])
            f_t = torch.tensor(is_first, dtype=torch.bool, device=trial_config["DEVICE"])

            h_state = h_state.detach()
            if f_t.any():
                h_state[:, f_t, :] = 0.0

            active_mask = m_t > 0.5
            eligible_mask = active_mask & (~f_t)
            reset_mask = (torch.rand(trial_config["BATCH_SIZE"], device=trial_config["DEVICE"]) < trial_config["P_RESET"]) & eligible_mask
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), trial_config["GRAD_CLIP"])
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
                "DEVICE": trial_config["DEVICE"],
                "D_MODEL": trial_config["D_MODEL"],
                "DISABLE_TQDM": trial_config.get("DISABLE_TQDM", False),
            },
            label=f"{trial_id} epoch {epoch + 1} val",
            output_dir=None,
            run_id=None,
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        row = {
            "trial_id": trial_id,
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
        pd.DataFrame(history).to_csv(csv_dir / f"log_{trial_id}.csv", index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(csv_dir / f"val_matrix_{trial_id}.csv")

        print(
            f"[{trial_id}] epoch={epoch + 1} "
            f"train_loss={row['train_loss']:.6f} val_mae={val_mae:.6f} lambda_ah={curr_lambda:.2f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": trial_config,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                },
                best_model_path,
            )
            print(f"[{trial_id}] new best checkpoint: epoch={best_epoch} val_mae={best_val_mae:.6f}")

    ckpt = load_torch_file(best_model_path, map_location=trial_config["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_random_mae, _ = evaluate_dataset(
        model,
        test_random_loader,
        scaler,
        {
            "DEVICE": trial_config["DEVICE"],
            "D_MODEL": trial_config["D_MODEL"],
            "DISABLE_TQDM": trial_config.get("DISABLE_TQDM", False),
        },
        label=f"{trial_id} test random",
        output_dir=csv_dir,
        run_id=trial_id,
    )
    test_fixed_mae, _ = evaluate_dataset(
        model,
        test_fixed_loader,
        scaler,
        {
            "DEVICE": trial_config["DEVICE"],
            "D_MODEL": trial_config["D_MODEL"],
            "DISABLE_TQDM": trial_config.get("DISABLE_TQDM", False),
        },
        label=f"{trial_id} test fixed",
        output_dir=csv_dir,
        run_id=trial_id,
    )

    return {
        "trial_id": trial_id,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_random_mae": test_random_mae,
        "test_fixed_mae": test_fixed_mae,
        "checkpoint": str(best_model_path),
        **{key: trial_config[key] for key in trial_config["TUNED_KEYS"]},
    }


def failed_trial_result(trial_config: dict, exc: BaseException) -> dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "trial_id": trial_config["TRIAL_ID"],
        "best_epoch": math.nan,
        "best_val_mae": math.nan,
        "test_random_mae": math.nan,
        "test_fixed_mae": math.nan,
        "checkpoint": "",
        "error_type": type(exc).__name__,
        "error": str(exc),
        **{key: trial_config[key] for key in trial_config["TUNED_KEYS"]},
    }


def run_trial_worker(trial_config: dict) -> dict:
    try:
        datasets = load_datasets(trial_config)
        scaler = PITDScaler()
        scaler.fit(datasets[0])
        return train_one_trial(
            trial_config,
            datasets,
            scaler,
            Path(trial_config["CSV_DIR"]),
            Path(trial_config["PTH_DIR"]),
            trial_config["TRIAL_ID"],
        )
    except BaseException as exc:
        return failed_trial_result(trial_config, exc)


def build_trial_config(base_config: dict, trial: dict, tuned_keys: list[str], args: argparse.Namespace, search_id: str, trial_index: int) -> dict:
    return {
        **base_config,
        **trial,
        "SEED": args.seed + trial_index - 1,
        "TRIAL_INDEX": trial_index,
        "TRIAL_ID": f"{search_id}_trial{trial_index:03d}",
        "TUNED_KEYS": tuned_keys,
        "DISABLE_TQDM": args.parallel_trials > 1,
    }


def write_summary(summary_rows: list[dict], summary_path: Path):
    summary_df = pd.DataFrame(summary_rows)
    if "best_val_mae" in summary_df.columns:
        summary_df = summary_df.sort_values("best_val_mae", na_position="last")
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for causal Transformer + AdaLN + hidden-state dropout.")
    parser.add_argument("--search", choices=["grid", "random"], default="grid")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lrs", default="5e-5,1e-4,2e-4")
    parser.add_argument("--p-resets", default="0.0,0.02,0.05,0.10")
    parser.add_argument("--dropouts", default="0.05,0.10,0.15")
    parser.add_argument("--d-models", default="64")
    parser.add_argument("--nheads", default="4")
    parser.add_argument("--num-layers", default="2")
    parser.add_argument("--batch-sizes", default="64")
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--lambda-ah-starts", default="10.0")
    parser.add_argument("--lambda-ah-steps", default="20.0")
    parser.add_argument("--weight-decays", default="1e-5")
    parser.add_argument("--grad-clips", default="1.0")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--parallel-trials", type=int, default=1, help="Number of trial processes to run at the same time on the same machine/GPU.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated trials and exit without loading data or training.")
    args = parser.parse_args()
    if args.parallel_trials < 1:
        raise ValueError("--parallel-trials must be >= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_id = datetime.now().strftime("hparam_%m%d_%H%M%S")
    run_dir = baseline_results_dir() / "hparam_search" / search_id
    cache_dir = baseline_results_dir() / "pt"
    pth_dir = run_dir / "pth_save"
    csv_dir = run_dir / "csv_save"
    meta_dir = run_dir / "metadata"
    trials = build_trials(args)
    if not trials:
        raise ValueError("No trials generated. Check the hyperparameter list arguments.")

    invalid_trials = [trial for trial in trials if trial["D_MODEL"] % trial["NHEAD"] != 0]
    if invalid_trials:
        raise ValueError("Every D_MODEL must be divisible by NHEAD. Invalid trials: " + json.dumps(invalid_trials[:5]))

    base_config = {
        "PROJECT": PROJECT,
        "DATA_DIR": str(processed_segments_dir()),
        "SPLIT_FILE": str(split_file_path()),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "EPOCHS": args.epochs,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "TEST_BATCH_SIZE": args.test_batch_size,
        "DEVICE": device,
        "WINDOW_SIZE": args.window_size,
        "STRIDE": args.stride,
    }

    tuned_keys = sorted(trials[0].keys())

    print("=== Hyperparameter search ===")
    print(f"SEARCH_ID : {search_id}")
    print(f"DEVICE    : {device}")
    print(f"TRIALS    : {len(trials)}")
    print(f"EPOCHS    : {args.epochs}")
    print(f"PARALLEL  : {args.parallel_trials}")
    print(f"CSV_DIR   : {csv_dir}")
    print(f"PTH_DIR   : {pth_dir}")

    if args.dry_run:
        print("Dry run only. Generated trials:")
        for trial_index, trial in enumerate(trials, start=1):
            print(f"trial{trial_index:03d}: {json.dumps(trial, sort_keys=True)}")
        return

    cache_dir, pth_dir, csv_dir, meta_dir = make_results_dirs(search_id)
    base_config.update(
        {
            "CACHE_DIR": str(cache_dir),
            "PTH_DIR": str(pth_dir),
            "CSV_DIR": str(csv_dir),
            "META_DIR": str(meta_dir),
        }
    )
    metadata = {
        "search_id": search_id,
        "search_args": vars(args),
        "base_config": base_config,
        "num_trials": len(trials),
        "tuned_keys": tuned_keys,
    }
    (meta_dir / "search_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_rows = []
    summary_path = csv_dir / f"hparam_summary_{search_id}.csv"

    trial_configs = [
        build_trial_config(base_config, trial, tuned_keys, args, search_id, trial_index)
        for trial_index, trial in enumerate(trials, start=1)
    ]

    if args.parallel_trials == 1:
        datasets = load_datasets(base_config)
        train_ds = datasets[0]
        scaler = PITDScaler()
        scaler.fit(train_ds)

        for trial_config in trial_configs:
            trial_id = trial_config["TRIAL_ID"]
            try:
                result = train_one_trial(trial_config, datasets, scaler, csv_dir, pth_dir, trial_id)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    result = failed_trial_result(trial_config, exc)
                    print(f"[{trial_id}] failed with CUDA/RuntimeError: {exc}")
                else:
                    raise

            summary_rows.append(result)
            write_summary(summary_rows, summary_path)
            print(f"[{trial_id}] summary updated: {summary_path}")
    else:
        print("Prebuilding shared dataset caches before launching parallel trial processes...")
        cache_datasets = load_datasets(base_config)
        cache_scaler = PITDScaler()
        cache_scaler.fit(cache_datasets[0])
        del cache_scaler
        del cache_datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        worker_count = min(args.parallel_trials, len(trial_configs))
        print(f"Launching {worker_count} parallel trial processes.")
        mp_context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_to_trial = {executor.submit(run_trial_worker, trial_config): trial_config for trial_config in trial_configs}
            for future in as_completed(future_to_trial):
                trial_config = future_to_trial[future]
                trial_id = trial_config["TRIAL_ID"]
                try:
                    result = future.result()
                except BaseException as exc:
                    result = failed_trial_result(trial_config, exc)
                    print(f"[{trial_id}] worker crashed: {type(exc).__name__}: {exc}")

                summary_rows.append(result)
                write_summary(summary_rows, summary_path)
                status = "failed" if result.get("error") else "finished"
                print(f"[{trial_id}] {status}; summary updated: {summary_path}")

    final_df = write_summary(summary_rows, summary_path)
    print("\n=== Search finished ===")
    print(final_df.head(10).to_string(index=False))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
