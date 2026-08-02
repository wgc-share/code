from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import train_causal_adaln_dropout_filtered_temps_dmodel96 as base


PROJECT = "PINT_SchemeB_CAUSAL_ADALN_DROPOUT_FILTERED_TEMPS_CAPACITY_SWEEP"


CAPACITY_TRIALS = [
    {"trial_index": 1, "d_model": 64, "nhead": 4, "num_layers": 2, "tag": "baseline_d64_h4_l2"},
    {"trial_index": 2, "d_model": 48, "nhead": 4, "num_layers": 2, "tag": "d48_h4_l2"},
    {"trial_index": 3, "d_model": 80, "nhead": 4, "num_layers": 2, "tag": "d80_h4_l2"},
    {"trial_index": 4, "d_model": 96, "nhead": 4, "num_layers": 2, "tag": "d96_h4_l2"},
    {"trial_index": 5, "d_model": 112, "nhead": 4, "num_layers": 2, "tag": "d112_h4_l2"},
    {"trial_index": 6, "d_model": 128, "nhead": 4, "num_layers": 2, "tag": "d128_h4_l2"},
    {"trial_index": 7, "d_model": 160, "nhead": 4, "num_layers": 2, "tag": "d160_h4_l2"},
    {"trial_index": 8, "d_model": 64, "nhead": 2, "num_layers": 2, "tag": "d64_h2_l2"},
    {"trial_index": 9, "d_model": 64, "nhead": 8, "num_layers": 2, "tag": "d64_h8_l2"},
    {"trial_index": 10, "d_model": 96, "nhead": 2, "num_layers": 2, "tag": "d96_h2_l2"},
    {"trial_index": 11, "d_model": 96, "nhead": 8, "num_layers": 2, "tag": "d96_h8_l2"},
    {"trial_index": 12, "d_model": 128, "nhead": 2, "num_layers": 2, "tag": "d128_h2_l2"},
    {"trial_index": 13, "d_model": 128, "nhead": 8, "num_layers": 2, "tag": "d128_h8_l2"},
    {"trial_index": 14, "d_model": 64, "nhead": 4, "num_layers": 1, "tag": "d64_h4_l1"},
    {"trial_index": 15, "d_model": 64, "nhead": 4, "num_layers": 3, "tag": "d64_h4_l3"},
    {"trial_index": 16, "d_model": 64, "nhead": 4, "num_layers": 4, "tag": "d64_h4_l4"},
    {"trial_index": 17, "d_model": 96, "nhead": 4, "num_layers": 1, "tag": "d96_h4_l1"},
    {"trial_index": 18, "d_model": 96, "nhead": 4, "num_layers": 3, "tag": "d96_h4_l3"},
    {"trial_index": 19, "d_model": 96, "nhead": 4, "num_layers": 4, "tag": "d96_h4_l4"},
    {"trial_index": 20, "d_model": 128, "nhead": 4, "num_layers": 1, "tag": "d128_h4_l1"},
    {"trial_index": 21, "d_model": 128, "nhead": 4, "num_layers": 3, "tag": "d128_h4_l3"},
    {"trial_index": 22, "d_model": 128, "nhead": 4, "num_layers": 4, "tag": "d128_h4_l4"},
    {"trial_index": 23, "d_model": 80, "nhead": 8, "num_layers": 2, "tag": "d80_h8_l2"},
    {"trial_index": 24, "d_model": 112, "nhead": 8, "num_layers": 2, "tag": "d112_h8_l2"},
    {"trial_index": 25, "d_model": 160, "nhead": 8, "num_layers": 2, "tag": "d160_h8_l2"},
    {"trial_index": 26, "d_model": 80, "nhead": 4, "num_layers": 3, "tag": "d80_h4_l3"},
    {"trial_index": 27, "d_model": 112, "nhead": 4, "num_layers": 3, "tag": "d112_h4_l3"},
    {"trial_index": 28, "d_model": 160, "nhead": 4, "num_layers": 3, "tag": "d160_h4_l3"},
    {"trial_index": 29, "d_model": 96, "nhead": 8, "num_layers": 3, "tag": "d96_h8_l3"},
    {"trial_index": 30, "d_model": 128, "nhead": 8, "num_layers": 3, "tag": "d128_h8_l3"},
]


def make_run_dirs(run_name: str) -> dict[str, Path]:
    run_root = base.organized_results_dir() / "baseline_tune" / "capacity_sweep" / run_name
    dirs = {
        "run_root": run_root,
        "cache_dir": run_root / "pt",
        "pth_dir": run_root / "pth_save",
        "csv_dir": run_root / "csv_save",
        "meta_dir": run_root / "metadata",
        "log_dir": run_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def count_parameters(trial: dict) -> int:
    config = base.make_config()
    config["DEVICE"] = "cpu"
    config["D_MODEL"] = trial["d_model"]
    config["NHEAD"] = trial["nhead"]
    config["NUM_LAYERS"] = trial["num_layers"]
    model = base.build_model(config)
    return int(sum(p.numel() for p in model.parameters()))


def trial_table(trials: list[dict]) -> pd.DataFrame:
    rows = []
    for trial in trials:
        if trial["d_model"] % trial["nhead"] != 0:
            raise ValueError(f"Invalid trial {trial['trial_index']}: d_model must be divisible by nhead.")
        rows.append(
            {
                "trial_index": trial["trial_index"],
                "tag": trial["tag"],
                "d_model": trial["d_model"],
                "nhead": trial["nhead"],
                "head_dim": trial["d_model"] // trial["nhead"],
                "num_layers": trial["num_layers"],
                "params": count_parameters(trial),
            }
        )
    return pd.DataFrame(rows)


def parse_trial_selection(value: str | None, max_trials: int | None) -> list[dict]:
    trials = CAPACITY_TRIALS
    if value:
        wanted = {int(item.strip()) for item in value.split(",") if item.strip()}
        trials = [trial for trial in trials if trial["trial_index"] in wanted]
    if max_trials is not None:
        trials = trials[:max_trials]
    if not trials:
        raise ValueError("No trials selected.")
    return trials


def make_trial_config(base_config: dict, run_dirs: dict[str, Path], trial: dict, args: argparse.Namespace) -> dict:
    config = dict(base_config)
    config.update(
        {
            "PROJECT": PROJECT,
            "CACHE_DIR": str(run_dirs["cache_dir"]),
            "PTH_DIR": str(run_dirs["pth_dir"]),
            "CSV_DIR": str(run_dirs["csv_dir"]),
            "META_DIR": str(run_dirs["meta_dir"]),
            "D_MODEL": trial["d_model"],
            "NHEAD": trial["nhead"],
            "NUM_LAYERS": trial["num_layers"],
            "DROPOUT": args.dropout,
            "P_RESET": args.p_reset,
            "LR": args.lr,
            "EPOCHS": args.epochs,
            "BATCH_SIZE": args.batch_size,
            "VAL_BATCH_SIZE": args.val_batch_size,
            "SOC_BATCH_SIZE": args.soc_batch_size,
            "DEVICE": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
            "TRIAL_INDEX": trial["trial_index"],
            "TRIAL_TAG": trial["tag"],
        }
    )
    return config


def write_progress(progress_rows: list[dict], progress_path: Path):
    pd.DataFrame(progress_rows).to_csv(progress_path, index=False, encoding="utf-8-sig")


def total_metric(metrics_df: pd.DataFrame, split: str) -> float:
    rows = metrics_df[(metrics_df["metric_level"].eq("total")) & (metrics_df["split"].eq(split))]
    if rows.empty:
        return float("nan")
    return float(rows["avg_mae"].mean())


def run_trial(
    trial: dict,
    base_config: dict,
    datasets: tuple,
    scaler_stats: dict,
    run_dirs: dict[str, Path],
    run_name: str,
    args: argparse.Namespace,
    progress_rows: list[dict],
    final_rows: list[dict],
    progress_path: Path,
    final_summary_path: Path,
) -> dict:
    config = make_trial_config(base_config, run_dirs, trial, args)
    trial_id = f"{run_name}_trial{trial['trial_index']:02d}_{trial['tag']}"
    best_model_path = Path(config["PTH_DIR"]) / f"best_model_{trial_id}.pth"
    trial_log_path = Path(config["CSV_DIR"]) / f"log_{trial_id}.csv"
    val_matrix_path = Path(config["CSV_DIR"]) / f"val_matrix_{trial_id}.csv"

    print("\n" + "=" * 88)
    print(f"Starting trial {trial['trial_index']:02d}: {trial['tag']}")
    base.print_config(config)
    print(f"TRIAL_ID   : {trial_id}")
    print(f"PARAMS     : {count_parameters(trial):,}")

    train_ds, val_ds, _, _ = datasets
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    scaler = base.PITDScaler()
    scaler.stats = scaler_stats

    base.set_seed(args.seed)
    model = base.build_model(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=args.weight_decay)
    criterion = base.PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])

    best_val_mae = float("inf")
    history = []
    val_mae_matrix = base.defaultdict(list)

    for epoch in range(config["EPOCHS"]):
        model.train()
        manager = base.BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
        h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
        pbar = tqdm(
            total=manager.total_steps,
            desc=f"Trial {trial['trial_index']:02d} Epoch {epoch + 1}/{config['EPOCHS']}",
            colour="blue",
        )

        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]
        print(
            f"[Trial {trial['trial_index']:02d} Epoch {epoch + 1}] start | "
            f"d_model={config['D_MODEL']} | nhead={config['NHEAD']} | layers={config['NUM_LAYERS']} | "
            f"lambda_ah={curr_lambda:.2f} | p_reset={config['P_RESET']:.3f}"
        )

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_reset_lanes = 0
        epoch_reset_batches = 0

        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [train_ds[i] for i in indices]
            x = torch.stack([s["x_dyn"] for s in samples]).to(config["DEVICE"])
            t = torch.stack([s["t_mean"] for s in samples]).to(config["DEVICE"])
            y = torch.stack([s["soc"] for s in samples]).to(config["DEVICE"])
            q = torch.stack([s["Q"] for s in samples]).to(config["DEVICE"])
            m_t = torch.tensor(masks, dtype=torch.float32, device=config["DEVICE"])
            f_t = torch.tensor(is_first, dtype=torch.bool, device=config["DEVICE"])

            h_state = h_state.detach()
            if f_t.any():
                for lane_idx in range(config["BATCH_SIZE"]):
                    if f_t[lane_idx]:
                        h_state[:, lane_idx, :] = 0.0

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
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Reset": reset_count})

        pbar.close()

        avg_val_mae, file_maes = base.evaluate_dataset(
            model,
            val_loader,
            scaler,
            {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]},
            label=f"Trial {trial['trial_index']:02d} Epoch {epoch + 1} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        train_total = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        train_data = float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan")
        train_ah = float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan")
        train_range = float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan")

        history.append(
            {
                "trial_index": trial["trial_index"],
                "trial_tag": trial["tag"],
                "epoch": epoch + 1,
                "d_model": config["D_MODEL"],
                "nhead": config["NHEAD"],
                "num_layers": config["NUM_LAYERS"],
                "lr": config["LR"],
                "dropout": config["DROPOUT"],
                "p_reset": config["P_RESET"],
                "train_total": train_total,
                "train_data": train_data,
                "train_ah": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
                "train_range": train_range,
                "val_mae": avg_val_mae,
                "best_val_mae_so_far": min(best_val_mae, avg_val_mae),
                "lambda_ah": curr_lambda,
                "reset_lanes": epoch_reset_lanes,
                "reset_batches": epoch_reset_batches,
            }
        )
        pd.DataFrame(history).to_csv(trial_log_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(val_mae_matrix).T.to_csv(val_matrix_path, encoding="utf-8-sig")

        progress_rows.append(
            {
                "run_name": run_name,
                "trial_index": trial["trial_index"],
                "trial_tag": trial["tag"],
                "status": "running",
                "epoch": epoch + 1,
                "epochs": config["EPOCHS"],
                "d_model": config["D_MODEL"],
                "nhead": config["NHEAD"],
                "num_layers": config["NUM_LAYERS"],
                "params": count_parameters(trial),
                "val_mae": avg_val_mae,
                "best_val_mae_so_far": min(best_val_mae, avg_val_mae),
                "log_path": str(trial_log_path),
            }
        )
        write_progress(progress_rows, progress_path)

        print(
            f"[Trial {trial['trial_index']:02d} Epoch {epoch + 1}] summary | "
            f"train_total={train_total:.6f} | train_data={train_data:.6f} | "
            f"train_ah={float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float('nan'):.6f} | "
            f"train_range={train_range:.6f} | val_mae={avg_val_mae:.6f} | "
            f"best_val={min(best_val_mae, avg_val_mae):.6f}"
        )

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            base.save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                    "trial": trial,
                    "trial_id": trial_id,
                },
                best_model_path,
            )
            print(f"[Trial {trial['trial_index']:02d}] new best checkpoint saved: {best_model_path}")

    if not best_model_path.exists():
        raise RuntimeError(f"Trial {trial['trial_index']} finished without a checkpoint.")

    best_ckpt = base.load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"[Trial {trial['trial_index']:02d}] loaded best checkpoint for final test and multi-SOC evaluation.")

    test_df = base.evaluate_test_splits(model, scaler, datasets, config, trial_id)
    soc_df = base.evaluate_soc_start_splits(model, scaler, datasets, config, trial_id)

    final_row = {
        "run_name": run_name,
        "trial_index": trial["trial_index"],
        "trial_tag": trial["tag"],
        "status": "complete",
        "d_model": config["D_MODEL"],
        "nhead": config["NHEAD"],
        "head_dim": config["D_MODEL"] // config["NHEAD"],
        "num_layers": config["NUM_LAYERS"],
        "params": count_parameters(trial),
        "epochs": config["EPOCHS"],
        "best_val_mae": best_val_mae,
        "test_random_mae": total_metric(test_df, "test_random"),
        "test_fixed_mae": total_metric(test_df, "test_fixed"),
        "soc_random_mean_mae": total_metric(soc_df, "test_random"),
        "soc_fixed_mean_mae": total_metric(soc_df, "test_fixed"),
        "checkpoint": str(best_model_path),
        "trial_log": str(trial_log_path),
    }
    final_row["test_mean_mae"] = float(np.nanmean([final_row["test_random_mae"], final_row["test_fixed_mae"]]))
    final_row["soc_mean_mae"] = float(np.nanmean([final_row["soc_random_mean_mae"], final_row["soc_fixed_mean_mae"]]))
    final_rows.append(final_row)
    pd.DataFrame(final_rows).to_csv(final_summary_path, index=False, encoding="utf-8-sig")
    print(f"[Trial {trial['trial_index']:02d}] final summary updated: {final_summary_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_row


def main():
    parser = argparse.ArgumentParser(description="Capacity sweep for filtered-temperature causal AdaLN dropout baseline.")
    parser.add_argument("--run-name", default=None, help="Result run folder name. Default: capacity_0802_203000 style timestamp.")
    parser.add_argument("--trials", default=None, help="Comma-separated trial indexes, for example: 1,4,6.")
    parser.add_argument("--max-trials", type=int, default=None, help="Run only the first N selected trials.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--p-reset", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    parser.add_argument("--dry-run", action="store_true", help="Only write and print the trial table.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the sweep if one trial fails.")
    args = parser.parse_args()

    selected_trials = parse_trial_selection(args.trials, args.max_trials)
    run_name = args.run_name or f"capacity_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dirs = make_run_dirs(run_name)

    table = trial_table(selected_trials)
    trial_table_path = run_dirs["meta_dir"] / f"capacity_trial_table_{run_name}.csv"
    table.to_csv(trial_table_path, index=False, encoding="utf-8-sig")

    print("=== Capacity sweep trial table ===")
    print(table.to_string(index=False))
    print(f"Trial table saved: {trial_table_path}")

    if args.dry_run:
        print("Dry run only. No training or evaluation executed.")
        return

    base_config = base.make_config()
    base_config.update(
        {
            "PROJECT": PROJECT,
            "CACHE_DIR": str(run_dirs["cache_dir"]),
            "PTH_DIR": str(run_dirs["pth_dir"]),
            "CSV_DIR": str(run_dirs["csv_dir"]),
            "META_DIR": str(run_dirs["meta_dir"]),
            "EPOCHS": args.epochs,
            "BATCH_SIZE": args.batch_size,
            "VAL_BATCH_SIZE": args.val_batch_size,
            "SOC_BATCH_SIZE": args.soc_batch_size,
            "LR": args.lr,
            "DROPOUT": args.dropout,
            "P_RESET": args.p_reset,
            "DEVICE": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        }
    )

    base.print_config(base_config)
    datasets = base.load_all_datasets(base_config)
    train_ds, _, _, _ = datasets
    scaler = base.PITDScaler()
    scaler.fit(train_ds)
    scaler_stats = scaler.stats

    progress_rows: list[dict] = []
    final_rows: list[dict] = []
    progress_path = run_dirs["csv_dir"] / f"capacity_epoch_progress_{run_name}.csv"
    final_summary_path = run_dirs["csv_dir"] / f"capacity_final_summary_{run_name}.csv"

    for trial in selected_trials:
        try:
            run_trial(
                trial,
                base_config,
                datasets,
                scaler_stats,
                run_dirs,
                run_name,
                args,
                progress_rows,
                final_rows,
                progress_path,
                final_summary_path,
            )
        except Exception as exc:
            error_path = run_dirs["log_dir"] / f"error_trial{trial['trial_index']:02d}_{trial['tag']}.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            progress_rows.append(
                {
                    "run_name": run_name,
                    "trial_index": trial["trial_index"],
                    "trial_tag": trial["tag"],
                    "status": "failed",
                    "epoch": np.nan,
                    "epochs": args.epochs,
                    "d_model": trial["d_model"],
                    "nhead": trial["nhead"],
                    "num_layers": trial["num_layers"],
                    "params": count_parameters(trial),
                    "val_mae": np.nan,
                    "best_val_mae_so_far": np.nan,
                    "log_path": str(error_path),
                    "error": repr(exc),
                }
            )
            write_progress(progress_rows, progress_path)
            print(f"[Trial {trial['trial_index']:02d}] failed. Error saved: {error_path}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.stop_on_error:
                raise

    if final_rows:
        final_df = pd.DataFrame(final_rows).sort_values("soc_mean_mae", na_position="last")
        ranked_path = run_dirs["csv_dir"] / f"capacity_final_summary_ranked_{run_name}.csv"
        final_df.to_csv(ranked_path, index=False, encoding="utf-8-sig")
        print("\n=== Ranked capacity summary by multi-SOC mean MAE ===")
        print(
            final_df[
                [
                    "trial_index",
                    "trial_tag",
                    "d_model",
                    "nhead",
                    "num_layers",
                    "params",
                    "best_val_mae",
                    "test_mean_mae",
                    "soc_mean_mae",
                ]
            ].to_string(index=False)
        )
        print(f"Ranked summary saved: {ranked_path}")


if __name__ == "__main__":
    main()
