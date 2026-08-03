from __future__ import annotations

import argparse
import os
import sys
import traceback
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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import baseline_d96head4lay3 as base


PROJECT = "PINT_SchemeB_BASELINE_D96_H4_L3_REPEAT"


def make_repeat_dirs(run_name: str) -> dict[str, Path]:
    run_root = base.organized_results_dir() / "baseline_tune" / "baseline_d96h4l3_repeat" / run_name
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


def make_repeat_config(run_dirs: dict[str, Path], args: argparse.Namespace, repeat_index: int, seed: int) -> dict:
    config = base.make_config()
    config.update(
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
            "DEVICE": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
            "REPEAT_INDEX": repeat_index,
            "SEED": seed,
        }
    )
    return config


def metric_value(metrics_df: pd.DataFrame, split: str) -> float:
    rows = metrics_df[(metrics_df["metric_level"].eq("total")) & (metrics_df["split"].eq(split))]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows["avg_mae"], errors="coerce").mean())


def write_aggregate(final_rows: list[dict], aggregate_path: Path):
    df = pd.DataFrame(final_rows)
    metric_cols = [
        "best_val_mae",
        "test_random_mae",
        "test_fixed_mae",
        "test_mean_mae",
        "soc_random_mean_mae",
        "soc_fixed_mean_mae",
        "soc_mean_mae",
    ]
    rows = []
    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "runs": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else float("nan"),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else float("nan"),
                "min": float(values.min()) if not values.empty else float("nan"),
                "max": float(values.max()) if not values.empty else float("nan"),
                "median": float(values.median()) if not values.empty else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(aggregate_path, index=False, encoding="utf-8-sig")


def run_one_repeat(
    repeat_index: int,
    seed: int,
    run_name: str,
    run_dirs: dict[str, Path],
    args: argparse.Namespace,
    datasets: tuple,
    scaler_stats: dict,
    epoch_progress: list[dict],
    final_rows: list[dict],
    progress_path: Path,
    final_path: Path,
    aggregate_path: Path,
) -> dict:
    config = make_repeat_config(run_dirs, args, repeat_index, seed)
    repeat_id = f"{run_name}_rep{repeat_index:02d}_seed{seed}"
    run_id = f"{config['PROJECT']}_{repeat_id}"
    best_model_path = Path(config["PTH_DIR"]) / f"best_model_{run_id}.pth"
    log_path = Path(config["CSV_DIR"]) / f"log_{run_id}.csv"
    val_matrix_path = Path(config["CSV_DIR"]) / f"val_matrix_{run_id}.csv"

    print("\n" + "=" * 88)
    print(f"Repeat {repeat_index}/{args.runs} | seed={seed}")
    base.print_config(config)
    print(f"RUN_ID     : {run_id}")

    base.set_seed(seed)
    train_ds, val_ds, _, _ = datasets
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)

    scaler = base.PITDScaler()
    scaler.stats = scaler_stats
    model = base.build_model(config)

    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-5)
    criterion = base.PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])

    best_val_mae = float("inf")
    best_epoch = -1
    history = []
    val_mae_matrix = defaultdict(list)

    for epoch in range(config["EPOCHS"]):
        model.train()
        manager = base.BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
        h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
        pbar = tqdm(
            total=manager.total_steps,
            desc=f"Repeat {repeat_index:02d} Epoch {epoch + 1}/{config['EPOCHS']}",
            colour="blue",
        )

        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[Repeat {repeat_index:02d} Epoch {epoch + 1}] start | seed={seed} | "
            f"lr={current_lr:.8f} | lambda_ah={curr_lambda:.2f} | "
            f"train_windows={len(train_ds)} | val_windows={len(val_ds)} | steps={manager.total_steps}"
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
            label=f"Repeat {repeat_index:02d} Epoch {epoch + 1} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        train_total = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        train_data = float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan")
        train_ah = float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan")
        train_range = float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch + 1
            base.save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "run_id": run_id,
                },
                best_model_path,
            )
            print(f"[Repeat {repeat_index:02d} Epoch {epoch + 1}] new best saved: val_mae={best_val_mae:.6f}")

        history.append(
            {
                "Repeat": repeat_index,
                "Seed": seed,
                "Epoch": epoch + 1,
                "LR": current_lr,
                "Train_Loss": train_total,
                "Train_Data_Loss": train_data,
                "Train_Ah_Loss": train_ah,
                "Train_Range_Loss": train_range,
                "Val_MAE_Avg": avg_val_mae,
                "Best_Val_MAE_So_Far": best_val_mae,
                "Best_Epoch_So_Far": best_epoch,
                "Lambda": curr_lambda,
                "Reset_Lanes": epoch_reset_lanes,
                "Reset_Batches": epoch_reset_batches,
            }
        )
        pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(val_mae_matrix).T.to_csv(val_matrix_path, encoding="utf-8-sig")

        epoch_progress.append(
            {
                "run_name": run_name,
                "repeat": repeat_index,
                "seed": seed,
                "epoch": epoch + 1,
                "epochs": config["EPOCHS"],
                "status": "running",
                "val_mae": avg_val_mae,
                "best_val_mae_so_far": best_val_mae,
                "best_epoch_so_far": best_epoch,
                "lr": current_lr,
                "lambda_ah": curr_lambda,
                "log_path": str(log_path),
            }
        )
        pd.DataFrame(epoch_progress).to_csv(progress_path, index=False, encoding="utf-8-sig")

        print(
            f"[Repeat {repeat_index:02d} Epoch {epoch + 1}] summary | "
            f"train_total={train_total:.6f} | train_data={train_data:.6f} | "
            f"train_ah={train_ah:.6f} | train_range={train_range:.6f} | "
            f"val_mae={avg_val_mae:.6f} | best_val={best_val_mae:.6f} | "
            f"lr={current_lr:.8f} | lambda_ah={curr_lambda:.2f}"
        )

    if not best_model_path.exists():
        raise RuntimeError(f"Repeat {repeat_index} finished without a checkpoint.")

    best_ckpt = base.load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"[Repeat {repeat_index:02d}] loaded best checkpoint for final test and multi-SOC evaluation.")

    test_df = base.evaluate_test_splits(model, scaler, datasets, config, run_id)
    soc_df = base.evaluate_soc_start_splits(model, scaler, datasets, config, run_id)

    row = {
        "run_name": run_name,
        "repeat": repeat_index,
        "seed": seed,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_random_mae": metric_value(test_df, "test_random"),
        "test_fixed_mae": metric_value(test_df, "test_fixed"),
        "soc_random_mean_mae": metric_value(soc_df, "test_random"),
        "soc_fixed_mean_mae": metric_value(soc_df, "test_fixed"),
        "checkpoint": str(best_model_path),
        "log_path": str(log_path),
    }
    row["test_mean_mae"] = float(np.nanmean([row["test_random_mae"], row["test_fixed_mae"]]))
    row["soc_mean_mae"] = float(np.nanmean([row["soc_random_mean_mae"], row["soc_fixed_mean_mae"]]))
    final_rows.append(row)
    pd.DataFrame(final_rows).to_csv(final_path, index=False, encoding="utf-8-sig")
    write_aggregate(final_rows, aggregate_path)

    print(f"[Repeat {repeat_index:02d}] summary updated: {final_path}")
    print(f"[Repeat {repeat_index:02d}] aggregate updated: {aggregate_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description="Repeat baseline d96/head4/layer3 training and summarize metrics.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    run_name = args.run_name or f"repeat20_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dirs = make_repeat_dirs(run_name)

    base_config = make_repeat_config(run_dirs, args, repeat_index=0, seed=args.seed_start)
    base.print_config(base_config)
    print(f"REPEATS    : {args.runs}")
    print(f"SEEDS      : {args.seed_start}..{args.seed_start + args.runs - 1}")
    print(f"RUN_ROOT   : {run_dirs['run_root']}")

    planned = pd.DataFrame(
        [
            {
                "repeat": idx,
                "seed": args.seed_start + idx - 1,
                "d_model": base_config["D_MODEL"],
                "nhead": base_config["NHEAD"],
                "num_layers": base_config["NUM_LAYERS"],
                "epochs": base_config["EPOCHS"],
                "p_reset": base_config["P_RESET"],
                "lr": base_config["LR"],
            }
            for idx in range(1, args.runs + 1)
        ]
    )
    plan_path = run_dirs["meta_dir"] / f"repeat_plan_{run_name}.csv"
    planned.to_csv(plan_path, index=False, encoding="utf-8-sig")
    print("=== Repeat plan ===")
    print(planned.to_string(index=False))
    print(f"Repeat plan saved: {plan_path}")

    if args.dry_run:
        print("Dry run only. No training or evaluation executed.")
        return

    datasets = base.load_all_datasets(base_config)
    train_ds, _, _, _ = datasets
    scaler = base.PITDScaler()
    scaler.fit(train_ds)
    scaler_stats = scaler.stats

    epoch_progress: list[dict] = []
    final_rows: list[dict] = []
    progress_path = run_dirs["csv_dir"] / f"repeat_epoch_progress_{run_name}.csv"
    final_path = run_dirs["csv_dir"] / f"repeat_final_summary_{run_name}.csv"
    aggregate_path = run_dirs["csv_dir"] / f"repeat_metric_aggregate_{run_name}.csv"

    for repeat_index in range(1, args.runs + 1):
        seed = args.seed_start + repeat_index - 1
        try:
            run_one_repeat(
                repeat_index,
                seed,
                run_name,
                run_dirs,
                args,
                datasets,
                scaler_stats,
                epoch_progress,
                final_rows,
                progress_path,
                final_path,
                aggregate_path,
            )
        except Exception as exc:
            error_path = run_dirs["log_dir"] / f"error_repeat{repeat_index:02d}_seed{seed}.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            failure_row = {
                "run_name": run_name,
                "repeat": repeat_index,
                "seed": seed,
                "status": "failed",
                "best_epoch": np.nan,
                "best_val_mae": np.nan,
                "test_random_mae": np.nan,
                "test_fixed_mae": np.nan,
                "test_mean_mae": np.nan,
                "soc_random_mean_mae": np.nan,
                "soc_fixed_mean_mae": np.nan,
                "soc_mean_mae": np.nan,
                "checkpoint": "",
                "log_path": str(error_path),
                "error": repr(exc),
            }
            final_rows.append(failure_row)
            pd.DataFrame(final_rows).to_csv(final_path, index=False, encoding="utf-8-sig")
            write_aggregate(final_rows, aggregate_path)
            print(f"[Repeat {repeat_index:02d}] failed. Error saved: {error_path}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.stop_on_error:
                raise

    if final_rows:
        final_df = pd.DataFrame(final_rows)
        complete_df = final_df[final_df["status"].eq("complete")].copy()
        if not complete_df.empty:
            ranked_path = run_dirs["csv_dir"] / f"repeat_final_summary_ranked_{run_name}.csv"
            complete_df.sort_values("soc_mean_mae", na_position="last").to_csv(
                ranked_path,
                index=False,
                encoding="utf-8-sig",
            )
            print("\n=== Completed repeats ranked by multi-SOC mean MAE ===")
            print(
                complete_df.sort_values("soc_mean_mae")[
                    [
                        "repeat",
                        "seed",
                        "best_epoch",
                        "best_val_mae",
                        "test_mean_mae",
                        "soc_mean_mae",
                    ]
                ].to_string(index=False)
            )
            print(f"Ranked summary saved: {ranked_path}")
        print(f"Aggregate metrics saved: {aggregate_path}")


if __name__ == "__main__":
    main()
