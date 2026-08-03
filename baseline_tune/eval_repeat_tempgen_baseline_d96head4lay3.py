from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import baseline_d96head4lay3 as base


def latest_repeat_pth_dir() -> Path:
    repeat_root = base.organized_results_dir() / "baseline_tune" / "baseline_d96h4l3_repeat"
    candidates = sorted(
        repeat_root.glob("*/pth_save"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No repeat pth_save directory found under {repeat_root}")

    with_checkpoints = [path for path in candidates if list(path.glob("best_model_*.pth"))]
    return with_checkpoints[0] if with_checkpoints else candidates[0]


def parse_repeat_seed(checkpoint: Path) -> tuple[float, float]:
    match = re.search(r"_rep(?P<repeat>\d+)_seed(?P<seed>\d+)", checkpoint.stem)
    if not match:
        return float("nan"), float("nan")
    return float(match.group("repeat")), float(match.group("seed"))


def metric_mean(df: pd.DataFrame, temp_range: str, split: str) -> float:
    rows = df[
        df["metric_level"].eq("total")
        & df["temp_range"].eq(temp_range)
        & df["split"].eq(split)
    ]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows["avg_mae"], errors="coerce").mean())


def build_eval_config(ckpt_config: dict, pth_dir: Path, output_dir: Path, soc_batch_size: int, starts: list[float]):
    config = base.make_config()
    config.update(ckpt_config)
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["PTH_DIR"] = str(pth_dir)
    config["CACHE_DIR"] = str(pth_dir.parent / "pt")
    config["CSV_DIR"] = str(output_dir)
    config["META_DIR"] = str(pth_dir.parent / "metadata")
    config["SOC_BATCH_SIZE"] = soc_batch_size
    config["SOC_STARTS"] = starts
    return config


def write_checkpoint_summary(rows: list[dict], output_dir: Path, run_name: str):
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / f"repeat_tempgen_checkpoint_summary_{run_name}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    metric_cols = [
        col
        for col in summary_df.columns
        if col.endswith("_mean_mae") or col.endswith("_overall_mae")
    ]
    aggregate_rows = []
    for metric in metric_cols:
        values = pd.to_numeric(summary_df[metric], errors="coerce").dropna()
        aggregate_rows.append(
            {
                "metric": metric,
                "checkpoints": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else float("nan"),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else float("nan"),
                "min": float(values.min()) if not values.empty else float("nan"),
                "max": float(values.max()) if not values.empty else float("nan"),
                "median": float(values.median()) if not values.empty else float("nan"),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = output_dir / f"repeat_tempgen_metric_aggregate_{run_name}.csv"
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    return summary_path, aggregate_path, summary_df, aggregate_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all repeated baseline checkpoints on temperature-generalization multi-SOC tests."
    )
    parser.add_argument("--pth-dir", default=None, help="Directory containing best_model_*.pth. Default: latest repeat pth_save.")
    parser.add_argument("--output-dir", default=None, help="Directory for tempgen evaluation CSV files.")
    parser.add_argument("--temp-ranges", default="15-20,30-35")
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pth_dir = Path(args.pth_dir).resolve() if args.pth_dir else latest_repeat_pth_dir().resolve()
    checkpoints = sorted(pth_dir.glob("best_model_*.pth"), key=lambda path: path.name)
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints and not args.dry_run:
        raise FileNotFoundError(f"No best_model_*.pth checkpoint found in {pth_dir}")

    run_name = f"repeat_tempgen_{datetime.now().strftime('%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (pth_dir.parent / "csv_save" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_ranges = base.parse_temp_ranges(args.temp_ranges)
    starts = base.parse_float_list(args.starts)

    plan_rows = []
    for order, checkpoint in enumerate(checkpoints, start=1):
        repeat_index, seed = parse_repeat_seed(checkpoint)
        plan_rows.append(
            {
                "checkpoint_order": order,
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                "repeat": repeat_index,
                "seed": seed,
            }
        )
    plan_columns = ["checkpoint_order", "checkpoint", "checkpoint_name", "repeat", "seed"]
    plan_df = pd.DataFrame(plan_rows, columns=plan_columns)
    plan_path = output_dir / f"repeat_tempgen_checkpoint_plan_{run_name}.csv"
    plan_df.to_csv(plan_path, index=False, encoding="utf-8-sig")

    print("=== Repeat temperature-generalization checkpoint plan ===")
    print(f"PTH_DIR     : {pth_dir}")
    print(f"OUTPUT_DIR  : {output_dir}")
    print(f"CHECKPOINTS : {len(checkpoints)}")
    print(f"TEMP_RANGES : {temp_ranges}")
    print(f"SOC_STARTS  : {starts}")
    print(plan_df[["checkpoint_order", "checkpoint_name", "repeat", "seed"]].to_string(index=False))
    print(f"Plan saved: {plan_path}")

    if args.dry_run:
        if not checkpoints:
            print(f"WARNING: no best_model_*.pth checkpoint found in {pth_dir}")
        print("Dry run only. No evaluation executed.")
        return

    detailed_frames = []
    summary_rows = []

    for checkpoint_order, checkpoint in enumerate(checkpoints, start=1):
        repeat_index, seed = parse_repeat_seed(checkpoint)
        print("\n" + "=" * 88)
        print(f"Evaluating checkpoint {checkpoint_order}/{len(checkpoints)}")
        print(f"CHECKPOINT : {checkpoint}")
        ckpt = base.load_torch_file(checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
        ckpt_config = ckpt.get("config", {})
        config = build_eval_config(ckpt_config, pth_dir, output_dir, args.soc_batch_size, starts)
        base.print_config(config)

        scaler = base.PITDScaler()
        scaler.stats = ckpt["scaler_stats"]
        model = base.build_model(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        checkpoint_frames = []
        checkpoint_summary = {
            "checkpoint_order": checkpoint_order,
            "checkpoint_name": checkpoint.name,
            "checkpoint": str(checkpoint),
            "repeat": repeat_index,
            "seed": seed,
        }

        for range_order, (range_label, low_c, high_c) in enumerate(temp_ranges, start=1):
            datasets = base.load_temperature_generalization_test_datasets(config, range_label, low_c, high_c)
            repeat_label = int(repeat_index) if np.isfinite(repeat_index) else checkpoint_order
            seed_label = int(seed) if np.isfinite(seed) else 0
            run_id = f"{run_name}_ckpt{checkpoint_order:02d}_rep{repeat_label:02d}_seed{seed_label}_{range_label}"
            metrics_df = base.evaluate_soc_start_splits(model, scaler, datasets, config, run_id)
            metrics_df = metrics_df.copy()
            metrics_df.insert(0, "checkpoint_order", checkpoint_order)
            metrics_df.insert(1, "checkpoint_name", checkpoint.name)
            metrics_df.insert(2, "checkpoint", str(checkpoint))
            metrics_df.insert(3, "repeat", repeat_index)
            metrics_df.insert(4, "seed", seed)
            metrics_df.insert(5, "temp_range_order", range_order)
            metrics_df.insert(6, "temp_range", range_label)
            metrics_df.insert(7, "temp_low_C", low_c)
            metrics_df.insert(8, "temp_high_C", high_c)
            checkpoint_frames.append(metrics_df)
            detailed_frames.append(metrics_df)

        checkpoint_df = pd.concat(checkpoint_frames, ignore_index=True)
        for range_label, _, _ in temp_ranges:
            random_mean = metric_mean(checkpoint_df, range_label, "test_random")
            fixed_mean = metric_mean(checkpoint_df, range_label, "test_fixed")
            checkpoint_summary[f"{range_label}_random_mean_mae"] = random_mean
            checkpoint_summary[f"{range_label}_fixed_mean_mae"] = fixed_mean
            checkpoint_summary[f"{range_label}_overall_mae"] = float(np.nanmean([random_mean, fixed_mean]))
        all_total_rows = checkpoint_df[checkpoint_df["metric_level"].eq("total")]
        checkpoint_summary["all_tempgen_total_mean_mae"] = float(
            pd.to_numeric(all_total_rows["avg_mae"], errors="coerce").mean()
        )
        summary_rows.append(checkpoint_summary)
        summary_path, aggregate_path, _, aggregate_df = write_checkpoint_summary(summary_rows, output_dir, run_name)

        combined_so_far = pd.concat(detailed_frames, ignore_index=True)
        combined_path = output_dir / f"repeat_tempgen_soc_start_metrics_{run_name}.csv"
        combined_total_path = output_dir / f"repeat_tempgen_soc_start_total_summary_{run_name}.csv"
        combined_so_far.to_csv(combined_path, index=False, encoding="utf-8-sig")
        combined_so_far[combined_so_far["metric_level"].eq("total")].to_csv(
            combined_total_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Checkpoint summary updated: {summary_path}")
        print(f"Metric aggregate updated: {aggregate_path}")
        print("\n=== Current aggregate ===")
        print(aggregate_df.to_string(index=False))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Finished repeat temperature-generalization evaluation ===")
    print(f"Detailed metrics: {output_dir / f'repeat_tempgen_soc_start_metrics_{run_name}.csv'}")
    print(f"Total summary   : {output_dir / f'repeat_tempgen_soc_start_total_summary_{run_name}.csv'}")
    print(f"Checkpoint table: {output_dir / f'repeat_tempgen_checkpoint_summary_{run_name}.csv'}")
    print(f"Aggregate table : {output_dir / f'repeat_tempgen_metric_aggregate_{run_name}.csv'}")


if __name__ == "__main__":
    main()
