from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE_DIR = Path(__file__).resolve().parent
CODE_ROOT = BASE_DIR.parent
ABLATION_DIR = CODE_ROOT / "ablation"
SHARED_DIR = CODE_ROOT / "shared"

for path in (str(ABLATION_DIR), str(SHARED_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ablation_filtered_temps_runner as filtered_runner
import ablation_temp_extreme_runner as extreme_runner
from eval_soc_start_causal_adaln_dropout import filter_from_soc_start
from scaling import PITDScaler
from soccc_schemeB import parse_segment_filename
from torch_io import load_torch_file
from train_utils import BalancedSchemeBManager


METHOD = "temp_input_adapter_no_ah"
DEFAULT_SOC_STARTS = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]

RANDOM_CONDITIONS = {
    1: "calibration",
    2: "digested",
    3: "highway",
    4: "urban",
}

FIXED_CONDITIONS = {
    ("3-1", 1): "HWFET",
    ("3-1", 2): "US06_HWY",
    ("3-1", 3): "REP05",
    ("3-2", 1): "DST",
    ("3-2", 2): "EUDC",
    ("3-2", 3): "ARTERIAL",
    ("3-3", 1): "LA92",
    ("3-3", 2): "SC03",
    ("3-3", 3): "UDDS",
    ("3-4", 1): "MANHATTAN",
    ("3-4", 2): "NYCC",
    ("3-4", 3): "NUREMBERG",
}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def condition_name(split_name: str, filename: str) -> str:
    meta = parse_segment_filename(filename)
    cell_id = str(meta.get("cell_id", ""))
    segment_index = int(meta.get("segment_index", -1))
    if split_name == "test_random":
        return RANDOM_CONDITIONS.get(segment_index, f"seg{segment_index:02d}")
    if split_name == "test_fixed":
        return FIXED_CONDITIONS.get((cell_id, segment_index), f"unmapped_seg{segment_index:02d}")
    return f"seg{segment_index:02d}"


def latest_checkpoint(pth_dir: Path, project: str) -> Path:
    ckpts = list(pth_dir.glob(f"best_model_{project}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for {project} in {pth_dir}")
    return max(ckpts, key=lambda path: path.stat().st_mtime)


def prepare_filtered_config(args: argparse.Namespace) -> dict:
    fresh = filtered_runner.make_config(METHOD)
    fresh["DEVICE"] = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    fresh["VAL_BATCH_SIZE"] = args.batch_size
    fresh["SOC_BATCH_SIZE"] = args.batch_size
    fresh["DISABLE_TQDM"] = args.disable_tqdm
    return fresh


def prepare_extreme_config(args: argparse.Namespace) -> dict:
    fresh = extreme_runner.make_config(METHOD)
    fresh["DEVICE"] = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    fresh["VAL_BATCH_SIZE"] = args.batch_size
    fresh["SOC_BATCH_SIZE"] = args.batch_size
    fresh["DISABLE_TQDM"] = args.disable_tqdm
    return fresh


def load_model_and_scaler(scope: str, checkpoint_arg: str | None, args: argparse.Namespace):
    if scope == "filtered":
        config = prepare_filtered_config(args)
        ckpt_path = Path(checkpoint_arg) if checkpoint_arg else latest_checkpoint(Path(config["PTH_DIR"]), config["PROJECT"])
        build_model = filtered_runner.build_model
    elif scope == "extreme":
        config = prepare_extreme_config(args)
        ckpt_path = Path(checkpoint_arg) if checkpoint_arg else latest_checkpoint(Path(config["PTH_DIR"]), config["PROJECT"])
        build_model = extreme_runner.build_model
    else:
        raise ValueError(f"Unknown scope: {scope}")

    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    fresh = prepare_filtered_config(args) if scope == "filtered" else prepare_extreme_config(args)
    for key in ("CACHE_DIR", "PTH_DIR", "CSV_DIR", "META_DIR", "DEVICE", "VAL_BATCH_SIZE", "SOC_BATCH_SIZE", "DISABLE_TQDM"):
        config[key] = fresh[key]

    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt_path, config, model, scaler


def predict_dataset_rows(
    model,
    scaler: PITDScaler,
    dataset,
    config: dict,
    eval_scope: str,
    split_name: str,
    soc_start: float | None,
    checkpoint_path: Path,
) -> list[dict]:
    if len(dataset) == 0:
        return []

    batch_size = int(config["SOC_BATCH_SIZE"])
    manager = BalancedSchemeBManager(dataset, batch_size, shuffle_files=False)
    device = config["DEVICE"]
    h_state = torch.zeros(1, batch_size, config["D_MODEL"], device=device)
    file_window_counter = defaultdict(int)
    rows = []

    with torch.no_grad():
        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [dataset[i] for i in indices]
            x = torch.stack([sample["x_dyn"] for sample in samples]).to(device)
            t = torch.stack([sample["t_mean"] for sample in samples]).to(device)
            y = torch.stack([sample["soc"] for sample in samples]).to(device)
            active = torch.tensor(masks, dtype=torch.bool, device=device)
            first = torch.tensor(is_first, dtype=torch.bool, device=device)

            h_state = h_state.detach()
            if first.any():
                h_state[:, first, :] = 0.0

            x_n, t_n = scaler.transform(x, t)
            y_pred, h_state = model(x_n, t_n, h_state)
            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().numpy()

            for lane_idx, sample in enumerate(samples):
                if not bool(active[lane_idx].item()):
                    continue
                filename = sample["filenames"]
                window_index = file_window_counter[filename]
                file_window_counter[filename] += 1
                meta = parse_segment_filename(filename)
                condition = condition_name(split_name, filename)
                seq_len = y_true_np.shape[1]
                for step in range(seq_len):
                    true_soc = float(y_true_np[lane_idx, step, 0])
                    pred_soc = float(y_pred_np[lane_idx, step, 0])
                    rows.append(
                        {
                            "eval_scope": eval_scope,
                            "split": split_name,
                            "soc_start_percent": "" if soc_start is None or math.isnan(soc_start) else soc_start,
                            "checkpoint": str(checkpoint_path),
                            "filename": filename,
                            "cell_id": meta.get("cell_id", ""),
                            "source_idx": meta.get("idx", ""),
                            "segment_index": meta.get("segment_index", ""),
                            "condition": condition,
                            "temp_C": meta.get("temp_C", ""),
                            "window_index": window_index,
                            "step_in_window": step,
                            "sample_index_in_export": window_index * seq_len + step,
                            "soc_true": true_soc,
                            "soc_pred": pred_soc,
                            "abs_error": abs(pred_soc - true_soc),
                            "current_A": float(x_np[lane_idx, step, 0]),
                            "d_current_A": float(x_np[lane_idx, step, 1]),
                            "voltage_V": float(x_np[lane_idx, step, 2]),
                            "d_voltage_V": float(x_np[lane_idx, step, 3]),
                            "power_W": float(x_np[lane_idx, step, 4]),
                            "temperature_input_C": float(t_np[lane_idx, 0]),
                        }
                    )

    return rows


def write_predictions(rows: list[dict], out_dir: Path, run_id: str, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{name}_{run_id}.csv"
    metric_path = out_dir / f"prediction_metrics_{name}_{run_id}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if df.empty:
        metric_df = pd.DataFrame()
    else:
        group_cols = ["eval_scope", "split", "soc_start_percent", "filename", "cell_id", "segment_index", "condition", "temp_C"]
        metric_df = (
            df.groupby(group_cols, dropna=False)["abs_error"]
            .agg(mae="mean", points="count")
            .reset_index()
            .sort_values(group_cols)
        )
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    print(f"Prediction CSV saved: {out_path}", flush=True)
    print(f"Prediction metrics saved: {metric_path}", flush=True)


def export_full_and_soc_predictions(scope: str, ckpt_path: Path, config: dict, model, scaler, datasets, out_dir: Path, run_id: str, starts: list[float]):
    split_datasets = (("test_random", datasets[2]), ("test_fixed", datasets[3]))

    rows = []
    for split_name, dataset in split_datasets:
        rows.extend(
            predict_dataset_rows(
                model, scaler, dataset, config, f"{scope}_full_soc", split_name, None, ckpt_path
            )
        )
    write_predictions(rows, out_dir, run_id, f"{scope}_full_soc")

    rows = []
    for split_name, base_dataset in split_datasets:
        for soc_start in starts:
            filtered_ds = filter_from_soc_start(base_dataset, soc_start)
            rows.extend(
                predict_dataset_rows(
                    model,
                    scaler,
                    filtered_ds,
                    config,
                    f"{scope}_multi_soc",
                    split_name,
                    soc_start,
                    ckpt_path,
                )
            )
    write_predictions(rows, out_dir, run_id, f"{scope}_multi_soc")


def export_filtered(args: argparse.Namespace, run_id: str, starts: list[float]):
    ckpt_path, config, model, scaler = load_model_and_scaler("filtered", args.filtered_checkpoint, args)
    out_dir = Path(config["CSV_DIR"]) / "prediction_exports"
    print("=== Export filtered baseline predictions ===", flush=True)
    print(f"CHECKPOINT : {ckpt_path}", flush=True)
    print(f"OUT_DIR    : {out_dir}", flush=True)

    datasets = filtered_runner.base.load_all_datasets(config)
    export_full_and_soc_predictions("filtered", ckpt_path, config, model, scaler, datasets, out_dir, run_id, starts)

    for range_label, low_c, high_c in filtered_runner.base.TEMP_GENERALIZATION_RANGES:
        rows = []
        temp_datasets = filtered_runner.base.load_temperature_generalization_test_datasets(config, range_label, low_c, high_c)
        for split_name, base_dataset in (("test_random", temp_datasets[2]), ("test_fixed", temp_datasets[3])):
            for soc_start in starts:
                filtered_ds = filter_from_soc_start(base_dataset, soc_start)
                rows.extend(
                    predict_dataset_rows(
                        model,
                        scaler,
                        filtered_ds,
                        config,
                        f"temperature_interpolation_{range_label}",
                        split_name,
                        soc_start,
                        ckpt_path,
                    )
                )
        write_predictions(rows, out_dir, run_id, f"temperature_interpolation_{range_label}")


def export_extreme(args: argparse.Namespace, run_id: str, starts: list[float]):
    ckpt_path, config, model, scaler = load_model_and_scaler("extreme", args.extreme_checkpoint, args)
    out_dir = Path(config["CSV_DIR"]) / "prediction_exports"
    print("=== Export extreme-temperature baseline predictions ===", flush=True)
    print(f"CHECKPOINT : {ckpt_path}", flush=True)
    print(f"OUT_DIR    : {out_dir}", flush=True)

    datasets = extreme_runner.load_extreme_train_datasets(config)
    export_full_and_soc_predictions("extreme", ckpt_path, config, model, scaler, datasets, out_dir, run_id, starts)

    for range_label, side in (("T_LT13", "lt"), ("T_GT37", "gt")):
        rows = []
        temp_datasets = extreme_runner.load_extreme_test_datasets(config, range_label, side)
        for split_name, base_dataset in (("test_random", temp_datasets[2]), ("test_fixed", temp_datasets[3])):
            for soc_start in starts:
                filtered_ds = filter_from_soc_start(base_dataset, soc_start)
                rows.extend(
                    predict_dataset_rows(
                        model,
                        scaler,
                        filtered_ds,
                        config,
                        f"temperature_extrapolation_{range_label}",
                        split_name,
                        soc_start,
                        ckpt_path,
                    )
                )
        write_predictions(rows, out_dir, run_id, f"temperature_extrapolation_{range_label}")


def main():
    parser = argparse.ArgumentParser(
        description="Export SOC true-vs-estimated CSVs for the accepted temperature-adapter baseline."
    )
    parser.add_argument("--scope", choices=["filtered", "extreme", "both"], default="both")
    parser.add_argument("--filtered-checkpoint", default=None, help="Checkpoint for filtered/interpolation tests. Defaults to latest.")
    parser.add_argument("--extreme-checkpoint", default=None, help="Checkpoint for extreme-temperature tests. Defaults to latest.")
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    starts = parse_float_list(args.starts) or DEFAULT_SOC_STARTS
    run_id = datetime.now().strftime("adapter_predictions_%m%d_%H%M%S")
    print("=== Prediction export plan ===", flush=True)
    print(f"METHOD : {METHOD}", flush=True)
    print(f"SCOPE  : {args.scope}", flush=True)
    print(f"STARTS : {starts}", flush=True)
    print("Filtered checkpoint is used for filtered full-SOC, filtered multi-SOC, and 15-20C/30-35C interpolation.", flush=True)
    print("Extreme checkpoint is used for T<13/T>37 full-SOC and multi-SOC extrapolation.", flush=True)

    if args.scope in {"filtered", "both"}:
        export_filtered(args, run_id, starts)
    if args.scope in {"extreme", "both"}:
        export_extreme(args, run_id, starts)

    print("Prediction export finished.", flush=True)


if __name__ == "__main__":
    main()
