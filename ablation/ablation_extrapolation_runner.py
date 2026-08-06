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
BASELINE_TUNE_DIR = CODE_ROOT / "baseline_tune"

for path in (str(SHARED_DIR), str(BASELINE_TUNE_DIR), str(BASE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import baseline_d96head4lay3 as base
from ablation_models import AdaLNNoStateTransferModel, TempInputAdaLNCausalTransformerModel, TempInputCausalTransformerModel
from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, parse_segment_filename, split_soccc_by_cells
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, set_seed
from tune_preset_soc_start_causal_adaln_dropout import aggregate_segment_rows


PROJECT = "PINT_ABLATION_TEMP_EXTRAP_NOAH_D96_H4_L3"
TRAIN_SCOPE_LABEL = "mid_15_20_25pm2_30_35"
TEST_SCOPE_LABELS = [
    ("T10_pm2", 10.0, 2.0),
    ("T40_pm2", 40.0, 2.0),
]
SOC_STARTS = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]


ABLATIONS = {
    "no_ahloss": {
        "project": f"{PROJECT}_NO_AHLOSS",
        "result_dir": "temp_extrap_no_ahloss_d96h4l3",
        "title": "Temperature extrapolation: no Ah loss",
        "model": "adaln",
        "use_ah_loss": False,
    },
    "no_adaln_temp_input_no_ah": {
        "project": f"{PROJECT}_NO_ADALN_TEMP_INPUT_NO_AH",
        "result_dir": "temp_extrap_no_adaln_temp_input_no_ah_d96h4l3",
        "title": "Temperature extrapolation: no AdaLN, temp appended to input, no Ah loss",
        "model": "temp_input",
        "use_ah_loss": False,
    },
    "temp_input_adaln_no_ah": {
        "project": f"{PROJECT}_TEMP_INPUT_ADALN_NO_AH",
        "result_dir": "temp_extrap_temp_input_adaln_no_ah_d96h4l3",
        "title": "Temperature extrapolation: temp appended to input + AdaLN modulation, no Ah loss",
        "model": "temp_input_adaln",
        "use_ah_loss": False,
    },
}


def temp_scope_suffix() -> str:
    return "trainT15_20_T25pm2_T30_35__testT10pm2_T40pm2"


def _make_results_dirs(result_dir: str):
    base_dir = base.organized_results_dir() / "ablation" / result_dir
    cache_dir = base_dir / "pt"
    pth_dir = base_dir / "pth_save"
    csv_dir = base_dir / "csv_save"
    meta_dir = base_dir / "metadata"
    for d in (cache_dir, pth_dir, csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"run_root": base_dir, "cache_dir": cache_dir, "pth_dir": pth_dir, "csv_dir": csv_dir, "meta_dir": meta_dir}


def filter_files_by_temperature_ranges(files: list[str], ranges: list[tuple[float, float]]) -> list[str]:
    kept = []
    for filename in files:
        temp_c = float(parse_segment_filename(filename).get("temp_C", float("nan")))
        if any(low <= temp_c <= high for low, high in ranges):
            kept.append(filename)
    return kept


def filter_files_by_temperature_targets(files: list[str], targets: list[float], tolerance: float) -> list[str]:
    kept = []
    for filename in files:
        temp_c = float(parse_segment_filename(filename).get("temp_C", float("nan")))
        if any(abs(temp_c - target) <= tolerance for target in targets):
            kept.append(filename)
    return kept


def make_config(ablation_name: str, args: argparse.Namespace | None = None) -> dict:
    spec = ABLATIONS[ablation_name]
    dirs = _make_results_dirs(spec["result_dir"])
    cfg = {
        "PROJECT": spec["project"],
        "ABLATION": ablation_name,
        "ABLATION_TITLE": spec["title"],
        "MODEL_KIND": spec["model"],
        "DATA_DIR": str(base.processed_segments_dir()),
        "SPLIT_FILE": str(base.split_file_path()),
        "CACHE_DIR": str(dirs["cache_dir"]),
        "PTH_DIR": str(dirs["pth_dir"]),
        "CSV_DIR": str(dirs["csv_dir"]),
        "META_DIR": str(dirs["meta_dir"]),
        "BATCH_SIZE": 64,
        "VAL_BATCH_SIZE": 64,
        "SOC_BATCH_SIZE": 64,
        "LR": 2e-4,
        "EPOCHS": 50,
        "D_MODEL": 96,
        "NHEAD": 4,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10,
        "LAMBDA_AH_STEP": 20.0,
        "GRAD_CLIP": 1.0,
        "P_RESET": 0.05,
        "TRAIN_RANGES": [(15.0, 20.0), (30.0, 35.0)],
        "TRAIN_TARGETS": [25.0],
        "TRAIN_TARGET_TOLERANCE": 2.0,
        "TEST_TARGETS": [10.0, 40.0],
        "TEST_TARGET_TOLERANCE": 2.0,
        "SOC_STARTS": SOC_STARTS,
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "USE_AH_LOSS": bool(spec["use_ah_loss"]),
        "DISABLE_TQDM": False,
    }
    if args is not None:
        cfg["EPOCHS"] = int(args.epochs)
        cfg["BATCH_SIZE"] = int(args.batch_size)
        cfg["VAL_BATCH_SIZE"] = int(args.val_batch_size)
        cfg["SOC_BATCH_SIZE"] = int(args.soc_batch_size)
        cfg["DEVICE"] = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        cfg["DISABLE_TQDM"] = bool(args.disable_tqdm)
    return cfg


def print_config(config: dict):
    print(f"=== {config['ABLATION_TITLE']} ===")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR  : {config['CACHE_DIR']}")
    print(f"PTH_DIR    : {config['PTH_DIR']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"BATCH_SIZE : {config['BATCH_SIZE']}")
    print(f"VAL_BATCH  : {config['VAL_BATCH_SIZE']}")
    print(f"D_MODEL    : {config['D_MODEL']}")
    print(f"NHEAD      : {config['NHEAD']}")
    print(f"LAYERS     : {config['NUM_LAYERS']}")
    print(f"DROPOUT    : {config['DROPOUT']}")
    print(f"P_RESET    : {config['P_RESET']}")
    print(f"TRAIN      : ranges={config['TRAIN_RANGES']} + targets={config['TRAIN_TARGETS']} +/-{config['TRAIN_TARGET_TOLERANCE']}C")
    print(f"TEST       : targets={config['TEST_TARGETS']} +/-{config['TEST_TARGET_TOLERANCE']}C")
    print(f"WINDOW     : {config['WINDOW_SIZE']}")
    print(f"STRIDE     : {config['STRIDE']}")
    print(f"USE_AH_LOSS: {config['USE_AH_LOSS']}")
    sys.stdout.flush()


def build_model(config: dict):
    kwargs = {
        "d_model": config["D_MODEL"],
        "nhead": config["NHEAD"],
        "num_layers": config["NUM_LAYERS"],
        "dropout": config["DROPOUT"],
    }
    if config["MODEL_KIND"] == "adaln":
        return AdaLNBatteryTDGCMModel(**kwargs, use_causal=True).to(config["DEVICE"])
    if config["MODEL_KIND"] == "temp_input":
        return TempInputCausalTransformerModel(**kwargs).to(config["DEVICE"])
    if config["MODEL_KIND"] == "temp_input_adaln":
        return TempInputAdaLNCausalTransformerModel(**kwargs).to(config["DEVICE"])
    raise ValueError(f"Unknown MODEL_KIND: {config['MODEL_KIND']}")


def latest_checkpoint(pth_dir: Path, project: str) -> Path:
    ckpts = list(pth_dir.glob(f"best_model_{project}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for {project} in {pth_dir}.")
    return max(ckpts, key=lambda path: path.stat().st_mtime)


def make_cache_suffix(config: dict) -> str:
    return temp_scope_suffix()


def load_extrapolation_train_datasets(config: dict):
    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    raw_counts = {
        "train": len(train_f),
        "val": len(val_f),
        "test_random": len(test_random_f),
        "test_fixed": len(test_fixed_f),
    }
    train_ranges_f = filter_files_by_temperature_ranges(train_f, config["TRAIN_RANGES"])
    val_ranges_f = filter_files_by_temperature_ranges(val_f, config["TRAIN_RANGES"])
    train_targets_f = filter_files_by_temperature_targets(train_f, config["TRAIN_TARGETS"], config["TRAIN_TARGET_TOLERANCE"])
    val_targets_f = filter_files_by_temperature_targets(val_f, config["TRAIN_TARGETS"], config["TRAIN_TARGET_TOLERANCE"])
    train_f = sorted(set(train_ranges_f) | set(train_targets_f))
    val_f = sorted(set(val_ranges_f) | set(val_targets_f))
    test_random_f = filter_files_by_temperature_targets(test_random_f, config["TEST_TARGETS"], config["TEST_TARGET_TOLERANCE"])
    test_fixed_f = filter_files_by_temperature_targets(test_fixed_f, config["TEST_TARGETS"], config["TEST_TARGET_TOLERANCE"])
    print(
        f"Temperature extrapolation filter | "
        f"train={raw_counts['train']}->{len(train_f)} | "
        f"val={raw_counts['val']}->{len(val_f)} | "
        f"test_random={raw_counts['test_random']}->{len(test_random_f)} | "
        f"test_fixed={raw_counts['test_fixed']}->{len(test_fixed_f)}"
    )
    split_lists = {"train": train_f, "val": val_f, "test_random": test_random_f, "test_fixed": test_fixed_f}
    empty_splits = [name for name, files in split_lists.items() if not files]
    if empty_splits:
        raise RuntimeError(f"Empty split(s) after temperature extrapolation filtering: {', '.join(empty_splits)}")

    suffix = make_cache_suffix(config)
    train_ds = BatteryTDGCMDataset(config["DATA_DIR"], train_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"train_cache_extrap_{suffix}.pt"))
    val_ds = BatteryTDGCMDataset(config["DATA_DIR"], val_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"val_cache_extrap_{suffix}.pt"))
    test_random_ds = BatteryTDGCMDataset(config["DATA_DIR"], test_random_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"test_random_cache_extrap_{suffix}.pt"))
    test_fixed_ds = BatteryTDGCMDataset(config["DATA_DIR"], test_fixed_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"test_fixed_cache_extrap_{suffix}.pt"))
    return train_ds, val_ds, test_random_ds, test_fixed_ds


def load_extrapolation_test_datasets(config: dict, range_label: str, low_c: float, high_c: float):
    _, _, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    raw_random = len(test_random_f)
    raw_fixed = len(test_fixed_f)
    test_random_f = filter_files_by_temperature_targets(test_random_f, [low_c], high_c)
    test_fixed_f = filter_files_by_temperature_targets(test_fixed_f, [low_c], high_c)
    print(
        f"Temperature extrapolation filter: {range_label} {low_c:g}±{high_c:g} C | "
        f"test_random={raw_random}->{len(test_random_f)} | "
        f"test_fixed={raw_fixed}->{len(test_fixed_f)}"
    )
    if not test_random_f or not test_fixed_f:
        raise RuntimeError(f"Empty temperature-extrapolation test split for {range_label}: test_random={len(test_random_f)}, test_fixed={len(test_fixed_f)}")

    suffix = f"{temp_scope_suffix()}_{range_label}"
    test_random_ds = BatteryTDGCMDataset(config["DATA_DIR"], test_random_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"test_random_cache_extrap_{suffix}.pt"))
    test_fixed_ds = BatteryTDGCMDataset(config["DATA_DIR"], test_fixed_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"], cache_file=os.path.join(config["CACHE_DIR"], f"test_fixed_cache_extrap_{suffix}.pt"))
    return None, None, test_random_ds, test_fixed_ds


def write_test_segment_metrics(rows: list[dict], csv_dir: str, run_id: str):
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ("split_order", "metric_order", "segment_order") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")
    out_path = Path(csv_dir) / f"test_segment_metrics_{run_id}.csv"
    total_path = Path(csv_dir) / f"test_total_summary_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    df[df["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"Test segment metrics saved: {out_path}")
    print(f"Test total summary saved: {total_path}")
    return df


def metric_value(metrics_df: pd.DataFrame, split: str) -> float:
    rows = metrics_df[(metrics_df["metric_level"].eq("total")) & (metrics_df["split"].eq(split))]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows["avg_mae"], errors="coerce").mean())


def evaluate_test_splits(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str):
    _, _, test_random_ds, test_fixed_ds = datasets
    rows = []
    for split_order, (split_name, dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        loader = DataLoader(dataset, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
        label = "Test random" if split_name == "test_random" else "Test fixed"
        avg_mae, file_maes = base.evaluate_dataset(
            model,
            loader,
            scaler,
            {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"], "DISABLE_TQDM": config.get("DISABLE_TQDM", False)},
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
            {**config, "TRIAL_ID": run_id, "TRIAL_INDEX": 1, "SOC_BATCH_SIZE": config["VAL_BATCH_SIZE"]},
            split_name,
            split_order,
            math.nan,
            0,
            file_maes,
            dataset,
        )
        rows.extend(segment_rows)
    metrics_df = write_test_segment_metrics(rows, config["CSV_DIR"], run_id)
    return metrics_df


def evaluate_soc_start_splits(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str):
    _, _, test_random_ds, test_fixed_ds = datasets
    rows = []
    for split_order, (split_name, base_dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        for soc_order, soc_start in enumerate(config["SOC_STARTS"], start=1):
            filtered_ds = base.filter_from_soc_start(base_dataset, soc_start)
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
            avg_mae, file_maes = base.evaluate_dataset(
                model,
                loader,
                scaler,
                {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"], "DISABLE_TQDM": True},
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
                {**config, "TRIAL_ID": run_id, "TRIAL_INDEX": 1},
                split_name,
                split_order,
                soc_start,
                soc_order,
                file_maes,
                filtered_ds,
            )
            rows.extend(segment_rows)

    metrics_df = write_test_segment_metrics(rows, config["CSV_DIR"], run_id)
    return metrics_df


def evaluate_temperature_extrapolation(model, scaler: PITDScaler, config: dict, run_id: str) -> pd.DataFrame:
    frames = []
    for range_order, (range_label, low_c, high_c) in enumerate(TEST_SCOPE_LABELS, start=1):
        datasets = load_extrapolation_test_datasets(config, range_label, low_c, high_c)
        range_run_id = f"{run_id}_{range_label}"
        metrics_df = evaluate_soc_start_splits(model, scaler, datasets, config, range_run_id).copy()
        metrics_df.insert(0, "temp_range_order", range_order)
        metrics_df.insert(1, "temp_range", range_label)
        metrics_df.insert(2, "temp_low_C", low_c)
        metrics_df.insert(3, "temp_high_C", high_c)
        frames.append(metrics_df)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = Path(config["CSV_DIR"]) / f"temp_extrap_soc_start_metrics_{run_id}.csv"
    combined_total_path = Path(config["CSV_DIR"]) / f"temp_extrap_soc_start_total_summary_{run_id}.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    combined[combined["metric_level"].eq("total")].to_csv(combined_total_path, index=False, encoding="utf-8-sig")
    print(f"Temperature-extrapolation detailed metrics saved: {combined_path}")
    print(f"Temperature-extrapolation total summary saved: {combined_total_path}")
    return combined


def summarize_final(test_df: pd.DataFrame, soc_df: pd.DataFrame, temp_df: pd.DataFrame, config: dict, run_id: str, best_epoch, best_val_mae):
    def mv(df, split):
        rows = df[df["metric_level"].eq("total") & df["split"].eq(split)]
        return float(pd.to_numeric(rows["avg_mae"], errors="coerce").mean()) if not rows.empty else float("nan")

    row = {
        "run_id": run_id,
        "ablation": config["ABLATION"],
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_random_mae": mv(test_df, "test_random"),
        "test_fixed_mae": mv(test_df, "test_fixed"),
        "soc_random_mean_mae": mv(soc_df, "test_random"),
        "soc_fixed_mean_mae": mv(soc_df, "test_fixed"),
    }
    row["test_mean_mae"] = float(np.nanmean([row["test_random_mae"], row["test_fixed_mae"]]))
    row["soc_mean_mae"] = float(np.nanmean([row["soc_random_mean_mae"], row["soc_fixed_mean_mae"]]))
    for range_label, _, _ in TEST_SCOPE_LABELS:
        safe = range_label.replace("-", "_")
        subset = temp_df[temp_df["temp_range"].eq(range_label)]
        row[f"tempgen_{safe}_random_mean_mae"] = mv(subset, "test_random")
        row[f"tempgen_{safe}_fixed_mean_mae"] = mv(subset, "test_fixed")
        row[f"tempgen_{safe}_mean_mae"] = float(np.nanmean([row[f"tempgen_{safe}_random_mean_mae"], row[f"tempgen_{safe}_fixed_mean_mae"]]))
    row["tempgen_mean_mae"] = float(np.nanmean([row[f"tempgen_{label.replace('-', '_')}_mean_mae"] for label, _, _ in TEST_SCOPE_LABELS]))
    out = Path(config["CSV_DIR"]) / f"final_summary_{run_id}.csv"
    pd.DataFrame([row]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Final summary saved: {out}")
    print(pd.DataFrame([row]).T.to_string(header=False))


def train_one_epoch_stateful(model, train_ds, scaler, optimizer, criterion, config, epoch: int):
    model.train()
    manager = BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
    h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
    pbar = tqdm(total=manager.total_steps, desc=f"Epoch {epoch}/{config['EPOCHS']} Training", colour="blue", disable=config.get("DISABLE_TQDM", False))
    curr_lambda = 0.0 if not config["USE_AH_LOSS"] else config["LAMBDA_AH_START"] + (epoch - 1) * config["LAMBDA_AH_STEP"]
    losses = defaultdict(list)
    print(
        f"[Epoch {epoch}/{config['EPOCHS']}] start | mode=stateful | train_windows={len(train_ds)} | steps={manager.total_steps} | "
        f"p_reset={config['P_RESET']:.3f} | lambda_ah={curr_lambda:.2f}",
        flush=True,
    )

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
            h_state[:, f_t, :] = 0.0
        active_mask = m_t > 0.5
        eligible_mask = active_mask & (~f_t)
        reset_mask = (torch.rand(config["BATCH_SIZE"], device=config["DEVICE"]) < config["P_RESET"]) & eligible_mask
        if reset_mask.any():
            h_state[:, reset_mask, :] = 0.0
        x_n, t_n = scaler.transform(x, t)
        y_p, h_state = model(x_n, t_n, h_state)
        loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], q, f_t, curr_lambda, m_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
        optimizer.step()
        losses["total"].append(loss.item())
        losses["data"].append(l_d.item())
        losses["ah"].append(l_ah.item())
        losses["range"].append(l_range.item())
        pbar.update(1)
    pbar.close()
    return losses, curr_lambda


def train_ablation(ablation_name: str, args: argparse.Namespace):
    set_seed(args.seed)
    config = make_config(ablation_name, args)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"{config['PROJECT']}_{timestamp}"
    print_config(config)
    datasets = load_extrapolation_train_datasets(config)
    train_ds, val_ds, _, _ = datasets
    print(f"Dataset windows | train={len(train_ds)} | val={len(val_ds)} | test_random={len(datasets[2])} | test_fixed={len(datasets[3])}", flush=True)
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    scaler = PITDScaler()
    scaler.fit(train_ds)
    model = build_model(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-5)
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])
    best_val_mae = float("inf")
    best_epoch = -1
    best_model_path = Path(config["PTH_DIR"]) / f"best_model_{run_id}.pth"
    history = []
    val_mae_matrix = defaultdict(list)

    for epoch in range(1, config["EPOCHS"] + 1):
        losses, curr_lambda = train_one_epoch_stateful(model, train_ds, scaler, optimizer, criterion, config, epoch)
        avg_val_mae, file_maes = base.evaluate_dataset(
            model,
            val_loader,
            scaler,
            {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"], "DISABLE_TQDM": config.get("DISABLE_TQDM", False)},
            label=f"Epoch {epoch} Val",
        )
        scheduler.step()
        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            save_torch_file({"model_state_dict": model.state_dict(), "scaler_stats": scaler.stats, "config": config}, best_model_path)
            print(f"[Epoch {epoch}] new best model saved: val_mae={best_val_mae:.6f}")
        row = {
            "Epoch": epoch,
            "LR": float(optimizer.param_groups[0]["lr"]),
            "Train_Loss": float(np.mean(losses["total"])) if losses["total"] else float("nan"),
            "Train_Data_Loss": float(np.mean(losses["data"])) if losses["data"] else float("nan"),
            "Train_Ah_Loss": float(np.mean(losses["ah"])) if losses["ah"] else float("nan"),
            "Train_Range_Loss": float(np.mean(losses["range"])) if losses["range"] else float("nan"),
            "Val_MAE_Avg": avg_val_mae,
            "Best_Val_MAE_So_Far": best_val_mae,
            "Best_Epoch_So_Far": best_epoch,
            "Lambda": curr_lambda,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(Path(config["CSV_DIR"]) / f"log_{run_id}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_mae_matrix).T.to_csv(Path(config["CSV_DIR"]) / f"val_matrix_{run_id}.csv", encoding="utf-8-sig")
        print(
            f"[Epoch {epoch}] summary | train={row['Train_Loss']:.6f} | data={row['Train_Data_Loss']:.6f} | "
            f"ah={row['Train_Ah_Loss']:.6f} | val={avg_val_mae:.6f} | best={best_val_mae:.6f} | best_epoch={best_epoch} | lambda_ah={curr_lambda:.2f}",
            flush=True,
        )

    ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded best checkpoint for final evaluation: {best_model_path}")
    test_df = evaluate_test_splits(model, scaler, datasets, config, run_id)
    print("\nRunning multi-SOC-start evaluation...")
    soc_df = evaluate_soc_start_splits(model, scaler, datasets, config, run_id)
    print("\nRunning temperature-extrapolation multi-SOC-start evaluation...")
    temp_df = evaluate_temperature_extrapolation(model, scaler, config, run_id)
    summarize_final(test_df, soc_df, temp_df, config, run_id, best_epoch, best_val_mae)


def evaluate_latest(ablation_name: str, args: argparse.Namespace):
    config = make_config(ablation_name, args)
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]), config["PROJECT"])
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    fresh = make_config(ablation_name, args)
    for key in ("CACHE_DIR", "PTH_DIR", "CSV_DIR", "META_DIR", "VAL_BATCH_SIZE", "SOC_BATCH_SIZE", "DEVICE", "DISABLE_TQDM"):
        config[key] = fresh[key]
    print("=== Evaluate latest extrapolation checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print_config(config)
    datasets = load_extrapolation_train_datasets(config)
    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"eval_{ckpt_path.stem}_{timestamp}"
    test_df = evaluate_test_splits(model, scaler, datasets, config, run_id)
    print("\nRunning multi-SOC-start evaluation...")
    soc_df = evaluate_soc_start_splits(model, scaler, datasets, config, run_id)
    print("\nRunning temperature-extrapolation multi-SOC-start evaluation...")
    temp_df = evaluate_temperature_extrapolation(model, scaler, config, run_id)
    summarize_final(test_df, soc_df, temp_df, config, run_id, math.nan, math.nan)


def make_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["train", "eval"], default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def run_cli(ablation_name: str):
    parser = make_parser(ABLATIONS[ablation_name]["title"])
    args = parser.parse_args()
    mode = args.mode
    if mode is None:
        print(f"=== {ABLATIONS[ablation_name]['title']} ===")
        print("1 - start a new training run")
        print("2 - evaluate latest checkpoint with full SOC, multi-SOC, and temperature-extrapolation multi-SOC")
        choice = input("Enter 1 or 2: ").strip()
        mode = "train" if choice == "1" else "eval" if choice == "2" else ""
    if mode == "train":
        train_ablation(ablation_name, args)
    elif mode == "eval":
        evaluate_latest(ablation_name, args)
    else:
        raise ValueError("Invalid choice. Use 1/2 or --mode train/eval.")
