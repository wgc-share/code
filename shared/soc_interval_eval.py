from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from soccc_schemeB import parse_segment_filename
from train_utils import evaluate_dataset


RANDOM_SEGMENT_MAP = {
    1: (1, "random_segment01", "calibration"),
    2: (2, "random_segment02", "digested"),
    3: (3, "random_segment03", "highway"),
    4: (4, "random_segment04", "urban"),
}

FIXED_SEGMENT_MAP = {
    ("3-1", 1): (1, "fixed_segment01", "HWFET"),
    ("3-1", 2): (2, "fixed_segment02", "US06_HWY"),
    ("3-1", 3): (3, "fixed_segment03", "REP05"),
    ("3-2", 1): (4, "fixed_segment04", "DST"),
    ("3-2", 2): (5, "fixed_segment05", "EUDC"),
    ("3-2", 3): (6, "fixed_segment06", "ARTERIAL"),
    ("3-3", 1): (7, "fixed_segment07", "LA92"),
    ("3-3", 2): (8, "fixed_segment08", "SC03"),
    ("3-3", 3): (9, "fixed_segment09", "UDDS"),
    ("3-4", 1): (10, "fixed_segment10", "MANHATTAN"),
    ("3-4", 2): (11, "fixed_segment11", "NYCC"),
    ("3-4", 3): (12, "fixed_segment12", "NUREMBERG"),
}


class FilteredSamplesDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_soc_intervals(text: str) -> list[tuple[float, float]]:
    intervals = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid SOC interval: {item}. Use format like 100-90.")
        left, right = item.split("-", 1)
        high = float(left.strip())
        low = float(right.strip())
        if high < low:
            high, low = low, high
        if high > 100 or low < 0:
            raise ValueError(f"SOC interval out of range [0,100]: {item}")
        intervals.append((high, low))
    if not intervals:
        raise ValueError("No valid SOC intervals were provided.")
    return intervals


def prompt_soc_intervals() -> list[tuple[float, float]]:
    count = int(input("Enter SOC interval count: ").strip())
    text = input("Enter SOC intervals, e.g. 100-90,90-80,80-70: ").strip()
    intervals = parse_soc_intervals(text)
    if len(intervals) != count:
        raise ValueError(f"Expected {count} SOC intervals, got {len(intervals)}.")
    return intervals


def format_interval(high: float, low: float) -> str:
    def fmt(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")

    return f"{fmt(high)}-{fmt(low)}"


def filter_to_soc_interval(dataset, high: float, low: float) -> FilteredSamplesDataset:
    kept = []
    for sample in dataset.samples:
        start_soc_percent = float(sample["soc"][0, 0]) * 100.0
        if low <= start_soc_percent <= high:
            kept.append(sample)
    return FilteredSamplesDataset(kept)


def segment_info(split_name: str, filename: str):
    meta = parse_segment_filename(filename)
    cell_id = str(meta.get("cell_id", ""))
    segment_index = int(meta.get("segment_index", -1))
    if split_name == "test_random":
        mapped = RANDOM_SEGMENT_MAP.get(segment_index)
    elif split_name == "test_fixed":
        mapped = FIXED_SEGMENT_MAP.get((cell_id, segment_index))
    else:
        mapped = None
    if mapped is None:
        return None
    order, segment_id, condition = mapped
    return {
        "segment_order": order,
        "segment_id": segment_id,
        "condition": condition,
        "cell_id": cell_id,
        "segment_index": segment_index,
    }


def aggregate_interval_segment_rows(config: dict, split_name: str, split_order: int, interval_order: int, interval_label: str, high: float, low: float, file_maes: dict, dataset):
    grouped_maes = defaultdict(list)
    grouped_files = defaultdict(set)
    grouped_windows = defaultdict(int)
    grouped_info = {}

    for filename, mae in file_maes.items():
        info = segment_info(split_name, filename)
        if info is None:
            continue
        key = info["segment_id"]
        grouped_maes[key].append(mae)
        grouped_files[key].add(filename)
        grouped_info[key] = info

    for sample in dataset.samples:
        info = segment_info(split_name, sample["filenames"])
        if info is None:
            continue
        key = info["segment_id"]
        grouped_windows[key] += 1
        grouped_info[key] = info

    rows = []
    for key, maes in grouped_maes.items():
        info = grouped_info[key]
        rows.append(
            {
                "trial_id": config["TRIAL_ID"],
                "trial_index": config.get("TRIAL_INDEX", 1),
                "p_reset": config.get("P_RESET", np.nan),
                "split_order": split_order,
                "split": split_name,
                "interval_order": interval_order,
                "soc_interval": interval_label,
                "soc_high_percent": high,
                "soc_low_percent": low,
                "metric_order": 2,
                "metric_level": "segment",
                "segment_order": info["segment_order"],
                "segment_id": info["segment_id"],
                "condition": info["condition"],
                "cell_id": info["cell_id"],
                "segment_index": info["segment_index"],
                "avg_mae": float(np.mean(maes)),
                "files": len(grouped_files[key]),
                "windows": grouped_windows[key],
                "batch_size": config["SOC_INTERVAL_BATCH_SIZE"],
            }
        )
    return sorted(rows, key=lambda row: row["segment_order"])


def evaluate_soc_intervals(model, scaler, datasets: tuple, config: dict, run_id: str, intervals: list[tuple[float, float]]):
    _, _, test_random_ds, test_fixed_ds = datasets
    rows = []
    for split_order, (split_name, base_dataset) in enumerate((("test_random", test_random_ds), ("test_fixed", test_fixed_ds)), start=1):
        for interval_order, (high, low) in enumerate(intervals, start=1):
            interval_label = format_interval(high, low)
            filtered_ds = filter_to_soc_interval(base_dataset, high, low)
            if len(filtered_ds) == 0:
                rows.append(
                    {
                        "trial_id": run_id,
                        "trial_index": 1,
                        "p_reset": config.get("P_RESET", np.nan),
                        "split_order": split_order,
                        "split": split_name,
                        "interval_order": interval_order,
                        "soc_interval": interval_label,
                        "soc_high_percent": high,
                        "soc_low_percent": low,
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
                        "batch_size": config["SOC_INTERVAL_BATCH_SIZE"],
                    }
                )
                continue

            loader = DataLoader(filtered_ds, batch_size=config["SOC_INTERVAL_BATCH_SIZE"], shuffle=False)
            avg_mae, file_maes = evaluate_dataset(
                model,
                loader,
                scaler,
                {
                    "DEVICE": config["DEVICE"],
                    "D_MODEL": config["D_MODEL"],
                    "DISABLE_TQDM": not config.get("SHOW_PROGRESS", False),
                },
                label=f"{run_id} {split_name} SOC {interval_label}%",
            )
            rows.append(
                {
                    "trial_id": run_id,
                    "trial_index": 1,
                    "p_reset": config.get("P_RESET", np.nan),
                    "split_order": split_order,
                    "split": split_name,
                    "interval_order": interval_order,
                    "soc_interval": interval_label,
                    "soc_high_percent": high,
                    "soc_low_percent": low,
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
                    "batch_size": config["SOC_INTERVAL_BATCH_SIZE"],
                }
            )
            rows.extend(
                aggregate_interval_segment_rows(
                    {
                        **config,
                        "TRIAL_ID": run_id,
                        "TRIAL_INDEX": 1,
                    },
                    split_name,
                    split_order,
                    interval_order,
                    interval_label,
                    high,
                    low,
                    file_maes,
                    filtered_ds,
                )
            )
    return rows


def write_soc_interval_metrics(rows: list[dict], csv_dir: str, run_id: str):
    df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ("split_order", "interval_order", "metric_order", "segment_order")
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    out_path = Path(csv_dir) / f"soc_interval_metrics_{run_id}.csv"
    total_path = Path(csv_dir) / f"soc_interval_total_summary_{run_id}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    df[df["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"SOC-interval detailed metrics saved: {out_path}")
    print(f"SOC-interval total metrics saved: {total_path}")
    return df
