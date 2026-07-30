import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from torch_io import load_torch_file, save_torch_file


SEGMENT_RE = re.compile(
    r"^cell(?P<cell_id>\d+-\d+)_idx(?P<idx>\d+)_CAPtest(?P<cap_test>[0-9]+(?:p[0-9]+)?)mAh_"
    r"seg(?P<seg>\d+)_T(?P<temp>-?\d+)C_Q(?P<q>[0-9]+(?:p[0-9]+)?)mAh(?:_dup\d+)?\.csv$"
)


def _token_to_float(token: Optional[str]) -> Optional[float]:
    if token is None:
        return None
    return float(token.replace("p", "."))


def parse_segment_filename(filename: str) -> Dict[str, object]:
    """Parse metadata from a processed segment filename."""
    base = os.path.basename(filename)
    match = SEGMENT_RE.match(base)
    if match:
        gd = match.groupdict()
        return {
            "cell_id": gd["cell_id"],
            "idx": int(gd["idx"]),
            "cap_test_mAh": _token_to_float(gd["cap_test"]),
            "segment_index": int(gd["seg"]),
            "temp_C": float(gd["temp"]),
            "q_mAh": _token_to_float(gd["q"]),
        }

    # Fallback parser for incomplete names.
    cell_match = re.search(r"cell(?P<cell_id>\d+-\d+)", base)
    idx_match = re.search(r"idx(?P<idx>\d+)", base)
    cap_match = re.search(r"CAPtest(?P<cap>[0-9]+(?:p[0-9]+)?)mAh", base)
    seg_match = re.search(r"seg(?P<seg>\d+)", base)
    temp_match = re.search(r"_T(?P<temp>-?\d+)C", base)
    q_match = re.search(r"_Q(?P<q>[0-9]+(?:p[0-9]+)?)mAh", base)
    return {
        "cell_id": cell_match.group("cell_id") if cell_match else "",
        "idx": int(idx_match.group("idx")) if idx_match else -1,
        "cap_test_mAh": _token_to_float(cap_match.group("cap")) if cap_match else None,
        "segment_index": int(seg_match.group("seg")) if seg_match else -1,
        "temp_C": float(temp_match.group("temp")) if temp_match else float("nan"),
        "q_mAh": _token_to_float(q_match.group("q")) if q_match else None,
    }


def _resolve_split_file(data_dir: str, split_file: Optional[str] = None) -> str:
    if split_file:
        return split_file

    data_path = Path(data_dir).resolve()
    candidates = [
        data_path.parent.parent.parent / "splits" / "split_by_cell.csv",
        data_path.parent / "metadata" / "split_by_cell.csv",
        data_path.parent.parent / "metadata" / "split_by_cell.csv",
        data_path.parent.parent / "soccc_segments" / "metadata" / "split_by_cell.csv",
        data_path / "metadata" / "split_by_cell.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"Could not locate split_by_cell.csv for data_dir={data_dir}. "
        "Pass split_file explicitly."
    )


def _segment_sort_key(filename: str):
    meta = parse_segment_filename(filename)
    return (
        meta.get("cell_id", ""),
        meta.get("idx", -1),
        meta.get("segment_index", -1),
        meta.get("temp_C", float("nan")),
        meta.get("q_mAh", -1.0) or -1.0,
        filename,
    )


def temperature_bin_from_value(temp_c: float) -> str:
    """Map a numeric temperature to the coarse bins used in this dataset."""
    if temp_c <= 12:
        return "10C"
    if temp_c <= 20:
        return "15-20C"
    if temp_c <= 27:
        return "25C"
    if temp_c <= 35:
        return "30-35C"
    return "40C"


def split_soccc_by_cells(data_dir: str, split_file: Optional[str] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return train/val/test_random/test_fixed file lists based on cell-level split metadata."""
    split_path = _resolve_split_file(data_dir, split_file)
    split_df = pd.read_csv(split_path)
    if not {"cell_id", "split"}.issubset(split_df.columns):
        raise ValueError(f"Invalid split file: {split_path}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    split_map = dict(zip(split_df["cell_id"].astype(str), split_df["split"].astype(str)))

    buckets = {
        "train": [],
        "val": [],
        "test_random": [],
        "test_fixed": [],
    }

    for f in files:
        meta = parse_segment_filename(f)
        cell_id = meta["cell_id"]
        split_name = split_map.get(cell_id)
        if split_name is None:
            continue
        if split_name not in buckets:
            continue
        buckets[split_name].append(f)

    for key in buckets:
        buckets[key] = sorted(buckets[key], key=_segment_sort_key)

    return buckets["train"], buckets["val"], buckets["test_random"], buckets["test_fixed"]


def split_soccc_by_temperature(
    data_dir: str,
    train_bins: Tuple[str, ...] = ("10C", "25C", "40C"),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split files by temperature bins.

    Train/val are sampled only from the exact-temperature bins in `train_bins`.
    Files in the interval-temperature bins are held out as test splits:
      - test_15_20C
      - test_30_35C

    The split is done at the source-file level, so all segments from the same
    original cycle stay together.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    grouped: Dict[Tuple[str, int, float], List[str]] = defaultdict(list)
    temp_bin_map: Dict[Tuple[str, int, float], str] = {}

    for f in files:
        meta = parse_segment_filename(f)
        cell_id = str(meta.get("cell_id", ""))
        idx = int(meta.get("idx", -1))
        temp_c = float(meta.get("temp_C", float("nan")))
        if not cell_id or idx < 0 or np.isnan(temp_c):
            continue
        key = (cell_id, idx, temp_c)
        grouped[key].append(f)
        temp_bin_map[key] = temperature_bin_from_value(temp_c)

    rng = np.random.default_rng(seed)
    train_files: List[str] = []
    val_files: List[str] = []
    test_15_20: List[str] = []
    test_30_35: List[str] = []

    train_bins_set = set(train_bins)
    train_source_keys_by_bin: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

    for key, files_in_source in grouped.items():
        tbin = temp_bin_map[key]
        if tbin == "15-20C":
            test_15_20.extend(files_in_source)
        elif tbin == "30-35C":
            test_30_35.extend(files_in_source)
        elif tbin in train_bins_set:
            train_source_keys_by_bin[tbin].append(key)

    for tbin, keys in train_source_keys_by_bin.items():
        if not keys:
            continue
        keys = list(keys)
        rng.shuffle(keys)
        n_val = int(round(len(keys) * val_ratio))
        if len(keys) > 1:
            n_val = max(1, min(n_val, len(keys) - 1))
        else:
            n_val = 0
        val_keys = set(keys[:n_val])
        for key in keys:
            if key in val_keys:
                val_files.extend(grouped[key])
            else:
                train_files.extend(grouped[key])

    train_files = sorted(train_files, key=_segment_sort_key)
    val_files = sorted(val_files, key=_segment_sort_key)
    test_15_20 = sorted(test_15_20, key=_segment_sort_key)
    test_30_35 = sorted(test_30_35, key=_segment_sort_key)
    return train_files, val_files, test_15_20, test_30_35


def split_soccc_by_cell_and_temperature(
    data_dir: str,
    split_file: Optional[str] = None,
    train_bins: Tuple[str, ...] = ("15-20C", "25C", "30-35C"),
    test_low_bin: str = "10C",
    test_high_bin: str = "40C",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split files by the intersection of a main cell split and temperature bins.

    Rules:
      - Only files from cell splits in {"train", "val"} are eligible for train/val.
      - Only files from cell splits in {"test_random", "test_fixed"} are eligible
        for test splits.
      - Train/val are sampled only from `train_bins`.
      - `test_low_bin` and `test_high_bin` are held out as the two test splits.
      - The split is done at the source-file level so all segments from the same
        original cycle stay together.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    split_path = _resolve_split_file(data_dir, split_file)
    split_df = pd.read_csv(split_path)
    if not {"cell_id", "split"}.issubset(split_df.columns):
        raise ValueError(f"Invalid split file: {split_path}")

    split_map = dict(zip(split_df["cell_id"].astype(str), split_df["split"].astype(str)))
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    grouped: Dict[Tuple[str, int, float], List[str]] = defaultdict(list)
    meta_map: Dict[Tuple[str, int, float], Dict[str, object]] = {}

    for f in files:
        meta = parse_segment_filename(f)
        cell_id = str(meta.get("cell_id", ""))
        idx = int(meta.get("idx", -1))
        temp_c = float(meta.get("temp_C", float("nan")))
        if not cell_id or idx < 0 or np.isnan(temp_c):
            continue
        key = (cell_id, idx, temp_c)
        grouped[key].append(f)
        meta_map[key] = {
            "cell_split": split_map.get(cell_id, ""),
            "temp_bin": temperature_bin_from_value(temp_c),
        }

    rng = np.random.default_rng(seed)
    train_files: List[str] = []
    val_files: List[str] = []
    test_low_files: List[str] = []
    test_high_files: List[str] = []

    train_bins_set = set(train_bins)
    train_source_keys_by_bin: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

    for key, files_in_source in grouped.items():
        cell_split = str(meta_map[key]["cell_split"])
        tbin = str(meta_map[key]["temp_bin"])

        if cell_split in {"test_random", "test_fixed"}:
            if tbin == test_low_bin:
                test_low_files.extend(files_in_source)
            elif tbin == test_high_bin:
                test_high_files.extend(files_in_source)
            continue

        if cell_split not in {"train", "val"}:
            continue

        if tbin in train_bins_set:
            train_source_keys_by_bin[tbin].append(key)

    for _, keys in train_source_keys_by_bin.items():
        if not keys:
            continue
        keys = list(keys)
        rng.shuffle(keys)
        n_val = int(round(len(keys) * val_ratio))
        if len(keys) > 1:
            n_val = max(1, min(n_val, len(keys) - 1))
        else:
            n_val = 0
        val_keys = set(keys[:n_val])
        for key in keys:
            if key in val_keys:
                val_files.extend(grouped[key])
            else:
                train_files.extend(grouped[key])

    train_files = sorted(train_files, key=_segment_sort_key)
    val_files = sorted(val_files, key=_segment_sort_key)
    test_low_files = sorted(test_low_files, key=_segment_sort_key)
    test_high_files = sorted(test_high_files, key=_segment_sort_key)
    return train_files, val_files, test_low_files, test_high_files


def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing expected columns: {candidates}")


class BatteryTDGCMDataset(Dataset):
    CACHE_VERSION = 2

    def __init__(self, data_dir, file_list, window_size=100, stride=100, cache_file="dataset_cache.pt"):
        if window_size <= 1:
            raise ValueError(f"window_size must be greater than 1, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        data_dir = str(Path(data_dir).resolve())
        file_list = list(file_list)
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        cache_signature = self._cache_signature(data_dir, file_list, window_size, stride)

        if os.path.exists(cache_file):
            try:
                payload = load_torch_file(cache_file, map_location="cpu")
                if (
                    isinstance(payload, dict)
                    and payload.get("version") == self.CACHE_VERSION
                    and payload.get("signature") == cache_signature
                    and isinstance(payload.get("samples"), list)
                ):
                    self.samples = payload["samples"]
                    print(f"Found valid cache: [{cache_file}] ({len(self.samples)} windows)")
                    return
                print(f"Ignoring stale or legacy cache: [{cache_file}]")
            except Exception as exc:
                print(f"Ignoring unreadable cache [{cache_file}]: {exc}")

        print(f"Parsing {len(file_list)} files and building samples...")
        for file in file_list:
            filepath = os.path.join(data_dir, file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Split references a missing segment file: {filepath}")

            meta = parse_segment_filename(file)
            df = pd.read_csv(filepath).ffill().bfill()
            if len(df) < window_size:
                continue

            current_col = _pick_column(df, ("Current(mA)", "Current(A)"))
            voltage_col = _pick_column(df, ("Voltage(mV)", "Voltage(V)"))
            temp_col = _pick_column(df, ("T(C)", "T"))
            soc_col = _pick_column(df, ("SOC", "SOC/DOD(%)"))

            current_raw = pd.to_numeric(df[current_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            voltage_raw = pd.to_numeric(df[voltage_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            temp_raw = pd.to_numeric(df[temp_col], errors="coerce").ffill().bfill().to_numpy(dtype=np.float32)
            soc_raw = pd.to_numeric(df[soc_col], errors="coerce").ffill().bfill().to_numpy(dtype=np.float32)

            if current_col == "Current(mA)":
                current_a = current_raw / 1000.0
                d_current = np.insert(np.diff(current_a), 0, 0.0)
            else:
                current_a = current_raw
                d_current = np.insert(np.diff(current_a), 0, 0.0)

            if voltage_col == "Voltage(mV)":
                voltage_v = voltage_raw / 1000.0
                d_voltage = np.insert(np.diff(voltage_v), 0, 0.0)
            else:
                voltage_v = voltage_raw
                d_voltage = np.insert(np.diff(voltage_v), 0, 0.0)

            power_col = "Power(W)" if "Power(W)" in df.columns else None
            if power_col is not None:
                power_w = pd.to_numeric(df[power_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            else:
                power_w = voltage_v * current_a

            if soc_col == "SOC/DOD(%)":
                soc = soc_raw / 100.0
            else:
                soc = soc_raw

            q_mAh = meta.get("q_mAh")
            if q_mAh is None:
                cap_test = meta.get("cap_test_mAh")
                q_mAh = cap_test if cap_test is not None else 0.0
            q_As = float(q_mAh) * 3.6
            if not np.isfinite(q_As) or q_As <= 0.0:
                raise ValueError(f"Invalid capacity parsed from segment filename: {file}")

            if not np.isfinite(temp_raw).all():
                raise ValueError(f"Temperature contains non-finite values after filling: {file}")
            if not np.isfinite(soc).all():
                raise ValueError(f"SOC contains non-finite values after filling: {file}")

            features_tensor = torch.tensor(
                np.column_stack([current_a, d_current, voltage_v, d_voltage, power_w]),
                dtype=torch.float32,
            )
            soc_tensor = torch.tensor(soc, dtype=torch.float32).unsqueeze(-1)
            t_tensor = torch.tensor(temp_raw, dtype=torch.float32)

            start_offset = len(df) % stride
            num_windows = (len(df) - start_offset - window_size) // stride + 1
            if num_windows <= 0:
                continue

            for i in range(num_windows):
                start = start_offset + i * stride
                end = start + window_size
                self.samples.append(
                    {
                        "x_dyn": features_tensor[start:end, :],
                        "t_mean": t_tensor[start:end].mean().unsqueeze(0),
                        "soc": soc_tensor[start:end, :],
                        "Q": torch.tensor(q_As, dtype=torch.float32),
                        "Q_mAh": torch.tensor(float(q_mAh), dtype=torch.float32),
                        "dt": torch.tensor(1.0, dtype=torch.float32),
                        "is_first": torch.tensor(i == 0, dtype=torch.bool),
                        "filenames": file,
                        "cell_id": meta.get("cell_id", ""),
                        "idx": meta.get("idx", -1),
                        "segment_index": meta.get("segment_index", -1),
                        "temp_C": meta.get("temp_C", float("nan")),
                        "cap_test_mAh": torch.tensor(
                            float(meta.get("cap_test_mAh") or 0.0), dtype=torch.float32
                        ),
                    }
                )

        cache_parent = os.path.dirname(cache_file)
        if cache_parent:
            os.makedirs(cache_parent, exist_ok=True)
        payload = {
            "version": self.CACHE_VERSION,
            "signature": cache_signature,
            "samples": self.samples,
        }
        save_torch_file(payload, cache_file)
        print(f"Cache saved to {cache_file} (total {len(self.samples)} windows)")

    @staticmethod
    def _cache_signature(data_dir, file_list, window_size, stride):
        digest = hashlib.sha256()
        digest.update(str(Path(data_dir).resolve()).encode("utf-8"))
        digest.update(f"|window={window_size}|stride={stride}".encode("ascii"))
        for filename in file_list:
            filepath = Path(data_dir) / filename
            if not filepath.is_file():
                raise FileNotFoundError(f"Split references a missing segment file: {filepath}")
            stat = filepath.stat()
            digest.update(filename.encode("utf-8"))
            digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}".encode("ascii"))
        return digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
