from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STEP_SIM = "\u6a21\u62df\u5de5\u51b5"
STEP_CC_DISCHARGE = "\u6052\u6d41\u653e\u7535"


COL_STEP = "step_type"
COL_CURRENT_MA = "current_mA"
COL_VOLTAGE_MV = "voltage_mV"
COL_CAPACITY_MAH = "capacity_mAh"
COL_TEMP_C = "temperature_C"


@dataclass(frozen=True)
class StepBlock:
    index: int
    step_type: str
    start: int
    end: int


def safe_float_token(value: float, digits: int = 1) -> str:
    text = f"{value:.{digits}f}"
    return text.replace("-", "m").replace(".", "p")


def rounded_temperature_token(temp_c: float) -> str:
    return str(int(round(temp_c)))


def unique_name(base_name: str, used_names: dict[str, int]) -> str:
    count = used_names.get(base_name, 0) + 1
    used_names[base_name] = count
    if count == 1:
        return base_name

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    return f"{stem}_dup{count:02d}{suffix}"


def temperature_bin(temp_c: float) -> str:
    if 8.0 <= temp_c <= 12.5:
        return "10C"
    if 12.5 < temp_c <= 22.0:
        return "15-20C"
    if 22.0 < temp_c <= 28.0:
        return "25C"
    if 28.0 < temp_c <= 36.0:
        return "30-35C"
    if 36.0 < temp_c <= 45.0:
        return "40C"
    return "other"


def pick_source_column(df: pd.DataFrame, candidates: tuple[str, ...], path: Path) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"{path.name} is missing required columns: {candidates}")


def numeric_source_column(
    df: pd.DataFrame,
    candidates: tuple[tuple[str, float], ...],
    path: Path,
) -> pd.Series:
    for column, multiplier in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce") * multiplier
    names = tuple(column for column, _ in candidates)
    raise KeyError(f"{path.name} is missing required numeric columns: {names}")


def read_raw_csv(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="gbk")
    step_column = pick_source_column(source, ("工步类型",), path)
    normalized = pd.DataFrame(
        {
            COL_STEP: source[step_column],
            COL_CURRENT_MA: numeric_source_column(
                source, (("电流(mA)", 1.0), ("电流(A)", 1000.0)), path
            ),
            COL_VOLTAGE_MV: numeric_source_column(
                source, (("电压(mV)", 1.0), ("电压(V)", 1000.0)), path
            ),
            COL_CAPACITY_MAH: numeric_source_column(
                source, (("容量(mAh)", 1.0), ("容量(Ah)", 1000.0)), path
            ),
            COL_TEMP_C: numeric_source_column(
                source, (("T1(℃)", 1.0), ("T(C)", 1.0), ("T", 1.0)), path
            ),
        }
    )
    return normalized.ffill().bfill()


def write_csv_replace(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"__tmp__{path.name}")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    os.replace(tmp_path, path)


def contiguous_blocks(df: pd.DataFrame) -> list[StepBlock]:
    steps = df[COL_STEP].astype(str)
    block_ids = (steps != steps.shift()).cumsum()
    blocks: list[StepBlock] = []
    for block_index, (_, block) in enumerate(df.groupby(block_ids), start=1):
        blocks.append(
            StepBlock(
                index=block_index,
                step_type=str(block.iloc[0][COL_STEP]),
                start=int(block.index[0]),
                end=int(block.index[-1]),
            )
        )
    return blocks


def block_frame(df: pd.DataFrame, block: StepBlock) -> pd.DataFrame:
    return df.loc[block.start : block.end].copy()


def capacity_delta_mAh(block_df: pd.DataFrame) -> float:
    cap = pd.to_numeric(block_df[COL_CAPACITY_MAH], errors="coerce")
    if cap.empty:
        return np.nan
    return float(abs(cap.max() - cap.min()))


def first_capacity_test(df: pd.DataFrame, blocks: list[StepBlock]) -> tuple[float, int, float]:
    for block in blocks:
        if block.step_type == STEP_CC_DISCHARGE:
            part = block_frame(df, block)
            cap_mAh = capacity_delta_mAh(part)
            temp = pd.to_numeric(part[COL_TEMP_C], errors="coerce")
            return cap_mAh, block.index, float(temp.mean())
    return np.nan, -1, np.nan


def find_tail_discharge(blocks: list[StepBlock], sim_pos: int) -> StepBlock | None:
    for block in blocks[sim_pos + 1 :]:
        if block.step_type == STEP_CC_DISCHARGE:
            return block
        if block.step_type == STEP_SIM:
            return None
        # Other blocks are usually short rests; keep scanning through them.
    return None


def build_segment(
    sim_df: pd.DataFrame,
    tail_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    sim_cap = pd.to_numeric(sim_df[COL_CAPACITY_MAH], errors="coerce")
    sim_direction = 1.0 if sim_cap.iloc[-1] >= sim_cap.iloc[0] else -1.0
    sim_used_mAh = ((sim_cap - sim_cap.iloc[0]) * sim_direction).astype(float)
    sim_total_mAh = float(sim_used_mAh.iloc[-1])

    tail_cap = pd.to_numeric(tail_df[COL_CAPACITY_MAH], errors="coerce")
    tail_direction = 1.0 if tail_cap.iloc[-1] >= tail_cap.iloc[0] else -1.0
    tail_used_mAh = ((tail_cap - tail_cap.iloc[0]) * tail_direction).astype(float)
    tail_total_mAh = capacity_delta_mAh(tail_df)
    full_discharge_mAh = sim_total_mAh + tail_total_mAh
    if not np.isfinite(full_discharge_mAh) or full_discharge_mAh <= 0:
        raise ValueError("invalid full discharge capacity")

    combined_df = pd.concat([sim_df, tail_df], ignore_index=True)
    current_mA = -pd.to_numeric(combined_df[COL_CURRENT_MA], errors="coerce")
    current_mA = current_mA.mask(current_mA.abs() < 1e-9, 0.0)
    voltage_mV = pd.to_numeric(combined_df[COL_VOLTAGE_MV], errors="coerce")
    temp_c = pd.to_numeric(combined_df[COL_TEMP_C], errors="coerce")

    used_mAh = pd.concat(
        [
            sim_used_mAh.reset_index(drop=True),
            (sim_total_mAh + tail_used_mAh).reset_index(drop=True),
        ],
        ignore_index=True,
    )
    soc = 1.0 - (used_mAh / full_discharge_mAh)
    soc = soc.clip(lower=0.0, upper=1.0)

    out = pd.DataFrame(
        {
            "Current(mA)": current_mA.astype(float),
            "Voltage(mV)": voltage_mV.astype(float),
            "dI(mA)": current_mA.diff().fillna(0.0).astype(float),
            "dV(mV)": voltage_mV.diff().fillna(0.0).astype(float),
            "Power(W)": ((current_mA / 1000.0) * (voltage_mV / 1000.0)).astype(float),
            "RawCapacity(mAh)": used_mAh.astype(float),
            "T(C)": temp_c.astype(float),
            "SOC": soc.astype(float),
        }
    )
    return out, sim_total_mAh, tail_total_mAh


def process_file(
    path: Path,
    output_dir: Path,
    used_names: dict[str, int],
) -> tuple[list[dict], dict]:
    match = re.fullmatch(r"250075-(\d+)-(\d+)-(\d+)\.csv", path.name)
    if not match:
        return [], {}

    bank, unit, suffix = match.groups()
    cell_id = f"{bank}-{unit}"
    aging_suffix = int(suffix)

    df = read_raw_csv(path)
    blocks = contiguous_blocks(df)
    cap_test_mAh, cap_block_index, cap_temp_c = first_capacity_test(df, blocks)
    file_summary = {
        "source_file": path.name,
        "cell_id": cell_id,
        "aging_suffix": aging_suffix,
        "capacity_test_mAh": cap_test_mAh,
        "capacity_test_Ah": cap_test_mAh / 1000.0 if np.isfinite(cap_test_mAh) else np.nan,
        "capacity_test_block": cap_block_index,
        "capacity_test_temperature_C": cap_temp_c,
    }

    if not np.isfinite(cap_test_mAh):
        return [], file_summary

    manifest_rows: list[dict] = []
    sim_count = 0
    for pos, block in enumerate(blocks):
        if block.step_type != STEP_SIM:
            continue

        tail = find_tail_discharge(blocks, pos)
        if tail is None:
            continue

        sim_count += 1
        sim_df = block_frame(df, block)
        tail_df = block_frame(df, tail)
        segment_df, sim_mAh, tail_mAh = build_segment(sim_df, tail_df)

        temp_mean = float(segment_df["T(C)"].mean())
        temp_min = float(segment_df["T(C)"].min())
        temp_max = float(segment_df["T(C)"].max())
        cap_token = safe_float_token(cap_test_mAh, digits=1)
        cycle_cap_token = safe_float_token(float(segment_df["RawCapacity(mAh)"].iloc[-1]), digits=1)
        temp_token = rounded_temperature_token(temp_mean)
        base_out_name = (
            f"cell{cell_id}_idx{aging_suffix}_CAPtest{cap_token}mAh_"
            f"seg{sim_count:02d}_T{temp_token}C_Q{cycle_cap_token}mAh.csv"
        )
        out_name = unique_name(base_out_name, used_names)
        out_path = output_dir / "segments" / out_name
        write_csv_replace(segment_df, out_path)

        manifest_rows.append(
            {
                "segment_file": out_name,
                "source_file": path.name,
                "cell_id": cell_id,
                "aging_suffix": aging_suffix,
                "segment_index": sim_count,
                "temperature_mean_C": temp_mean,
                "temperature_min_C": temp_min,
                "temperature_max_C": temp_max,
                "temperature_bin": temperature_bin(temp_mean),
                "capacity_test_mAh": cap_test_mAh,
                "capacity_test_Ah": cap_test_mAh / 1000.0,
                "sim_discharge_mAh": sim_mAh,
                "tail_discharge_mAh": tail_mAh,
                "full_discharge_mAh": sim_mAh + tail_mAh,
                "sim_rows": len(sim_df),
                "tail_rows": len(tail_df),
                "total_rows": len(segment_df),
                "sim_block_index": block.index,
                "tail_block_index": tail.index,
                "soc_start": float(segment_df["SOC"].iloc[0]),
                "soc_end": float(segment_df["SOC"].iloc[-1]),
            }
        )

    file_summary["segment_count"] = sim_count
    return manifest_rows, file_summary


def add_soh(file_summary: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if file_summary.empty:
        return file_summary, manifest

    base = (
        file_summary.sort_values(["cell_id", "aging_suffix"])
        .groupby("cell_id", as_index=False)
        .first()[["cell_id", "capacity_test_mAh"]]
        .rename(columns={"capacity_test_mAh": "baseline_capacity_test_mAh"})
    )
    file_summary = file_summary.merge(base, on="cell_id", how="left")
    file_summary["soh"] = (
        file_summary["capacity_test_mAh"] / file_summary["baseline_capacity_test_mAh"]
    )

    if not manifest.empty:
        manifest = manifest.merge(
            file_summary[["source_file", "baseline_capacity_test_mAh", "soh"]],
            on="source_file",
            how="left",
        )
    return file_summary, manifest


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=workspace_root / "data" / "raw" / "soccc")
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_root / "data" / "processed" / "soccc_segments",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    segments_dir = output_dir / "segments"
    metadata_dir = output_dir / "metadata"
    segments_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    file_rows: list[dict] = []
    used_names: dict[str, int] = {}
    raw_files = sorted(input_dir.glob("250075-*-*-*.csv"))
    for index, path in enumerate(raw_files, start=1):
        try:
            rows, summary = process_file(path, output_dir, used_names)
        except Exception as exc:
            raise RuntimeError(f"Failed to process raw source: {path}") from exc
        manifest_rows.extend(rows)
        if summary:
            file_rows.append(summary)
        if index % 50 == 0:
            print(f"processed {index}/{len(raw_files)} files")

    manifest = pd.DataFrame(manifest_rows)
    file_summary = pd.DataFrame(file_rows)
    file_summary, manifest = add_soh(file_summary, manifest)

    write_csv_replace(
        file_summary.sort_values(["cell_id", "aging_suffix"]),
        metadata_dir / "file_capacity_summary.csv",
    )
    write_csv_replace(
        manifest.sort_values(["cell_id", "aging_suffix", "segment_index"]),
        metadata_dir / "segment_manifest.csv",
    )

    print(f"raw files: {len(raw_files)}")
    print(f"segments: {len(manifest)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
