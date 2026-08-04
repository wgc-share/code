from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE / "data" / "splits" / "segment_manifest.csv"
SEGMENT_DIR = WORKSPACE / "data" / "processed" / "soccc_segments" / "segments"
OUTPUT_DIR = WORKSPACE / "results" / "data_audit" / "condition_soc_curves_v2"

TEMP_ORDER = ["10C", "15-20C", "25C", "30-35C", "40C"]
TEMP_LABELS = {
    "10C": "10 C",
    "15-20C": "15-20 C",
    "25C": "25 C",
    "30-35C": "30-35 C",
    "40C": "40 C",
}
TEMP_COLORS = {
    "10C": "#0072B2",
    "15-20C": "#56B4E9",
    "25C": "#009E73",
    "30-35C": "#E69F00",
    "40C": "#D55E00",
}

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


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def condition_rows(manifest: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    groups = []
    random_cells = manifest["cell_id"].astype(str).str.match(r"^[12]-\d+$")
    for segment_index, condition in RANDOM_CONDITIONS.items():
        rows = manifest[
            random_cells & manifest["segment_index"].astype(int).eq(segment_index)
        ].copy()
        groups.append(("random", condition, rows))

    for (cell_id, segment_index), condition in FIXED_CONDITIONS.items():
        rows = manifest[
            manifest["cell_id"].astype(str).eq(cell_id)
            & manifest["segment_index"].astype(int).eq(segment_index)
        ].copy()
        groups.append(("fixed", condition, rows))
    return groups


def load_soc(segment_file: str) -> np.ndarray:
    frame = pd.read_csv(SEGMENT_DIR / segment_file, usecols=["SOC"])
    soc = pd.to_numeric(frame["SOC"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    if not np.isfinite(soc).all():
        raise ValueError(f"Non-finite SOC values: {segment_file}")
    return soc * 100.0


def plot_one(group_name: str, condition: str, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError(f"No rows found for {group_name}/{condition}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    trace_rows = []
    line_alpha = 0.13 if group_name == "random" else 0.38
    line_width = 0.55 if group_name == "random" else 0.9

    rows = rows.sort_values(["temperature_bin", "cell_id", "aging_suffix", "segment_index"])
    for _, row in rows.iterrows():
        temp_bin = str(row["temperature_bin"])
        if temp_bin not in TEMP_COLORS:
            continue
        segment_file = str(row["segment_file"])
        soc = load_soc(segment_file)
        elapsed_h = np.arange(len(soc), dtype=float) / 3600.0
        step = max(len(soc) // 1400, 1)
        ax.plot(
            elapsed_h[::step],
            soc[::step],
            color=TEMP_COLORS[temp_bin],
            alpha=line_alpha,
            linewidth=line_width,
        )

        sim_rows = int(row["sim_rows"])
        sim_discharge_mAh = float(row["sim_discharge_mAh"])
        capacity_mAh = float(row["capacity_test_mAh"])
        mean_current_a = sim_discharge_mAh * 3.6 / sim_rows
        mean_c_rate = mean_current_a / (capacity_mAh / 1000.0)
        trace_rows.append(
            {
                "group": group_name,
                "condition": condition,
                "cell_id": row["cell_id"],
                "segment_index": int(row["segment_index"]),
                "segment_file": segment_file,
                "source_file": row["source_file"],
                "aging_suffix": int(row["aging_suffix"]),
                "temperature_bin": temp_bin,
                "temperature_mean_C": float(row["temperature_mean_C"]),
                "soh": float(row["soh"]),
                "duration_h": len(soc) / 3600.0,
                "sim_duration_h": sim_rows / 3600.0,
                "mean_current_A": mean_current_a,
                "mean_c_rate": mean_c_rate,
                "soc_start_percent": float(soc[0]),
                "soc_end_percent": float(soc[-1]),
            }
        )

    ax.set_title(f"{condition} - all SOC trajectories (n={len(trace_rows)})", fontsize=12)
    ax.set_xlabel("Elapsed time (h)")
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(-2.0, 102.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.85)
    legend_handles = [
        Line2D([0], [0], color=TEMP_COLORS[temp], linewidth=2.0, label=TEMP_LABELS[temp])
        for temp in TEMP_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()

    group_dir = OUTPUT_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(condition)
    fig.savefig(group_dir / f"{stem}_all_soc.png", dpi=240, bbox_inches="tight")
    fig.savefig(group_dir / f"{stem}_all_soc.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(trace_rows)


def summarize(trace: pd.DataFrame) -> pd.DataFrame:
    return (
        trace.groupby(["group", "condition"], sort=False)
        .agg(
            curves=("segment_file", "count"),
            duration_h_min=("duration_h", "min"),
            duration_h_median=("duration_h", "median"),
            duration_h_max=("duration_h", "max"),
            mean_current_A_min=("mean_current_A", "min"),
            mean_current_A_median=("mean_current_A", "median"),
            mean_current_A_max=("mean_current_A", "max"),
            mean_c_rate_min=("mean_c_rate", "min"),
            mean_c_rate_median=("mean_c_rate", "median"),
            mean_c_rate_max=("mean_c_rate", "max"),
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    traces = []
    for group_name, condition, rows in condition_rows(manifest):
        trace = plot_one(group_name, condition, rows)
        traces.append(trace)
        print(f"Generated {group_name}/{condition}: {len(trace)} curves")

    all_trace = pd.concat(traces, ignore_index=True)
    all_trace.to_csv(OUTPUT_DIR / "condition_soc_curve_sources.csv", index=False, encoding="utf-8-sig")
    summarize(all_trace).to_csv(
        OUTPUT_DIR / "condition_soc_curve_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved 16 condition figures and audit tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
