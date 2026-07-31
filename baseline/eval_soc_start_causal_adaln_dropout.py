from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

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
from torch_io import load_torch_file
from train_utils import evaluate_dataset


def cache_name(split_name: str, window_size: int, stride: int) -> str:
    base = f"{split_name}_cache_causal_adaln_dropout"
    if window_size == 100 and stride == 100:
        return f"{base}.pt"
    return f"{base}_w{window_size}_s{stride}.pt"


class FilteredSamplesDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def filter_from_soc_start(dataset: BatteryTDGCMDataset, start_soc_percent: float) -> FilteredSamplesDataset:
    threshold = start_soc_percent / 100.0
    file_to_samples = defaultdict(list)
    for sample in dataset.samples:
        file_to_samples[sample["filenames"]].append(sample)

    filtered = []
    for filename in sorted(file_to_samples):
        samples = file_to_samples[filename]
        start_idx = None
        for idx, sample in enumerate(samples):
            window_start_soc = float(sample["soc"][0, 0])
            if window_start_soc <= threshold:
                start_idx = idx
                break
        if start_idx is not None:
            filtered.extend(samples[start_idx:])

    return FilteredSamplesDataset(filtered)


def load_latest_checkpoint(pth_dir: Path, checkpoint: str | None):
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        ckpts = list(pth_dir.glob("best_model_PINT_SchemeB_CAUSAL_ADALN_DROPOUT_V1_*.pth"))
        if not ckpts:
            raise FileNotFoundError(
                f"No baseline checkpoint found in {pth_dir}. Run train_causal_adaln_dropout.py first."
            )
        ckpt_path = max(ckpts, key=lambda path: path.stat().st_mtime)
    return ckpt_path, load_torch_file(ckpt_path, map_location="cpu")


def build_model(cfg: dict, device: str):
    model = AdaLNBatteryTDGCMModel(
        d_model=cfg["D_MODEL"],
        nhead=cfg["NHEAD"],
        num_layers=cfg["NUM_LAYERS"],
        dropout=cfg["DROPOUT"],
        use_causal=True,
    ).to(device)
    return model


def warn_if_not_mainline(cfg: dict):
    expected = {
        "LR": 2e-4,
        "DROPOUT": 0.1,
        "P_RESET": 0.05,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = cfg.get(key)
        if actual_value is None or abs(float(actual_value) - expected_value) > 1e-12:
            mismatches.append(f"{key}: checkpoint={actual_value}, expected={expected_value}")
    if mismatches:
        print("WARNING: checkpoint hyperparameters do not match the current mainline:")
        for item in mismatches:
            print(f"  - {item}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the mainline baseline from different SOC start points.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to the latest mainline checkpoint.")
    parser.add_argument("--starts", default="100,90,80,70,60,50,40,30,20,10", help="SOC start percentages.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size. Use 1 for strict sequential testing.")
    parser.add_argument("--show-progress", action="store_true", help="Show tqdm progress bars during evaluation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = str(processed_segments_dir())
    split_file = str(split_file_path())
    results_dir = baseline_results_dir()
    pth_dir = results_dir / "pth_save"
    pt_dir = results_dir / "pt"
    csv_dir = results_dir / "csv_save"
    pt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, ckpt = load_latest_checkpoint(pth_dir, args.checkpoint)
    cfg = ckpt["config"]
    warn_if_not_mainline(cfg)
    window_size = cfg.get("WINDOW_SIZE", 100)
    stride = cfg.get("STRIDE", 100)

    _, _, test_random_f, test_fixed_f = split_soccc_by_cells(data_dir, split_file)
    test_random_ds = BatteryTDGCMDataset(
        data_dir,
        test_random_f,
        window_size=window_size,
        stride=stride,
        cache_file=os.path.join(pt_dir, cache_name("test_random", window_size, stride)),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        data_dir,
        test_fixed_f,
        window_size=window_size,
        stride=stride,
        cache_file=os.path.join(pt_dir, cache_name("test_fixed", window_size, stride)),
    )

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]

    starts = [float(item.strip()) for item in args.starts.split(",") if item.strip()]
    run_id = datetime.now().strftime("soc_start_%m%d_%H%M%S")
    rows = []

    print("=== SOC-start evaluation ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print(f"DEVICE     : {device}")
    print(f"BATCH_SIZE : {args.batch_size}")
    print(f"STARTS     : {starts}")

    for split_name, base_dataset in (("test_random", test_random_ds), ("test_fixed", test_fixed_ds)):
        for start_soc in starts:
            filtered_ds = filter_from_soc_start(base_dataset, start_soc)
            if len(filtered_ds) == 0:
                rows.append(
                    {
                        "split": split_name,
                        "soc_start_percent": start_soc,
                        "avg_mae": float("nan"),
                        "files": 0,
                        "windows": 0,
                        "batch_size": args.batch_size,
                    }
                )
                continue

            loader = DataLoader(filtered_ds, batch_size=args.batch_size, shuffle=False)
            avg_mae, file_maes = evaluate_dataset(
                model,
                loader,
                scaler,
                {
                    "DEVICE": device,
                    "D_MODEL": cfg["D_MODEL"],
                    "DISABLE_TQDM": not args.show_progress,
                },
                label=f"{split_name} SOC<= {start_soc:g}%",
            )
            rows.append(
                {
                    "split": split_name,
                    "soc_start_percent": start_soc,
                    "avg_mae": avg_mae,
                    "files": len(file_maes),
                    "windows": len(filtered_ds),
                    "batch_size": args.batch_size,
                }
            )

    out_path = csv_dir / f"soc_start_metrics_{run_id}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== SOC-start metrics ===")
    print(df.to_string(index=False))
    print(f"Metrics saved: {out_path}")


if __name__ == "__main__":
    main()
