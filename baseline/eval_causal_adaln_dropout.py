from __future__ import annotations

import os
import sys
from pathlib import Path

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
from train_utils import evaluate_dataset
from torch_io import load_torch_file


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = str(processed_segments_dir())
    split_file = str(split_file_path())
    results_dir = baseline_results_dir()
    pth_dir = results_dir / "pth_save"
    pt_dir = results_dir / "pt"
    csv_dir = results_dir / "csv_save"
    pt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    ckpts = list(pth_dir.glob("best_model_PINT_SchemeB_CAUSAL_ADALN_DROPOUT_V1_*.pth"))
    if not ckpts:
        raise FileNotFoundError(
            f"No baseline checkpoint found in {pth_dir}. Run train_causal_adaln_dropout.py first."
        )
    ckpt_path = max(ckpts, key=lambda path: path.stat().st_mtime)
    ckpt = load_torch_file(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(data_dir, split_file)
    window_size = cfg.get("WINDOW_SIZE", 100)
    stride = cfg.get("STRIDE", 100)
    test_random_ds = BatteryTDGCMDataset(data_dir, test_random_f, window_size=window_size, stride=stride, cache_file=os.path.join(pt_dir, "test_random_cache_causal_adaln_dropout.pt"))
    test_fixed_ds = BatteryTDGCMDataset(data_dir, test_fixed_f, window_size=window_size, stride=stride, cache_file=os.path.join(pt_dir, "test_fixed_cache_causal_adaln_dropout.pt"))

    test_random_loader = DataLoader(test_random_ds, batch_size=1, shuffle=False)
    test_fixed_loader = DataLoader(test_fixed_ds, batch_size=1, shuffle=False)

    model = AdaLNBatteryTDGCMModel(
        d_model=cfg["D_MODEL"],
        nhead=cfg["NHEAD"],
        num_layers=cfg["NUM_LAYERS"],
        dropout=cfg["DROPOUT"],
        use_causal=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]

    eval_run_id = f"eval_{ckpt_path.stem.replace('best_model_', '')}"
    evaluate_dataset(
        model,
        test_random_loader,
        scaler,
        {"DEVICE": device, "D_MODEL": cfg["D_MODEL"]},
        label="Test random",
        output_dir=csv_dir,
        run_id=eval_run_id,
    )
    evaluate_dataset(
        model,
        test_fixed_loader,
        scaler,
        {"DEVICE": device, "D_MODEL": cfg["D_MODEL"]},
        label="Test fixed",
        output_dir=csv_dir,
        run_id=eval_run_id,
    )


if __name__ == "__main__":
    main()
