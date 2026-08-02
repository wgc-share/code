from __future__ import annotations

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

shared_path = str(SHARED_DIR)
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from project_paths import organized_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, split_soccc_by_cells
from torch_io import load_torch_file, save_torch_file
from train_utils import PITDPhysicsLoss, set_seed


def _make_results_dirs():
    base = organized_results_dir() / "ablation" / "no_hidden_state"
    cache_dir = base / "pt"
    pth_dir = base / "pth_save"
    csv_dir = base / "csv_save"
    meta_dir = base / "metadata"
    for directory in (cache_dir, pth_dir, csv_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def evaluate_independent_dataset(model, loader: DataLoader, scaler: PITDScaler, config: dict, label: str):
    if len(loader.dataset) == 0:
        raise ValueError(f"Cannot evaluate empty dataset: {label}")

    model.eval()
    file_errors = defaultdict(list)

    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc=f"{label} eval", leave=False, colour="green")
        for batch in loader:
            x = batch["x_dyn"].to(config["DEVICE"])
            t = batch["t_mean"].to(config["DEVICE"])
            y = batch["soc"].to(config["DEVICE"])
            filenames = _as_list(batch["filenames"])

            x_n, t_n = scaler.transform(x, t)
            y_p, _ = model(x_n, t_n, h_prev=None)
            sample_maes = torch.abs(y_p - y).mean(dim=(1, 2)).detach().cpu().numpy()

            for filename, mae in zip(filenames, sample_maes):
                file_errors[str(filename)].append(float(mae))
            pbar.update(1)
        pbar.close()

    file_maes = {filename: float(np.mean(errors)) for filename, errors in file_errors.items()}
    avg_mae = float(np.mean(list(file_maes.values()))) if file_maes else float("nan")
    worst = sorted(file_maes.items(), key=lambda item: item[1], reverse=True)[:3]
    print(f"[{label}] files={len(file_maes)} batch={loader.batch_size} avg_mae={avg_mae:.6f}")
    if worst:
        print(f"[{label}] worst files: " + ", ".join(f"{filename}:{mae:.6f}" for filename, mae in worst))
    return avg_mae, file_maes


def save_eval_metrics(label: str, avg_mae: float, file_maes: dict, csv_dir: str, run_id: str):
    label_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    summary_path = Path(csv_dir) / f"{label_name}_summary_{run_id}.csv"
    file_path = Path(csv_dir) / f"{label_name}_file_mae_{run_id}.csv"

    pd.DataFrame(
        [
            {
                "label": label,
                "avg_mae": avg_mae,
                "files": len(file_maes),
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"filename": filename, "mae": mae} for filename, mae in sorted(file_maes.items())]
    ).to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"[{label}] metrics saved: {summary_path}")
    print(f"[{label}] file MAE saved: {file_path}")


def train_causal_adaln_no_hidden_state():
    set_seed(42)
    data_dir = processed_segments_dir()
    split_file = split_file_path()
    cache_dir, pth_dir, csv_dir, meta_dir = _make_results_dirs()

    config = {
        "PROJECT": "PINT_SchemeB_CAUSAL_ADALN_NO_HIDDEN_STATE",
        "DATA_DIR": str(data_dir),
        "SPLIT_FILE": str(split_file),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "BATCH_SIZE": 64,
        "VAL_BATCH_SIZE": 64,
        "TEST_BATCH_SIZE": 64,
        "LR": 2e-4,
        "EPOCHS": 50,
        "D_MODEL": 64,
        "NHEAD": 4,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10.0,
        "LAMBDA_AH_STEP": 20.0,
        "GRAD_CLIP": 1.0,
        "HIDDEN_STATE_MODE": "none",
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
    }

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print("=== Ablation: causal + AdaLN, no hidden-state passing ===")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR  : {config['CACHE_DIR']}")
    print(f"PTH_DIR    : {config['PTH_DIR']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"BATCH_SIZE : {config['BATCH_SIZE']} (shuffle=True)")
    print(f"VAL_BATCH  : {config['VAL_BATCH_SIZE']} (independent windows)")
    print(f"TEST_BATCH : {config['TEST_BATCH_SIZE']} (independent windows)")
    print(f"LR         : {config['LR']}")
    print(f"DROPOUT    : {config['DROPOUT']}")
    print("TRANSFORMER: causal mask enabled")
    print("TEMP MOD   : AdaLN")
    print("H_STATE    : no cross-window passing")

    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    print(
        f"Split counts | train={len(train_f)} | val={len(val_f)} | "
        f"test_random={len(test_random_f)} | test_fixed={len(test_fixed_f)}"
    )

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        train_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "train_cache_causal_adaln_no_hidden_state.pt"),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        val_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "val_cache_causal_adaln_no_hidden_state.pt"),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_random_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_random_cache_causal_adaln_no_hidden_state.pt"),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"],
        test_fixed_f,
        window_size=config["WINDOW_SIZE"],
        stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_fixed_cache_causal_adaln_no_hidden_state.pt"),
    )

    train_loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    test_random_loader = DataLoader(test_random_ds, batch_size=config["TEST_BATCH_SIZE"], shuffle=False)
    test_fixed_loader = DataLoader(test_fixed_ds, batch_size=config["TEST_BATCH_SIZE"], shuffle=False)

    scaler = PITDScaler()
    scaler.fit(train_ds)

    model = AdaLNBatteryTDGCMModel(
        d_model=config["D_MODEL"],
        nhead=config["NHEAD"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        use_causal=True,
    ).to(config["DEVICE"])

    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-5)
    criterion = PITDPhysicsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])

    best_overall_mae = float("inf")
    best_model_path = os.path.join(config["PTH_DIR"], f"best_model_{run_id}.pth")
    history = []
    val_mae_matrix = defaultdict(list)

    for epoch in range(config["EPOCHS"]):
        model.train()
        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]
        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{config['EPOCHS']} Training", colour="blue")
        for batch in train_loader:
            x = batch["x_dyn"].to(config["DEVICE"])
            t = batch["t_mean"].to(config["DEVICE"])
            y = batch["soc"].to(config["DEVICE"])
            q = batch["Q"].to(config["DEVICE"])
            is_first = batch["is_first"].to(config["DEVICE"]).bool()
            mask = torch.ones(x.size(0), dtype=torch.float32, device=config["DEVICE"])

            x_n, t_n = scaler.transform(x, t)
            y_p, _ = model(x_n, t_n, h_prev=None)
            loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], q, is_first, curr_lambda, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_data_losses.append(l_d.item())
            epoch_ah_losses.append(l_ah.item())
            epoch_range_losses.append(l_range.item())

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        pbar.close()

        avg_val_mae, file_maes = evaluate_independent_dataset(
            model,
            val_loader,
            scaler,
            {"DEVICE": config["DEVICE"]},
            label=f"Epoch {epoch + 1} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        row = {
            "Epoch": epoch + 1,
            "Train_Loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
            "Train_Data_Loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan"),
            "Train_Ah_Loss": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
            "Train_Range_Loss": float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan"),
            "Val_MAE_Avg": avg_val_mae,
            "Lambda": curr_lambda,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(config["CSV_DIR"], f"log_{run_id}.csv"), index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(os.path.join(config["CSV_DIR"], f"val_matrix_{run_id}.csv"))

        print(
            f"[Epoch {epoch + 1}] summary | "
            f"train_total={row['Train_Loss']:.6f} | "
            f"train_data={row['Train_Data_Loss']:.6f} | "
            f"train_ah={row['Train_Ah_Loss']:.6f} | "
            f"train_range={row['Train_Range_Loss']:.6f} | "
            f"val_mae={avg_val_mae:.6f} | lambda_ah={curr_lambda:.2f}"
        )

        if avg_val_mae < best_overall_mae:
            best_overall_mae = avg_val_mae
            save_torch_file(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_stats": scaler.stats,
                    "config": config,
                },
                best_model_path,
            )
            print(f"[Epoch {epoch + 1}] new best model saved: val_mae={best_overall_mae:.6f}")

    print(f"Training finished. Best val MAE = {best_overall_mae:.6f}")

    if not os.path.exists(best_model_path):
        raise RuntimeError("Training completed without a valid validation checkpoint")
    best_ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best checkpoint for final independent-window evaluation.")

    test_random_mae, test_random_file_maes = evaluate_independent_dataset(
        model,
        test_random_loader,
        scaler,
        {"DEVICE": config["DEVICE"]},
        label="Test random",
    )
    save_eval_metrics("Test random", test_random_mae, test_random_file_maes, config["CSV_DIR"], run_id)

    test_fixed_mae, test_fixed_file_maes = evaluate_independent_dataset(
        model,
        test_fixed_loader,
        scaler,
        {"DEVICE": config["DEVICE"]},
        label="Test fixed",
    )
    save_eval_metrics("Test fixed", test_fixed_mae, test_fixed_file_maes, config["CSV_DIR"], run_id)


if __name__ == "__main__":
    train_causal_adaln_no_hidden_state()
