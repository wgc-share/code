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
from project_paths import baseline_results_dir, processed_segments_dir, split_file_path
from scaling import PITDScaler
from soccc_schemeB import BatteryTDGCMDataset, split_soccc_by_cells
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, evaluate_dataset, set_seed
from torch_io import load_torch_file, save_torch_file


def _make_results_dirs():
    base = baseline_results_dir()
    cache_dir = base / "pt"
    pth_dir = base / "pth_save"
    csv_dir = base / "csv_save"
    meta_dir = base / "metadata"
    for d in (cache_dir, pth_dir, csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cache_dir, pth_dir, csv_dir, meta_dir


def train_causal_adaln_dropout():
    set_seed(42)
    data_dir = processed_segments_dir()
    split_file = split_file_path()
    cache_dir, pth_dir, csv_dir, meta_dir = _make_results_dirs()

    config = {
        "PROJECT": "PINT_SchemeB_CAUSAL_ADALN_DROPOUT_V1",
        "DATA_DIR": str(data_dir),
        "SPLIT_FILE": str(split_file),
        "CACHE_DIR": str(cache_dir),
        "PTH_DIR": str(pth_dir),
        "CSV_DIR": str(csv_dir),
        "META_DIR": str(meta_dir),
        "BATCH_SIZE": 64,
        "VAL_BATCH_SIZE": 64,
        "LR": 1e-4,
        "EPOCHS": 50,
        "D_MODEL": 64,
        "NHEAD": 4,
        "NUM_LAYERS": 2,
        "DROPOUT": 0.1,
        "LAMBDA_AH_START": 10.0,
        "LAMBDA_AH_STEP": 20.0,
        "GRAD_CLIP": 1.0,
        "P_RESET": 0.05,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "WINDOW_SIZE": 100,
        "STRIDE": 100,
    }

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print("=== SchemeB causal + AdaLN + dropout run ===")
    print(f"DATA_DIR   : {config['DATA_DIR']}")
    print(f"SPLIT_FILE : {config['SPLIT_FILE']}")
    print(f"CACHE_DIR  : {config['CACHE_DIR']}")
    print(f"PTH_DIR    : {config['PTH_DIR']}")
    print(f"CSV_DIR    : {config['CSV_DIR']}")
    print(f"DEVICE     : {config['DEVICE']}")
    print(f"BATCH_SIZE : {config['BATCH_SIZE']}")
    print(f"VAL_BATCH  : {config['VAL_BATCH_SIZE']} (deterministic stateful lanes)")
    print(f"P_RESET    : {config['P_RESET']}")
    print(f"WINDOW     : {config['WINDOW_SIZE']}")
    print(f"STRIDE     : {config['STRIDE']}")
    print("TRANSFORMER: causal mask enabled")
    print("TEMP MOD   : AdaLN")

    train_f, val_f, test_random_f, test_fixed_f = split_soccc_by_cells(config["DATA_DIR"], config["SPLIT_FILE"])
    print(
        f"Split counts | train={len(train_f)} | val={len(val_f)} | "
        f"test_random={len(test_random_f)} | test_fixed={len(test_fixed_f)}"
    )
    split_lists = {
        "train": train_f,
        "val": val_f,
        "test_random": test_random_f,
        "test_fixed": test_fixed_f,
    }
    empty_splits = [name for name, files in split_lists.items() if not files]
    if empty_splits:
        raise RuntimeError(f"Empty data split(s): {', '.join(empty_splits)}")

    train_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], train_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "train_cache_causal_adaln_dropout.pt"),
    )
    val_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], val_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "val_cache_causal_adaln_dropout.pt"),
    )
    test_random_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], test_random_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_random_cache_causal_adaln_dropout.pt"),
    )
    test_fixed_ds = BatteryTDGCMDataset(
        config["DATA_DIR"], test_fixed_f, window_size=config["WINDOW_SIZE"], stride=config["STRIDE"],
        cache_file=os.path.join(config["CACHE_DIR"], "test_fixed_cache_causal_adaln_dropout.pt"),
    )

    val_loader = DataLoader(val_ds, batch_size=config["VAL_BATCH_SIZE"], shuffle=False)
    test_random_loader = DataLoader(test_random_ds, batch_size=1, shuffle=False)
    test_fixed_loader = DataLoader(test_fixed_ds, batch_size=1, shuffle=False)

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
        manager = BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
        h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
        pbar = tqdm(total=manager.total_steps, desc=f"Epoch {epoch+1}/{config['EPOCHS']} Training", colour="blue")

        curr_lambda = config["LAMBDA_AH_START"] + epoch * config["LAMBDA_AH_STEP"]
        print(
            f"[Epoch {epoch+1}] start | lambda_ah={curr_lambda:.2f} | "
            f"train_windows={len(train_ds)} | val_windows={len(val_ds)} | steps={manager.total_steps} | "
            f"p_reset={config['P_RESET']:.3f}"
        )

        epoch_train_losses = []
        epoch_data_losses = []
        epoch_ah_losses = []
        epoch_range_losses = []
        epoch_reset_lanes = 0
        epoch_reset_batches = 0

        while True:
            indices, masks, is_first, finished = manager.get_next_batch()
            if finished:
                break

            samples = [train_ds[i] for i in indices]
            x = torch.stack([s["x_dyn"] for s in samples]).to(config["DEVICE"])
            t = torch.stack([s["t_mean"] for s in samples]).to(config["DEVICE"])
            y = torch.stack([s["soc"] for s in samples]).to(config["DEVICE"])
            Q = torch.stack([s["Q"] for s in samples]).to(config["DEVICE"])
            m_t = torch.tensor(masks, dtype=torch.float32, device=config["DEVICE"])
            f_t = torch.tensor(is_first, dtype=torch.bool, device=config["DEVICE"])

            h_state = h_state.detach()
            if f_t.any():
                for i in range(config["BATCH_SIZE"]):
                    if f_t[i]:
                        h_state[:, i, :] = 0.0

            active_mask = m_t > 0.5
            eligible_mask = active_mask & (~f_t)
            reset_mask = (torch.rand(config["BATCH_SIZE"], device=config["DEVICE"]) < config["P_RESET"]) & eligible_mask
            reset_count = int(reset_mask.sum().item())
            if reset_count > 0:
                h_state[:, reset_mask, :] = 0.0
                epoch_reset_lanes += reset_count
                epoch_reset_batches += 1

            x_n, t_n = scaler.transform(x, t)
            y_p, h_state = model(x_n, t_n, h_state)
            loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], Q, f_t, curr_lambda, m_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_data_losses.append(l_d.item())
            epoch_ah_losses.append(l_ah.item())
            epoch_range_losses.append(l_range.item())

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Reset": reset_count})

        pbar.close()

        avg_val_mae, file_maes = evaluate_dataset(model, val_loader, scaler, {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]}, label=f"Epoch {epoch+1} Val")
        scheduler.step()
        for fn, mae in file_maes.items():
            val_mae_matrix[fn].append(mae)

        history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan"),
                "Train_Data_Loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan"),
                "Train_Ah_Loss": float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan"),
                "Train_Range_Loss": float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan"),
                "Val_MAE_Avg": avg_val_mae,
                "Lambda": curr_lambda,
                "Reset_Lanes": epoch_reset_lanes,
                "Reset_Batches": epoch_reset_batches,
            }
        )

        pd.DataFrame(history).to_csv(os.path.join(config["CSV_DIR"], f"log_{run_id}.csv"), index=False)
        pd.DataFrame(val_mae_matrix).T.to_csv(os.path.join(config["CSV_DIR"], f"val_matrix_{run_id}.csv"))

        train_total = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        train_data = float(np.mean(epoch_data_losses)) if epoch_data_losses else float("nan")
        train_ah = float(np.mean(epoch_ah_losses)) if epoch_ah_losses else float("nan")
        train_range = float(np.mean(epoch_range_losses)) if epoch_range_losses else float("nan")
        print(
            f"[Epoch {epoch+1}] summary | "
            f"train_total={train_total:.6f} | "
            f"train_data={train_data:.6f} | "
            f"train_ah={train_ah:.6f} | "
            f"train_range={train_range:.6f} | "
            f"val_mae={avg_val_mae:.4f} | lambda_ah={curr_lambda:.2f} | "
            f"reset_lanes={epoch_reset_lanes} | reset_batches={epoch_reset_batches}"
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
            print(f"[Epoch {epoch+1}] new best model saved: val_mae={best_overall_mae:.4f}")

    print(f"Training finished. Best val MAE = {best_overall_mae:.4f}")

    if not os.path.exists(best_model_path):
        raise RuntimeError("Training completed without a valid validation checkpoint")
    best_ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Loaded best checkpoint for final evaluation.")

    evaluate_dataset(model, test_random_loader, scaler, {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]}, label="Test random")
    evaluate_dataset(model, test_fixed_loader, scaler, {"DEVICE": config["DEVICE"], "D_MODEL": config["D_MODEL"]}, label="Test fixed")


if __name__ == "__main__":
    train_causal_adaln_dropout()
