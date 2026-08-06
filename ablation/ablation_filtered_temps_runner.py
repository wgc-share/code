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
from ablation_models import (
    AdaLNNoStateTransferModel,
    FiLMCausalTransformerModel,
    TempInputAdaLNCausalTransformerModel,
    TempInputCausalTransformerModel,
    TempInputGatedCausalTransformerModel,
    TempInputResidualAdapterCausalTransformerModel,
)
from adaln_model import BatteryTDGCMModel as AdaLNBatteryTDGCMModel
from scaling import PITDScaler
from torch_io import load_torch_file, save_torch_file
from train_utils import BalancedSchemeBManager, PITDPhysicsLoss, set_seed


ABLATIONS = {
    "no_adaln_temp_input": {
        "project": "PINT_ABLATION_NO_ADALN_TEMP_INPUT_D96_H4_L3",
        "result_dir": "no_adaln_temp_input_d96h4l3",
        "title": "Ablation: no AdaLN, temperature appended to input",
        "model": "temp_input",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": True,
    },
    "no_adaln_temp_input_no_ah": {
        "project": "PINT_ABLATION_NO_ADALN_TEMP_INPUT_NO_AH_D96_H4_L3",
        "result_dir": "no_adaln_temp_input_no_ah_d96h4l3",
        "title": "Ablation: no AdaLN, temperature appended to input, no Ah loss",
        "model": "temp_input",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
    },
    "temp_input_adaln_no_ah": {
        "project": "PINT_ABLATION_TEMP_INPUT_ADALN_NO_AH_D96_H4_L3",
        "result_dir": "temp_input_adaln_no_ah_d96h4l3",
        "title": "Ablation: temperature appended to input + AdaLN modulation, no Ah loss",
        "model": "temp_input_adaln",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
    },
    "temp_input_adapter_no_ah": {
        "project": "PINT_TEMPERATURE_INPUT_ADAPTER_NO_AH_D96_H4_L3",
        "result_dir": "temperature_temp_input_adapter_no_ah_d96h4l3",
        "title": "Temperature method: temp input + temperature residual adapter, no Ah loss",
        "model": "temp_input_adapter",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
    },
    "temp_input_gate_no_ah": {
        "project": "PINT_TEMPERATURE_INPUT_GATE_NO_AH_D96_H4_L3",
        "result_dir": "temperature_temp_input_gate_no_ah_d96h4l3",
        "title": "Temperature method: temp input + temperature gated Transformer output, no Ah loss",
        "model": "temp_input_gate",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
    },
    "film": {
        "project": "PINT_ABLATION_FILM_D96_H4_L3",
        "result_dir": "film_d96h4l3",
        "title": "Ablation: FiLM temperature modulation",
        "model": "film",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": True,
    },
    "no_state_transfer": {
        "project": "PINT_ABLATION_NO_STATE_TRANSFER_D96_H4_L3",
        "result_dir": "no_state_transfer_d96h4l3",
        "title": "Ablation: no GRU state transfer between windows",
        "model": "adaln_no_state",
        "stateful": False,
        "p_reset": 0.0,
        "use_ah_loss": True,
    },
    "no_state_transfer_no_ah": {
        "project": "PINT_ABLATION_NO_STATE_TRANSFER_NO_AH_D96_H4_L3",
        "result_dir": "no_state_transfer_no_ah_d96h4l3",
        "title": "Ablation: no GRU state transfer between windows, no Ah loss",
        "model": "adaln_no_state",
        "stateful": False,
        "p_reset": 0.0,
        "use_ah_loss": False,
    },
    "preset0": {
        "project": "PINT_ABLATION_PRESET0_D96_H4_L3",
        "result_dir": "preset0_d96h4l3",
        "title": "Ablation: state transfer without hidden-state dropout",
        "model": "adaln",
        "stateful": True,
        "p_reset": 0.0,
        "use_ah_loss": True,
    },
    "preset0_no_ah": {
        "project": "PINT_ABLATION_PRESET0_NO_AH_D96_H4_L3",
        "result_dir": "preset0_no_ah_d96h4l3",
        "title": "Ablation: state transfer without hidden-state dropout, no Ah loss",
        "model": "adaln",
        "stateful": True,
        "p_reset": 0.0,
        "use_ah_loss": False,
    },
    "no_ahloss": {
        "project": "PINT_ABLATION_NO_AHLOSS_D96_H4_L3",
        "result_dir": "no_ahloss_d96h4l3",
        "title": "Ablation: no ampere-hour consistency loss",
        "model": "adaln",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
    },
    "soc_monotonic": {
        "project": "PINT_ABLATION_SOC_MONOTONIC_D96_H4_L3",
        "result_dir": "soc_monotonic_d96h4l3",
        "title": "Ablation: discharge-aware SOC monotonicity constraint",
        "model": "adaln",
        "stateful": True,
        "p_reset": 0.05,
        "use_ah_loss": False,
        "use_soc_monotonic_loss": True,
        "lambda_soc_mono": 0.01,
        "soc_mono_segments": 5,
        "soc_mono_current_thr": 0.0,
        "soc_mono_margin": 0.0,
    },
}


def make_results_dirs(result_dir: str) -> dict[str, Path]:
    run_root = base.organized_results_dir() / "ablation" / result_dir
    dirs = {
        "run_root": run_root,
        "cache_dir": run_root / "pt",
        "pth_dir": run_root / "pth_save",
        "csv_dir": run_root / "csv_save",
        "meta_dir": run_root / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_config(ablation_name: str, args: argparse.Namespace | None = None) -> dict:
    spec = ABLATIONS[ablation_name]
    dirs = make_results_dirs(spec["result_dir"])
    cfg = base.make_config()
    cfg.update(
        {
            "PROJECT": spec["project"],
            "ABLATION": ablation_name,
            "ABLATION_TITLE": spec["title"],
            "MODEL_KIND": spec["model"],
            "STATEFUL_TRAINING": bool(spec["stateful"]),
            "P_RESET": float(spec["p_reset"]),
            "USE_AH_LOSS": bool(spec["use_ah_loss"]),
            "USE_SOC_MONOTONIC_LOSS": bool(spec.get("use_soc_monotonic_loss", False)),
            "LAMBDA_SOC_MONO": float(spec.get("lambda_soc_mono", 0.0)),
            "SOC_MONO_SEGMENTS": int(spec.get("soc_mono_segments", 5)),
            "SOC_MONO_CURRENT_THR": float(spec.get("soc_mono_current_thr", 0.0)),
            "SOC_MONO_MARGIN": float(spec.get("soc_mono_margin", 0.0)),
            "CACHE_DIR": str(dirs["cache_dir"]),
            "PTH_DIR": str(dirs["pth_dir"]),
            "CSV_DIR": str(dirs["csv_dir"]),
            "META_DIR": str(dirs["meta_dir"]),
            "D_MODEL": 96,
            "NHEAD": 4,
            "NUM_LAYERS": 3,
            "DROPOUT": 0.1,
        }
    )
    if args is not None:
        cfg["EPOCHS"] = int(args.epochs)
        cfg["BATCH_SIZE"] = int(args.batch_size)
        cfg["VAL_BATCH_SIZE"] = int(args.val_batch_size)
        cfg["SOC_BATCH_SIZE"] = int(args.soc_batch_size)
        cfg["DEVICE"] = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        cfg["DISABLE_TQDM"] = bool(args.disable_tqdm)
        if args.lambda_soc_mono is not None:
            cfg["LAMBDA_SOC_MONO"] = float(args.lambda_soc_mono)
        if args.soc_mono_segments is not None:
            cfg["SOC_MONO_SEGMENTS"] = int(args.soc_mono_segments)
        if args.soc_mono_current_thr is not None:
            cfg["SOC_MONO_CURRENT_THR"] = float(args.soc_mono_current_thr)
        if args.soc_mono_margin is not None:
            cfg["SOC_MONO_MARGIN"] = float(args.soc_mono_margin)
    return cfg


def print_config(config: dict):
    print(f"=== {config['ABLATION_TITLE']} ===")
    base.print_config(config)
    print(f"ABLATION   : {config['ABLATION']}")
    print(f"MODEL_KIND : {config['MODEL_KIND']}")
    print(f"STATEFUL   : {config['STATEFUL_TRAINING']}")
    print(f"USE_AH_LOSS: {config['USE_AH_LOSS']}")
    print(f"USE_SOC_MONO: {config.get('USE_SOC_MONOTONIC_LOSS', False)}")
    if config.get("USE_SOC_MONOTONIC_LOSS", False):
        print(
            f"SOC_MONO   : lambda={config['LAMBDA_SOC_MONO']} | "
            f"segments={config['SOC_MONO_SEGMENTS']} | "
            f"current_thr={config['SOC_MONO_CURRENT_THR']} | margin={config['SOC_MONO_MARGIN']}"
        )
    sys.stdout.flush()


def build_model(config: dict):
    kwargs = {
        "d_model": config["D_MODEL"],
        "nhead": config["NHEAD"],
        "num_layers": config["NUM_LAYERS"],
        "dropout": config["DROPOUT"],
    }
    kind = config["MODEL_KIND"]
    if kind == "adaln":
        return AdaLNBatteryTDGCMModel(**kwargs, use_causal=True).to(config["DEVICE"])
    if kind == "adaln_no_state":
        return AdaLNNoStateTransferModel(**kwargs, use_causal=True).to(config["DEVICE"])
    if kind == "temp_input":
        return TempInputCausalTransformerModel(**kwargs).to(config["DEVICE"])
    if kind == "temp_input_adaln":
        return TempInputAdaLNCausalTransformerModel(**kwargs).to(config["DEVICE"])
    if kind == "temp_input_adapter":
        return TempInputResidualAdapterCausalTransformerModel(**kwargs).to(config["DEVICE"])
    if kind == "temp_input_gate":
        return TempInputGatedCausalTransformerModel(**kwargs).to(config["DEVICE"])
    if kind == "film":
        return FiLMCausalTransformerModel(**kwargs).to(config["DEVICE"])
    raise ValueError(f"Unknown MODEL_KIND: {kind}")


def latest_checkpoint(pth_dir: Path, project: str) -> Path:
    ckpts = list(pth_dir.glob(f"best_model_{project}_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for {project} in {pth_dir}.")
    return max(ckpts, key=lambda path: path.stat().st_mtime)


def soc_monotonic_loss(
    soc_pred,
    current_raw,
    mask,
    segments: int = 5,
    current_thr: float = 0.0,
    margin: float = 0.0,
):
    if segments <= 0:
        return soc_pred.new_tensor(0.0)

    batch_size, seq_len, _ = soc_pred.shape
    terms = []
    for seg_idx in range(segments):
        start = int(seq_len * seg_idx / segments)
        end = int(seq_len * (seg_idx + 1) / segments)
        if end <= start + 1:
            continue
        i_mean = current_raw[:, start:end].mean(dim=1)
        discharge = i_mean > current_thr
        soc_start = soc_pred[:, start, 0]
        soc_end = soc_pred[:, end - 1, 0]
        violation = torch.relu(soc_end - soc_start + margin)
        terms.append(violation * discharge.to(dtype=violation.dtype))

    if not terms:
        return soc_pred.new_tensor(0.0)

    per_sample = torch.stack(terms, dim=1).mean(dim=1)
    return (per_sample * mask).sum() / (mask.sum() + 1e-9)


def apply_optional_soc_monotonic_loss(loss, y_p, x_current_raw, mask, config, loss_bucket):
    if not config.get("USE_SOC_MONOTONIC_LOSS", False):
        loss_bucket.append(0.0)
        return loss

    mono = soc_monotonic_loss(
        y_p,
        x_current_raw,
        mask,
        segments=config["SOC_MONO_SEGMENTS"],
        current_thr=config["SOC_MONO_CURRENT_THR"],
        margin=config["SOC_MONO_MARGIN"],
    )
    loss_bucket.append(float(mono.detach().item()))
    return loss + config["LAMBDA_SOC_MONO"] * mono


def train_one_epoch_stateful(model, train_ds, scaler, optimizer, criterion, config, epoch: int):
    model.train()
    manager = BalancedSchemeBManager(train_ds, config["BATCH_SIZE"])
    h_state = torch.zeros(1, config["BATCH_SIZE"], config["D_MODEL"], device=config["DEVICE"])
    pbar = tqdm(
        total=manager.total_steps,
        desc=f"Epoch {epoch}/{config['EPOCHS']} Training",
        colour="blue",
        disable=config.get("DISABLE_TQDM", False),
    )
    curr_lambda = 0.0 if not config["USE_AH_LOSS"] else config["LAMBDA_AH_START"] + (epoch - 1) * config["LAMBDA_AH_STEP"]
    losses = defaultdict(list)
    reset_lanes = 0
    reset_batches = 0
    print(
        f"[Epoch {epoch}/{config['EPOCHS']}] start | "
        f"mode=stateful | train_windows={len(train_ds)} | steps={manager.total_steps} | "
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
        reset_count = int(reset_mask.sum().item())
        if reset_count > 0:
            h_state[:, reset_mask, :] = 0.0
            reset_lanes += reset_count
            reset_batches += 1

        x_n, t_n = scaler.transform(x, t)
        y_p, h_state = model(x_n, t_n, h_state)
        loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], q, f_t, curr_lambda, m_t)
        loss = apply_optional_soc_monotonic_loss(loss, y_p, x[:, :, 0], m_t, config, losses["soc_mono"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
        optimizer.step()

        losses["total"].append(loss.item())
        losses["data"].append(l_d.item())
        losses["ah"].append(l_ah.item())
        losses["range"].append(l_range.item())
        pbar.update(1)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Reset": reset_count})
    pbar.close()
    return losses, curr_lambda, reset_lanes, reset_batches


def train_one_epoch_independent(model, train_ds, scaler, optimizer, criterion, config, epoch: int):
    model.train()
    loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=True)
    pbar = tqdm(
        total=len(loader),
        desc=f"Epoch {epoch}/{config['EPOCHS']} Training",
        colour="blue",
        disable=config.get("DISABLE_TQDM", False),
    )
    curr_lambda = 0.0 if not config["USE_AH_LOSS"] else config["LAMBDA_AH_START"] + (epoch - 1) * config["LAMBDA_AH_STEP"]
    losses = defaultdict(list)
    print(
        f"[Epoch {epoch}/{config['EPOCHS']}] start | "
        f"mode=independent_windows | train_windows={len(train_ds)} | batches={len(loader)} | "
        f"lambda_ah={curr_lambda:.2f}",
        flush=True,
    )

    for batch in loader:
        x = batch["x_dyn"].to(config["DEVICE"])
        t = batch["t_mean"].to(config["DEVICE"])
        y = batch["soc"].to(config["DEVICE"])
        q = batch["Q"].to(config["DEVICE"])
        mask = torch.ones(x.size(0), dtype=torch.float32, device=config["DEVICE"])
        first = torch.ones(x.size(0), dtype=torch.bool, device=config["DEVICE"])

        x_n, t_n = scaler.transform(x, t)
        y_p, _ = model(x_n, t_n, h_prev=None)
        loss, l_d, l_ah, l_range = criterion(y_p, y, x[:, :, 0], q, first, curr_lambda, mask)
        loss = apply_optional_soc_monotonic_loss(loss, y_p, x[:, :, 0], mask, config, losses["soc_mono"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP"])
        optimizer.step()

        losses["total"].append(loss.item())
        losses["data"].append(l_d.item())
        losses["ah"].append(l_ah.item())
        losses["range"].append(l_range.item())
        pbar.update(1)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    pbar.close()
    return losses, curr_lambda, 0, 0


def evaluate_temperature_generalization(model, scaler: PITDScaler, config: dict, run_id: str) -> pd.DataFrame:
    frames = []
    for order, (range_label, low_c, high_c) in enumerate(base.TEMP_GENERALIZATION_RANGES, start=1):
        datasets = base.load_temperature_generalization_test_datasets(config, range_label, low_c, high_c)
        range_run_id = f"{run_id}_tempgen_{range_label}"
        df = base.evaluate_soc_start_splits(model, scaler, datasets, config, range_run_id).copy()
        df.insert(0, "temp_range_order", order)
        df.insert(1, "temp_range", range_label)
        df.insert(2, "temp_low_C", low_c)
        df.insert(3, "temp_high_C", high_c)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = Path(config["CSV_DIR"]) / f"tempgen_soc_start_metrics_{run_id}.csv"
    total_path = Path(config["CSV_DIR"]) / f"tempgen_soc_start_total_summary_{run_id}.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    combined[combined["metric_level"].eq("total")].to_csv(total_path, index=False, encoding="utf-8-sig")
    print(f"Temperature-generalization detailed metrics saved: {combined_path}")
    print(f"Temperature-generalization total summary saved: {total_path}")
    return combined


def summarize_final(test_df: pd.DataFrame, soc_df: pd.DataFrame, tempgen_df: pd.DataFrame, config: dict, run_id: str, best_epoch, best_val_mae):
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
    for range_label, _, _ in base.TEMP_GENERALIZATION_RANGES:
        range_rows = tempgen_df[tempgen_df["temp_range"].eq(range_label)]
        safe = range_label.replace("-", "_")
        row[f"tempgen_{safe}_random_mean_mae"] = mv(range_rows, "test_random")
        row[f"tempgen_{safe}_fixed_mean_mae"] = mv(range_rows, "test_fixed")
        row[f"tempgen_{safe}_mean_mae"] = float(
            np.nanmean([row[f"tempgen_{safe}_random_mean_mae"], row[f"tempgen_{safe}_fixed_mean_mae"]])
        )
    row["tempgen_mean_mae"] = float(
        np.nanmean([row[f"tempgen_{label.replace('-', '_')}_mean_mae"] for label, _, _ in base.TEMP_GENERALIZATION_RANGES])
    )
    out = Path(config["CSV_DIR"]) / f"final_summary_{run_id}.csv"
    pd.DataFrame([row]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Final summary saved: {out}")
    print(pd.DataFrame([row]).T.to_string(header=False))


def run_all_evaluations(model, scaler: PITDScaler, datasets: tuple, config: dict, run_id: str, best_epoch=math.nan, best_val_mae=math.nan):
    test_df = base.evaluate_test_splits(model, scaler, datasets, config, run_id)
    print("\nRunning multi-SOC-start evaluation...")
    soc_df = base.evaluate_soc_start_splits(model, scaler, datasets, config, run_id)
    print("\nRunning temperature-generalization multi-SOC-start evaluation...")
    tempgen_df = evaluate_temperature_generalization(model, scaler, config, run_id)
    summarize_final(test_df, soc_df, tempgen_df, config, run_id, best_epoch, best_val_mae)


def train_ablation(ablation_name: str, args: argparse.Namespace):
    set_seed(args.seed)
    config = make_config(ablation_name, args)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"{config['PROJECT']}_{timestamp}"

    print_config(config)
    datasets = base.load_all_datasets(config)
    train_ds, val_ds, _, _ = datasets
    print(
        f"Dataset windows | train={len(train_ds)} | val={len(val_ds)} | "
        f"test_random={len(datasets[2])} | test_fixed={len(datasets[3])}",
        flush=True,
    )
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
        if config["STATEFUL_TRAINING"]:
            losses, curr_lambda, reset_lanes, reset_batches = train_one_epoch_stateful(
                model, train_ds, scaler, optimizer, criterion, config, epoch
            )
        else:
            losses, curr_lambda, reset_lanes, reset_batches = train_one_epoch_independent(
                model, train_ds, scaler, optimizer, criterion, config, epoch
            )

        avg_val_mae, file_maes = base.evaluate_dataset(
            model,
            val_loader,
            scaler,
            {
                "DEVICE": config["DEVICE"],
                "D_MODEL": config["D_MODEL"],
                "DISABLE_TQDM": config.get("DISABLE_TQDM", False),
            },
            label=f"Epoch {epoch} Val",
        )
        scheduler.step()

        for filename, mae in file_maes.items():
            val_mae_matrix[filename].append(mae)

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            save_torch_file(
                {"model_state_dict": model.state_dict(), "scaler_stats": scaler.stats, "config": config},
                best_model_path,
            )
            print(f"[Epoch {epoch}] new best model saved: val_mae={best_val_mae:.6f}")

        row = {
            "Epoch": epoch,
            "LR": float(optimizer.param_groups[0]["lr"]),
            "Train_Loss": float(np.mean(losses["total"])) if losses["total"] else float("nan"),
            "Train_Data_Loss": float(np.mean(losses["data"])) if losses["data"] else float("nan"),
            "Train_Ah_Loss": float(np.mean(losses["ah"])) if losses["ah"] else float("nan"),
            "Train_SOC_Mono_Loss": float(np.mean(losses["soc_mono"])) if losses["soc_mono"] else float("nan"),
            "Train_Range_Loss": float(np.mean(losses["range"])) if losses["range"] else float("nan"),
            "Val_MAE_Avg": avg_val_mae,
            "Best_Val_MAE_So_Far": best_val_mae,
            "Best_Epoch_So_Far": best_epoch,
            "Lambda": curr_lambda,
            "Reset_Lanes": reset_lanes,
            "Reset_Batches": reset_batches,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(Path(config["CSV_DIR"]) / f"log_{run_id}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_mae_matrix).T.to_csv(Path(config["CSV_DIR"]) / f"val_matrix_{run_id}.csv", encoding="utf-8-sig")
        print(
            f"[Epoch {epoch}] summary | train={row['Train_Loss']:.6f} | data={row['Train_Data_Loss']:.6f} | "
            f"ah={row['Train_Ah_Loss']:.6f} | soc_mono={row['Train_SOC_Mono_Loss']:.6f} | "
            f"val={avg_val_mae:.6f} | best={best_val_mae:.6f} | "
            f"best_epoch={best_epoch} | lambda_ah={curr_lambda:.2f}",
            flush=True,
        )

    ckpt = load_torch_file(best_model_path, map_location=config["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded best checkpoint for final evaluation: {best_model_path}")
    run_all_evaluations(model, scaler, datasets, config, run_id, best_epoch, best_val_mae)


def evaluate_latest(ablation_name: str, args: argparse.Namespace):
    config = make_config(ablation_name, args)
    ckpt_path = latest_checkpoint(Path(config["PTH_DIR"]), config["PROJECT"])
    ckpt = load_torch_file(ckpt_path, map_location=config["DEVICE"])
    ckpt_config = ckpt.get("config", {})
    config.update(ckpt_config)
    fresh = make_config(ablation_name, args)
    for key in ("CACHE_DIR", "PTH_DIR", "CSV_DIR", "META_DIR", "VAL_BATCH_SIZE", "SOC_BATCH_SIZE", "DEVICE"):
        config[key] = fresh[key]

    print("=== Evaluate latest ablation checkpoint ===")
    print(f"CHECKPOINT : {ckpt_path}")
    print_config(config)
    datasets = base.load_all_datasets(config)
    scaler = PITDScaler()
    scaler.stats = ckpt["scaler_stats"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"eval_{ckpt_path.stem}_{timestamp}"
    run_all_evaluations(model, scaler, datasets, config, run_id)


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
    parser.add_argument("--lambda-soc-mono", type=float, default=None)
    parser.add_argument("--soc-mono-segments", type=int, default=None)
    parser.add_argument("--soc-mono-current-thr", type=float, default=None)
    parser.add_argument("--soc-mono-margin", type=float, default=None)
    return parser


def run_cli(ablation_name: str):
    parser = make_parser(ABLATIONS[ablation_name]["title"])
    args = parser.parse_args()
    mode = args.mode
    if mode is None:
        print(f"=== {ABLATIONS[ablation_name]['title']} ===")
        print("1 - start a new training run")
        print("2 - evaluate latest checkpoint with full SOC, multi-SOC, and temperature-generalization multi-SOC")
        choice = input("Enter 1 or 2: ").strip()
        mode = "train" if choice == "1" else "eval" if choice == "2" else ""
    if mode == "train":
        train_ablation(ablation_name, args)
    elif mode == "eval":
        evaluate_latest(ablation_name, args)
    else:
        raise ValueError("Invalid choice. Use 1/2 or --mode train/eval.")
