from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CODE_ROOT = BASE_DIR.parent
ABLATION_DIR = CODE_ROOT / "ablation"

for path in (str(ABLATION_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ablation_filtered_temps_runner as runner


MAINLINE_METHOD = "temp_input_adapter_no_ah"
MAINLINE_NAME = "temp_input_adapter_mainline_mono_start025_lam0p001"
MAINLINE_EPOCHS = 100
MAINLINE_SEED = 42
MAINLINE_LAMBDA_SOC_MONO = 0.001
MAINLINE_SOC_MONO_LAST_EPOCHS = 76  # epochs 25..100 inclusive


def register_mainline_ablation() -> str:
    if MAINLINE_NAME in runner.ABLATIONS:
        return MAINLINE_NAME

    spec = deepcopy(runner.ABLATIONS[MAINLINE_METHOD])
    spec.update(
        {
            "project": "PINT_TEMPERATURE_INPUT_ADAPTER_MAINLINE_MONO_START025_LAM0P001_D96_H4_L3",
            "result_dir": "temperature_temp_input_adapter_mainline_mono_start025_lam0p001_d96h4l3",
            "title": "Mainline: temp input adapter + SOC monotonicity from epoch 25, lambda=0.001, no Ah loss",
            "use_soc_monotonic_loss": True,
            "lambda_soc_mono": MAINLINE_LAMBDA_SOC_MONO,
            "soc_mono_segments": 5,
            "soc_mono_current_thr": 0.0,
            "soc_mono_margin": 0.0,
            "soc_mono_last_epochs": MAINLINE_SOC_MONO_LAST_EPOCHS,
        }
    )
    runner.ABLATIONS[MAINLINE_NAME] = spec
    return MAINLINE_NAME


def make_child_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        mode="train",
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        soc_batch_size=args.soc_batch_size,
        seed=seed,
        cpu=args.cpu,
        disable_tqdm=args.disable_tqdm,
        lambda_soc_mono=None,
        soc_mono_segments=None,
        soc_mono_current_thr=None,
        soc_mono_margin=None,
        soc_mono_last_epochs=None,
        lambda_soc_tail_mono=None,
        soc_tail_mono_steps=None,
        soc_tail_mono_segments=None,
        soc_tail_mono_current_thr=None,
        soc_tail_mono_margin=None,
    )


def print_execution_plan(args: argparse.Namespace) -> None:
    print("=== Mainline training: temp input adapter + SOC monotonicity ===", flush=True)
    print(f"METHOD      : {MAINLINE_METHOD}", flush=True)
    print(f"NAME        : {MAINLINE_NAME}", flush=True)
    print(f"ROUNDS      : {args.rounds}", flush=True)
    print(f"EPOCHS      : {args.epochs}", flush=True)
    print(f"BATCH_SIZE  : {args.batch_size}", flush=True)
    print(f"VAL_BATCH   : {args.val_batch_size}", flush=True)
    print(f"SOC_BATCH   : {args.soc_batch_size}", flush=True)
    print(f"SEED_START  : {args.seed}", flush=True)
    print(f"MONO START  : epoch 25", flush=True)
    print(f"MONO LAMBDA : {MAINLINE_LAMBDA_SOC_MONO}", flush=True)
    print(f"MONO LAST   : {MAINLINE_SOC_MONO_LAST_EPOCHS} epochs", flush=True)
    print("LOSS        : no Ah loss", flush=True)
    print("STATE       : GRU state transfer + p_reset=0.05", flush=True)
    print("MODEL       : causal Transformer + temperature as 6th input dim + residual adapter", flush=True)
    print("TEMP SCALAR : window mean t_mean", flush=True)
    print("EVAL        : filtered full-SOC, filtered multi-SOC, temperature interpolation multi-SOC", flush=True)
    print("NOTE        : mainline is fixed; only repeated training count changes", flush=True)
    print("", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the finalized mainline: temp input adapter + SOC monotonicity from epoch 25, lambda=0.001."
    )
    parser.add_argument("--rounds", type=int, default=1, help="Number of repeated runs for statistics.")
    parser.add_argument("--epochs", type=int, default=MAINLINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=MAINLINE_SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError(f"--rounds must be positive, got {args.rounds}")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")

    ablation_name = register_mainline_ablation()
    print_execution_plan(args)
    if args.dry_run:
        print("Dry run only. No training executed.", flush=True)
        return

    for round_idx in range(1, args.rounds + 1):
        seed = args.seed + round_idx - 1
        child_args = make_child_args(args, seed)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("=" * 80, flush=True)
        print(f"ROUND {round_idx}/{args.rounds} | seed={seed} | {stamp}", flush=True)
        print("=" * 80, flush=True)
        runner.train_ablation(ablation_name, child_args)

    print("All requested mainline runs finished.", flush=True)


if __name__ == "__main__":
    main()
