from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CODE_ROOT = BASE_DIR.parent
ABLATION_DIR = CODE_ROOT / "ablation"

for path in (str(ABLATION_DIR), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ablation_filtered_temps_runner as filtered_runner
import ablation_temp_extreme_runner as extreme_runner


METHOD = "temp_input_adapter_no_ah"
TEMP_INPUT_MODE = "sixth_dim_window_mean"


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
    )


def print_execution_plan(args: argparse.Namespace):
    print("=== Temperature Input + Residual Adapter baseline loop ===", flush=True)
    print(f"METHOD      : {METHOD}", flush=True)
    print(f"TEMP_MODE   : {TEMP_INPUT_MODE}", flush=True)
    print(f"ROUNDS      : {args.rounds}", flush=True)
    print(f"EPOCHS      : {args.epochs}", flush=True)
    print(f"BATCH_SIZE  : {args.batch_size}", flush=True)
    print(f"VAL_BATCH   : {args.val_batch_size}", flush=True)
    print(f"SOC_BATCH   : {args.soc_batch_size}", flush=True)
    print(f"SEED_START  : {args.seed}", flush=True)
    print("LOSS        : no Ah loss", flush=True)
    print("STATE       : GRU state transfer + p_reset=0.05", flush=True)
    print("MODEL       : causal Transformer + temperature as 6th input dim + residual adapter", flush=True)
    print("TEMP SCALAR : window mean t_mean", flush=True)
    print("", flush=True)
    print("Execution order in each round:", flush=True)
    print("  1. filtered training/eval", flush=True)
    print("     train/val/test temperature: 10C, 25C, 40C within +/-2C", flush=True)
    print("     after training: full-SOC test, multi-SOC test, 15-20C and 30-35C interpolation multi-SOC test", flush=True)
    print("  2. extreme training/eval", flush=True)
    print("     train/val temperature: 13C <= T <= 37C", flush=True)
    print("     test temperature: T < 13C and T > 37C", flush=True)
    print("     after training: full-SOC test, multi-SOC test, low/high extreme-temperature multi-SOC test", flush=True)
    print("", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Loop the accepted baseline: temp input + residual adapter, no Ah loss."
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of paired filtered/extreme runs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--soc-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError(f"--rounds must be positive, got {args.rounds}")

    print_execution_plan(args)
    if args.dry_run:
        print("Dry run only. No training executed.", flush=True)
        return

    for round_idx in range(1, args.rounds + 1):
        seed = args.seed + round_idx - 1
        child_args = make_child_args(args, seed)
        print(f"=== ROUND {round_idx}/{args.rounds} | seed={seed} | scope=filtered ===", flush=True)
        filtered_runner.train_ablation(METHOD, child_args)
        print(f"=== ROUND {round_idx}/{args.rounds} | seed={seed} | scope=extreme ===", flush=True)
        extreme_runner.train_ablation(METHOD, child_args)

    print("All requested paired runs finished.", flush=True)


if __name__ == "__main__":
    main()
