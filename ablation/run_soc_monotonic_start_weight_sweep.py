from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from datetime import datetime

import ablation_filtered_temps_runner as runner


BASE_ABLATION = "temp_input_adapter_no_ah"
EPOCHS = 100
START_EPOCHS = [25, 50, 75]
LAMBDA_VALUES = [0.001, 0.005, 0.01]


def make_args(lambda_soc_mono: float, soc_mono_last_epochs: int) -> Namespace:
    return Namespace(
        mode="train",
        epochs=EPOCHS,
        batch_size=64,
        val_batch_size=64,
        soc_batch_size=64,
        seed=42,
        cpu=False,
        disable_tqdm=False,
        lambda_soc_mono=lambda_soc_mono,
        soc_mono_segments=5,
        soc_mono_current_thr=0.0,
        soc_mono_margin=0.0,
        soc_mono_last_epochs=soc_mono_last_epochs,
        lambda_soc_tail_mono=None,
        soc_tail_mono_steps=None,
        soc_tail_mono_segments=None,
        soc_tail_mono_current_thr=None,
        soc_tail_mono_margin=None,
    )


def register_ablation(start_epoch: int, lambda_soc_mono: float) -> str:
    base_spec = deepcopy(runner.ABLATIONS[BASE_ABLATION])
    suffix = f"mono_start{start_epoch:03d}_lam{str(lambda_soc_mono).replace('.', 'p')}"
    ablation_name = f"{BASE_ABLATION}_{suffix}"

    # 100 epoch training is the assumed budget here, so start_epoch -> last_epochs.
    last_epochs = max(1, EPOCHS - start_epoch + 1)

    base_spec["project"] = f"PINT_{ablation_name.upper()}_D96_H4_L3"
    base_spec["result_dir"] = f"{ablation_name}_d96h4l3"
    base_spec["title"] = (
        f"Ablation: temp input adapter + SOC monotonicity from epoch {start_epoch}, "
        f"lambda={lambda_soc_mono}, no Ah loss"
    )
    base_spec["use_soc_monotonic_loss"] = True
    base_spec["lambda_soc_mono"] = float(lambda_soc_mono)
    base_spec["soc_mono_segments"] = 5
    base_spec["soc_mono_current_thr"] = 0.0
    base_spec["soc_mono_margin"] = 0.0
    base_spec["soc_mono_last_epochs"] = int(last_epochs)

    runner.ABLATIONS[ablation_name] = base_spec
    return ablation_name


def main() -> None:
    print("=== SOC monotonic sweep plan ===")
    print(f"BASE_ABLATION : {BASE_ABLATION}")
    print(f"EPOCHS        : {EPOCHS}")
    print(f"START_EPOCHS  : {START_EPOCHS}")
    print(f"LAMBDA_VALUES : {LAMBDA_VALUES}")
    print("Rule: monotonic loss is enabled only in the last (EPOCHS - start_epoch + 1) epochs.")
    print("Each run will automatically evaluate:")
    print("  - filtered temp full-SOC")
    print("  - filtered temp multi-SOC")
    print("  - temperature interpolation multi-SOC (15-20C, 30-35C)")
    print()

    for start_epoch in START_EPOCHS:
        for lambda_soc_mono in LAMBDA_VALUES:
            ablation_name = register_ablation(start_epoch, lambda_soc_mono)
            args = make_args(lambda_soc_mono, max(1, EPOCHS - start_epoch + 1))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print("\n" + "=" * 80)
            print(f"Running {ablation_name} @ {stamp}")
            print("=" * 80)
            runner.train_ablation(ablation_name, args)


if __name__ == "__main__":
    main()
