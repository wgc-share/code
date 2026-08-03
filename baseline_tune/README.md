# baseline_tune

This folder stores baseline variants that keep the main experimental protocol unchanged while tuning the baseline model design or hyperparameters.

Current mainline protocol:
- causal Transformer
- AdaLN temperature modulation
- hidden-state random reset with `p_reset = 0.05`
- filtered temperatures: `10 C`, `25 C`, `40 C` with `+/-2 C` tolerance
- cell split unchanged
- random and fixed test sets unchanged
- multi-SOC-start evaluation unchanged

Files:
- `train_causal_adaln_dropout_filtered_temps_dmodel96.py`: selected capacity variant from the sweep. It uses `D_MODEL=96`, `NHEAD=4`, `NUM_LAYERS=3`, `EPOCHS=100`, two-stage cosine learning rate (`2e-4 -> 2e-5` for the first 50 epochs, then `2e-5 -> 5e-6`), and `LAMBDA_AH_MAX=1000`; the mainline data split, filtered temperatures, normal test, and multi-SOC-start test are kept unchanged.
- `train_causal_adaln_dropout_filtered_temps_dmodel96_backup_before_epoch150_20260803.py`: backup of the same script before the `EPOCHS=150`, `eta_min`, and AhLoss cap changes.
- `train_causal_adaln_dropout_filtered_temps_dmodel96_backup_e150_ahmax1000_20260803.py`: backup of the `EPOCHS=150`, cosine `eta_min=2e-5`, and AhLoss cap version before changing to the two-stage schedule.
- `baseline_d96head4lay3.py`: accepted baseline method, using `D_MODEL=96`, `NHEAD=4`, `NUM_LAYERS=3`, `EPOCHS=50`, and the original cosine schedule. Mode `4` / `--mode tempgen-soc-eval` evaluates temperature-generalization ranges `15-20 C` and `30-35 C` on random/fixed multi-SOC tests without retraining.
- `repeat_baseline_d96head4lay3.py`: repeats the accepted baseline method over multiple seeds, default 20 runs, and writes per-run metrics plus mean/std/min/max/median aggregates.
- `eval_repeat_tempgen_baseline_d96head4lay3.py`: evaluates all repeated baseline checkpoints under temperature-generalization ranges `15-20 C` and `30-35 C`, including random/fixed multi-SOC metrics and aggregate mean/std across checkpoints.
- `tune_capacity_causal_adaln_dropout_filtered_temps.py`: 30-trial capacity sweep over `D_MODEL`, `NHEAD`, and `NUM_LAYERS`. It keeps the mainline data split, filtered temperatures, `p_reset`, normal test, and multi-SOC-start test unchanged.
