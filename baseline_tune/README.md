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
- `train_causal_adaln_dropout_filtered_temps_dmodel96.py`: selected capacity variant from the sweep. It uses `D_MODEL=96`, `NHEAD=4`, `NUM_LAYERS=3`, `EPOCHS=150`, cosine `eta_min=2e-5`, and `LAMBDA_AH_MAX=1000`; the mainline data split, filtered temperatures, normal test, and multi-SOC-start test are kept unchanged.
- `train_causal_adaln_dropout_filtered_temps_dmodel96_backup_before_epoch150_20260803.py`: backup of the same script before the `EPOCHS=150`, `eta_min`, and AhLoss cap changes.
- `tune_capacity_causal_adaln_dropout_filtered_temps.py`: 30-trial capacity sweep over `D_MODEL`, `NHEAD`, and `NUM_LAYERS`. It keeps the mainline data split, filtered temperatures, `p_reset`, normal test, and multi-SOC-start test unchanged.
