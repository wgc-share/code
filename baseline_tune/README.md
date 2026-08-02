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
- `train_causal_adaln_dropout_filtered_temps_dmodel96.py`: lightly enlarged capacity variant. It changes `D_MODEL` from `64` to `96`; other mainline settings are kept unchanged.
