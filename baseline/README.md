# Baseline

Current mainline:

- causal transformer
- AdaLN
- hidden-state dropout

Mainline hyperparameters:

- `LR`: `2e-4`
- `DROPOUT`: `0.1`
- `P_RESET`: `0.05`

Run:

```powershell
conda activate my_soc_env
python .\code\baseline\train_causal_adaln_dropout.py
python .\code\baseline\eval_causal_adaln_dropout.py
```

Filtered-temperature mainline:

```powershell
python .\code\baseline\train_causal_adaln_dropout_filtered_temps.py --dry-run
python .\code\baseline\train_causal_adaln_dropout_filtered_temps.py
```

This is the same mainline method (`causal transformer + AdaLN + hidden-state dropout`, `P_RESET=0.05`) but keeps only files within `10,25,40C +/-2C`. It writes caches, checkpoints, logs, and metrics under `results/baseline/filtered_temps/`.

When started without `--mode`, it asks:

- input `1`: start a new training run
- input `2`: load the latest checkpoint under `results/baseline/filtered_temps/pth_save/` and run testing only

Non-interactive alternatives:

```powershell
python .\code\baseline\train_causal_adaln_dropout_filtered_temps.py --mode train
python .\code\baseline\train_causal_adaln_dropout_filtered_temps.py --mode eval
python .\code\baseline\train_causal_adaln_dropout_filtered_temps.py --mode soc-eval
```

Testing writes total metrics plus segment-condition metrics to `results/baseline/filtered_temps/csv_save/test_segment_metrics_*.csv`. Segment metrics use the same mapping as the SOC-start experiments: 4 random-test conditions and 12 fixed-test conditions.

`--mode soc-eval` evaluates the latest filtered-temperature checkpoint from SOC start points `100,90,80,70,60,50,40,30,20,10` on the same `10,25,40C +/-2C` filtered random/fixed test sets. It writes `soc_start_metrics_*.csv` and `soc_start_total_summary_*.csv`; each SOC start contains total metrics plus 4 random-test segment conditions and 12 fixed-test segment conditions.

SOC-start augmented filtered-temperature training:

```powershell
python .\code\baseline\train_causal_adaln_soc_start_aug_filtered_temps.py --dry-run
python .\code\baseline\train_causal_adaln_soc_start_aug_filtered_temps.py
```

This keeps the same filtered temperature scope (`10,25,40C +/-2C`) and causal AdaLN model, but disables random hidden-state reset. For each training segment, it samples one division count from `{8,9,10,11,12}` and converts the segment into multiple SOC-start sub-trajectories from the sampled SOC starts down to 0 SOC. Each sub-trajectory is treated as an independent stateful training segment with zero initial hidden state.

When started without `--mode`, it asks:

- input `1`: start a new SOC-start augmented training run
- input `2`: load the latest checkpoint under `results/baseline/soc_start_aug_filtered_temps/pth_save/` and run multi-SOC-start testing only

Non-interactive alternatives:

```powershell
python .\code\baseline\train_causal_adaln_soc_start_aug_filtered_temps.py --mode train
python .\code\baseline\train_causal_adaln_soc_start_aug_filtered_temps.py --mode eval
```

Results are written under `results/baseline/soc_start_aug_filtered_temps/`. Testing writes `soc_start_metrics_*.csv` and `soc_start_total_summary_*.csv`, including total metrics plus 4 random-test segment conditions and 12 fixed-test segment conditions for each SOC start.

SOC-start evaluation:

```powershell
python .\code\baseline\eval_soc_start_causal_adaln_dropout.py
```

This evaluates `test_random` and `test_fixed` from SOC start points `100,90,80,70,60,50,40,30,20,10` and writes only aggregate metrics to `results/baseline/csv_save/soc_start_metrics_*.csv`.

Run these commands from the `soc_clean_workspace` root. Evaluation requires a checkpoint produced by the training script under `results/baseline/pth_save/`.

Training uses 64 stateful lanes by default. Validation also uses 64 deterministic stateful lanes for speed, with no hidden-state dropout. Random and fixed test evaluation remain sequential with batch size 1.

Hyperparameter search:

```powershell
conda activate my_soc_env
python .\code\baseline\tune_causal_adaln_dropout.py --dry-run
python .\code\baseline\tune_causal_adaln_dropout.py --epochs 50
```

Single-GPU parallel trial search:

```powershell
python .\code\baseline\tune_causal_adaln_dropout.py --epochs 50 --parallel-trials 10
```

P-reset SOC-start tuning:

```powershell
python .\code\baseline\tune_preset_soc_start_causal_adaln_dropout.py --dry-run
python .\code\baseline\tune_preset_soc_start_causal_adaln_dropout.py --epochs 50 --parallel-trials 8
```

This keeps the mainline hyperparameters fixed (`LR=2e-4`, `DROPOUT=0.1`, `D_MODEL=64`, `NHEAD=4`, `NUM_LAYERS=2`) and trains one model for each `P_RESET` value. Each best-validation checkpoint is evaluated on SOC start points `100,90,80,70,60,50,40,30,20,10`; only metrics are saved under `results/baseline/p_reset_soc_start/<search_id>/`. The summary contains both total MAE and segment-level MAE: 4 random-test segments and 12 fixed-test segments.

Default grid:

- `LR`: `5e-5,1e-4,2e-4`
- `P_RESET`: `0.0,0.02,0.05,0.10`
- `DROPOUT`: `0.05,0.10,0.15`

The search writes per-trial logs, best checkpoints, validation metrics, random-test metrics, fixed-test metrics, and the full summary under `results/baseline/hparam_search/<search_id>/`.
