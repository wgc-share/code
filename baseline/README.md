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
