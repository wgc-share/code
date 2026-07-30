# Baseline

Current mainline:

- causal transformer
- AdaLN
- hidden-state dropout

Run:

```powershell
conda activate my_soc_env
python .\code\baseline\train_causal_adaln_dropout.py
python .\code\baseline\eval_causal_adaln_dropout.py
```

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

Default grid:

- `LR`: `5e-5,1e-4,2e-4`
- `P_RESET`: `0.0,0.02,0.05,0.10`
- `DROPOUT`: `0.05,0.10,0.15`

The search writes per-trial logs, best checkpoints, validation metrics, random-test metrics, fixed-test metrics, and the full summary under `results/baseline/hparam_search/<search_id>/`.
