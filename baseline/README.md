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

Training uses 32 stateful lanes. Validation also uses 32 deterministic stateful lanes for speed, with no hidden-state dropout. Random and fixed test evaluation remain sequential with batch size 1.
