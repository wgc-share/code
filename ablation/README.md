# Ablation

Place ablation variants here, for example:

- causal transformer only
- causal transformer + AdaLN
- causal transformer + hidden-state dropout
- non-causal variants if needed for comparison

Implemented variants:

- `train_causal_adaln_no_hidden_state.py`
  - Model: causal transformer + AdaLN + GRU head
  - Difference from mainline: no cross-window hidden-state passing
  - Training: standard `DataLoader(..., shuffle=True)`
  - Validation/test: independent windows with zero initial hidden state
  - Fixed hyperparameters: `LR=2e-4`, `DROPOUT=0.1`, `D_MODEL=64`, `NHEAD=4`, `NUM_LAYERS=2`
  - Results: `results/ablation/no_hidden_state/`

- `train_causal_adaln_soc_guided_reset.py`
  - Model: causal transformer + AdaLN + GRU head
  - Difference from mainline: SOC-guided hidden-state reset during training
  - For each training segment, sample one `n` from `{8,9,10,11,12}` and keep it fixed for that segment
  - Reset hidden state when the window start SOC crosses a new `100/n` SOC interval
  - Validation: deterministic stateful lanes without SOC-guided reset
  - Test: SOC-start evaluation at `100,90,80,70,60,50,40,30,20,10`
  - Metrics: total MAE plus 4 random-test segment MAEs and 12 fixed-test segment MAEs
  - Fixed hyperparameters: `LR=2e-4`, `DROPOUT=0.1`, `D_MODEL=64`, `NHEAD=4`, `NUM_LAYERS=2`
  - Results: `results/ablation/soc_guided_reset/`

- `train_causal_adaln_soc_start_reset.py`
  - Model: causal transformer + AdaLN + GRU head
  - Data: target temperatures `10,25,40C` with `+/-2C` tolerance by default; middle-temperature intervals outside these ranges are discarded
  - Difference from mainline: each training segment samples one reset start from `100,90,80,70,60,50,40,30,20,10`
  - If the sampled start is `100`, only the natural zero initial hidden state is used
  - Otherwise, hidden state is reset once when the segment first reaches the sampled SOC start
  - Validation: deterministic stateful lanes without SOC-start reset
  - Test: SOC-start evaluation at `100,90,80,70,60,50,40,30,20,10`
  - Metrics: total MAE plus 4 random-test segment MAEs and 12 fixed-test segment MAEs
  - Fixed hyperparameters: `LR=2e-4`, `DROPOUT=0.1`, `D_MODEL=64`, `NHEAD=4`, `NUM_LAYERS=2`
  - Results: `results/ablation/soc_start_reset/`

- `train_causal_adaln_soc_start_reset_all_temps.py`
  - Model: causal transformer + AdaLN + GRU head
  - Data: all temperatures from the same cell split; no temperature filtering
  - Reset, validation, and SOC-start testing logic are the same as `train_causal_adaln_soc_start_reset.py`
  - Caches are labeled with `all_temps`
  - Results: `results/ablation/soc_start_reset_all_temps/`

Run:

```powershell
conda activate my_soc_env
python .\code\ablation\train_causal_adaln_soc_guided_reset.py --dry-run
python .\code\ablation\train_causal_adaln_soc_guided_reset.py --epochs 50
python .\code\ablation\train_causal_adaln_soc_start_reset.py --dry-run
python .\code\ablation\train_causal_adaln_soc_start_reset.py --epochs 50
python .\code\ablation\train_causal_adaln_soc_start_reset_all_temps.py --dry-run
python .\code\ablation\train_causal_adaln_soc_start_reset_all_temps.py --epochs 50
```

`train_causal_adaln_soc_start_reset.py` and `train_causal_adaln_soc_start_reset_all_temps.py` support an interactive mode:

- input `1`: start a new training run
- input `2`: load the latest checkpoint under the corresponding result path and run SOC-start testing only

Non-interactive alternatives:

```powershell
python .\code\ablation\train_causal_adaln_soc_start_reset.py --mode train --epochs 50
python .\code\ablation\train_causal_adaln_soc_start_reset.py --mode eval
python .\code\ablation\train_causal_adaln_soc_start_reset_all_temps.py --mode train --epochs 50
python .\code\ablation\train_causal_adaln_soc_start_reset_all_temps.py --mode eval
```

Use `--temps` and `--temp-tolerance` to override the default temperature filter in `train_causal_adaln_soc_start_reset.py`:

```powershell
python .\code\ablation\train_causal_adaln_soc_start_reset.py --mode train --temps "10,25,40" --temp-tolerance 2
```
