# Code Layout

- `shared/` - modules used by all experiments
- `baseline/` - current mainline: causal transformer + AdaLN + hidden-state dropout
- `ablation/` - remove one baseline component at a time
- `comparison/` - FiLM, temperature transfer, temperature generalization, and legacy variants
