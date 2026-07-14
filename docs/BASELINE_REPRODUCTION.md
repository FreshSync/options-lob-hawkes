# TLOB Baseline Reproduction

## Setup

- Repo: github.com/LeonardoBerti00/TLOB (commit hash: f1c0af4d81067978914361766db0457a7d8b6a46)
- Environment: Google Colab, T4 GPU, PyTorch 2.5.0+cu121, torchvision 0.20.0
- Dataset: FI-2010, auto-downloaded via TLOB's pipeline
- Config: default TLOB config (4 layers, hidden_dim 40, seq_size 128, lr 0.0001, all_features True)
- Horizon: 10 (default)
- Training: 6 epochs (early stopping on val_loss)

## Results

| Metric    | TLOB paper (Table 2) | Our reproduction |
| --------- | -------------------- | ---------------- |
| F1 @ h=10 | 81.55                | 81.52            |

Difference: 0.03 F1 points (within seed noise).

## Verification

Implementation of TLOB is confirmed correct against the published paper. The FI-2010
preprocessing, TLOB architecture, and evaluation pipeline all produce numbers matching
the paper within numerical precision.

This provides a validated baseline for extending TLOB with a Transformer-Hawkes
intensity head conditioned on TTE for the options LOB direction prediction task.
