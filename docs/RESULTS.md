# Experimental Results

Log of all training runs, in reverse chronological order (newest at top).

## Format

Each entry includes:

- Date
- Description of what was run
- Data used
- Model config
- Test set results
- Notes / interpretation

---

## 2026-07-14: Stage 2a - Vanilla TLOB on SPX options smoke test

**Purpose:** First end-to-end training run on options data. Baseline before TTE conditioning.

**Data:**

- Source: 1 hour of SPX options quotes (2025-06-17, 13:30-14:30 UTC)
- 18 contracts, 3 TTE buckets (weekly ~2.4d, monthly ~30d, quarterly ~93d)
- Sequences built with seq_len=128, horizon=50, stride=1
- 659,961 total samples, split 70/15/15 (random shuffle, not temporal)
- Labels: 20% down, 60% stable, 20% up

**Model config:**

- Architecture: TLOB adapted to 4-feature input
- Parameters: 1,134,462
- hidden_dim=40, num_layers=4, num_heads=1, seq_size=128
- Optimizer: Adam, lr=1e-4
- Batch size: 256, 10 epochs

**Test results:**
| Metric | Value |
|---|---|
| Macro F1 | 0.5064 |
| Accuracy | 0.6401 |
| Down F1 | 0.3954 |
| Stable F1 | 0.7547 |
| Up F1 | 0.3691 |

Best val F1 across epochs: 0.5042 (epoch 10)

**Notes:**

- Model was still improving at epoch 10 (train loss 0.92 → 0.71, val F1 0.26 → 0.50). More epochs would likely help.
- Directional class recall is low (~28-31%), model overweights "stable."
- This is a lower bound; real paper will use months of data instead of 1 hour.
- No TTE conditioning yet. Stage 2b adds TTE as an input feature.
- Random 70/15/15 split, not temporal. Real paper needs temporal splitting.

**Checkpoint:** `checkpoints/tlob_options_20260714_125914.pt` (on Colab Drive)
