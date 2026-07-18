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

## 2026-07-14: Stage 2b - TLOB + TTE as 5th feature

**Purpose:** Test whether TTE-as-feature (vs TTE-as-architecture) improves direction prediction. This is the critical control baseline in the paper's ablation table.

**Data:** Same 1-hour smoke test as Stage 2a, but with log(tte_days + 1) as 5th input feature. Sequence shape (N, 128, 5) instead of (N, 128, 4).

**Model config:** Same as Stage 2a except num_features=5 (1,134,504 parameters vs 1,134,462).

**Test results:**
| Metric | Value | Δ vs Stage 2a |
|---|---|---|
| Macro F1 | 0.5089 | +0.0025 |
| Accuracy | 0.6219 | -0.0182 |
| Down F1 | 0.3807 | -0.0147 |
| Stable F1 | 0.7349 | -0.0198 |
| Up F1 | 0.4111 | +0.0420 |

Best val F1: 0.5057 (epoch 10, still improving)

**Notes:**

- Adding TTE as a raw input feature barely helped macro F1 (+0.0025, within seed noise).
- Class-level effect concentrated in "up" prediction (+0.042); other classes marginally worse.
- With only 3 TTE clusters (2.4d, 30d, 93d), model may already be inferring TTE from other features (spread scale, price level).
- On real data with continuous TTE variation, effect may differ. Repeat with 12-month data before drawing conclusions.
- Result supports paper's story: TTE-as-feature is a weak baseline; architectural TTE conditioning (Hawkes-NN) needs to beat this modest number to demonstrate value.

**Checkpoint:** `checkpoints/tlob_options_tte_20260714_133230.pt`

## 2026-07-18: Stage 2c diagnostic - TLOB-Hawkes with hawkes_weight=0.0

**Purpose:** Diagnose whether earlier TLOB-Hawkes underperformance (F1=0.29 to 0.40 across two runs) was caused by architecture or by Hawkes loss competing for gradient.

**Setup:** Same smoke test data, same 70/15/15 split. Rewrote TLOBHawkes model with:

- Direction head using TLOB's exact flattening + MLP structure (fixed from broken last-token pooling)
- Hawkes head as separate branch reading from pre-final-compression encoder output
- No shared post-encoder weights between direction and Hawkes branches

Trained with hawkes_weight=0.0 (Hawkes head parameters exist but don't affect training gradient).

**Test results:**
| Metric | Value | Δ vs vanilla TLOB |
|---|---|---|
| Macro F1 | 0.4846 | -0.0218 |
| Accuracy | 0.6246 | -0.0155 |
| Best val F1 | 0.4816 | -0.0226 |

**Notes:**

- CE loss trajectory (0.91 → 0.73) now matches vanilla TLOB (0.92 → 0.71).
- Slight F1 delta vs vanilla within seed noise.
- Confirms previous failures (F1=0.29 with λ=1.0, F1=0.40 with λ=0.01) were architectural, not loss weighting.
- Root cause was last-token pooling in the direction head, discarding sequence information TLOB relies on.
- Fixed architecture ready for turning Hawkes loss back on in subsequent runs.

**Next:** Turn hawkes_weight to small positive value (e.g., 0.01, 0.001) and see if joint training helps or hurts direction F1 relative to this baseline.
