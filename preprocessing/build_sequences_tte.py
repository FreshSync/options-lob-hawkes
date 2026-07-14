"""
Build training sequences with TTE conditioning as a 5th input feature.

Same as build_sequences.py but concatenates log(tte_days + 1) as a 5th feature
at every timestep of every sequence.

Reads:  data/processed/spx_smoke_2025-06-17/*.parquet
Writes: data/sequences/spx_smoke_2025-06-17_h50_tte/sequences.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path


def build_for_contract(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    theta: float,
    stride: int,
) -> dict:
    """
    Build sequences with TTE-as-5th-feature, labels, and metadata for one contract.
    """
    n_events = len(df)
    min_needed = seq_len + horizon
    if n_events < min_needed:
        return None

    # LOB features
    features = df[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].to_numpy(
        dtype=np.float32
    )
    mid = df["mid_px"].to_numpy(dtype=np.float64)
    tte = df["tte_days"].to_numpy(dtype=np.float32)

    # Log-transform TTE
    log_tte = np.log(tte + 1.0).astype(np.float32)

    # Combined features: [bid_px, ask_px, bid_sz, ask_sz, log_tte]
    combined = np.concatenate(
        [features, log_tte.reshape(-1, 1)], axis=1
    )  # shape: (n_events, 5)

    # Anchors
    anchors = np.arange(seq_len - 1, n_events - horizon, stride)
    n_samples = len(anchors)
    if n_samples == 0:
        return None

    # Build sequences: (n_samples, seq_len, 5)
    seq_starts = anchors - seq_len + 1
    sequences = np.stack(
        [combined[start : start + seq_len] for start in seq_starts]
    )

    # Labels: same as before
    p_now = mid[anchors]
    p_future_avg = np.array(
        [mid[a + 1 : a + 1 + horizon].mean() for a in anchors]
    )
    l_pct = (p_future_avg - p_now) / p_now

    labels = np.full(n_samples, 1, dtype=np.int64)
    labels[l_pct > theta] = 2
    labels[l_pct < -theta] = 0

    tte_at_anchor = tte[anchors]

    return {
        "sequences": sequences,
        "labels": labels,
        "tte_days": tte_at_anchor,
        "n_samples": n_samples,
        "l_pct_stats": {
            "median": float(np.median(l_pct)),
            "std": float(np.std(l_pct)),
        },
    }


def pick_theta_for_balance(l_pct: np.ndarray, target_stable_frac: float = 0.60) -> float:
    tail_frac = 1.0 - target_stable_frac
    return float(np.quantile(np.abs(l_pct), 1.0 - tail_frac))


def build_all_contracts(
    processed_dir: str,
    out_dir: str,
    seq_len: int = 128,
    horizon: int = 50,
    stride: int = 1,
    target_stable_frac: float = 0.60,
) -> None:
    proc = Path(processed_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(proc / "_manifest.parquet")
    print(f"Building TTE-conditioned sequences from {len(manifest)} contracts")
    print(f"  seq_len={seq_len}, horizon={horizon}, stride={stride}")
    print(f"  Adding log(tte_days + 1) as 5th feature")
    print()

    all_sequences = []
    all_labels = []
    all_tte = []
    all_contract_ids = []

    for contract_id, row in manifest.iterrows():
        symbol = row["symbol"]
        file_path = proc / row["file"]
        df = pd.read_parquet(file_path)

        if len(df) < seq_len + horizon:
            print(f"  Skipping {symbol}: only {len(df)} events")
            continue

        # Compute theta for this contract
        mid = df["mid_px"].to_numpy(dtype=np.float64)
        anchors_preview = np.arange(seq_len - 1, len(mid) - horizon, stride)
        p_now = mid[anchors_preview]
        p_future_avg = np.array(
            [mid[a + 1 : a + 1 + horizon].mean() for a in anchors_preview]
        )
        l_pct = (p_future_avg - p_now) / p_now
        theta = pick_theta_for_balance(l_pct, target_stable_frac)

        result = build_for_contract(
            df, seq_len=seq_len, horizon=horizon, theta=theta, stride=stride
        )
        if result is None:
            continue

        all_sequences.append(result["sequences"])
        all_labels.append(result["labels"])
        all_tte.append(result["tte_days"])
        all_contract_ids.append(
            np.full(result["n_samples"], contract_id, dtype=np.int32)
        )

    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    tte_days = np.concatenate(all_tte, axis=0)
    contract_ids = np.concatenate(all_contract_ids, axis=0)

    np.savez(
        out / "sequences.npz",
        sequences=sequences,
        labels=labels,
        tte_days=tte_days,
        contract_ids=contract_ids,
    )

    print(f"Total samples: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Label distribution: down={np.mean(labels == 0):.2%}, "
          f"stable={np.mean(labels == 1):.2%}, "
          f"up={np.mean(labels == 2):.2%}")
    print(f"TTE range: {tte_days.min():.2f} to {tte_days.max():.2f} days")
    print(f"log(TTE+1) range: {np.log(tte_days.min() + 1):.3f} to {np.log(tte_days.max() + 1):.3f}")


if __name__ == "__main__":
    build_all_contracts(
        processed_dir="data/processed/spx_smoke_2025-06-17",
        out_dir="data/sequences/spx_smoke_2025-06-17_h50_tte",
        seq_len=128,
        horizon=50,
        stride=1,
        target_stable_frac=0.60,
    )