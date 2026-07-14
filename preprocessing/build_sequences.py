"""
Build training sequences from cleaned per-contract data.

For each contract, produces:
- sequences: [N, seq_len, 4] float32 arrays of (bid_px, ask_px, bid_sz, ask_sz)
- labels: [N] int64 arrays of {0=down, 1=stable, 2=up}
- tte_days: [N] float32 array of time-to-expiry at end of each sequence
- contract_id: [N] int32 array (which contract this sample came from)

Sampling: non-overlapping windows of length seq_len from the NBBO-changing event stream.
Labeling: standard TLOB/FI-2010 label defined as percentage change of average of next h mid-prices
          vs current mid-price. Threshold theta set per-contract to balance classes.
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
    Build sequences, labels, TTE, and metadata for a single contract.
    """
    n_events = len(df)
    # Need at least seq_len (for input) + horizon (for label) events
    min_needed = seq_len + horizon
    if n_events < min_needed:
        return None

    # Extract feature columns as numpy array
    features = df[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].to_numpy(
        dtype=np.float32
    )
    mid = df["mid_px"].to_numpy(dtype=np.float64)
    tte = df["tte_days"].to_numpy(dtype=np.float32)

    # Generate anchor points for sequences
    # Each anchor is the LAST index of a seq (inclusive), and we need `horizon` events after it
    anchors = np.arange(seq_len - 1, n_events - horizon, stride)
    n_samples = len(anchors)

    if n_samples == 0:
        return None

    # Build sequences: [n_samples, seq_len, 4]
    # For anchor a, sequence is features[a - seq_len + 1 : a + 1]
    seq_starts = anchors - seq_len + 1
    sequences = np.stack(
        [features[start : start + seq_len] for start in seq_starts]
    )

    # Build labels
    # Standard TLOB label: percentage change between current mid and average of next h mids
    p_now = mid[anchors]
    # Future window: for anchor a, use mid[a+1 : a+1+horizon]
    p_future_avg = np.array(
        [mid[a + 1 : a + 1 + horizon].mean() for a in anchors]
    )
    l_pct = (p_future_avg - p_now) / p_now

    labels = np.full(n_samples, 1, dtype=np.int64)  # default: stable
    labels[l_pct > theta] = 2  # up
    labels[l_pct < -theta] = 0  # down

    # TTE at the anchor
    tte_at_anchor = tte[anchors]

    return {
        "sequences": sequences,
        "labels": labels,
        "tte_days": tte_at_anchor,
        "n_samples": n_samples,
        "l_pct_stats": {
            "median": float(np.median(l_pct)),
            "std": float(np.std(l_pct)),
            "min": float(l_pct.min()),
            "max": float(l_pct.max()),
        },
    }


def pick_theta_for_balance(l_pct: np.ndarray, target_stable_frac: float = 0.60) -> float:
    """
    Pick theta so that a target fraction of samples fall in the stable class.
    For target_stable_frac=0.60, theta = 60th percentile of |l_pct|... 
    Actually, we want ~40% total outside the stable band, so 20% in each tail.
    Set theta to the 80th percentile of |l_pct| (approx equal to 20th and 80th quantiles).
    """
    # For symmetric threshold, use quantile of |l_pct|
    tail_frac = 1.0 - target_stable_frac
    return float(np.quantile(np.abs(l_pct), 1.0 - tail_frac))


def build_all_contracts(
    processed_dir: str,
    out_dir: str,
    seq_len: int = 128,
    horizon: int = 10,
    stride: int = 1,
    target_stable_frac: float = 0.60,
) -> None:
    """
    Process all contracts in the processed_dir into training arrays.
    """
    proc = Path(processed_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(proc / "_manifest.parquet")
    print(f"Building sequences from {len(manifest)} contracts")
    print(f"  seq_len={seq_len}, horizon={horizon}, stride={stride}")
    print(f"  target stable fraction: {target_stable_frac:.0%}")
    print()

    all_sequences = []
    all_labels = []
    all_tte = []
    all_contract_ids = []
    per_contract_stats = []

    for contract_id, row in manifest.iterrows():
        symbol = row["symbol"]
        file_path = proc / row["file"]
        df = pd.read_parquet(file_path)

        # First pass: compute l_pct to pick theta
        # We repeat some work but it's cheap
        features_shape_check = len(df)
        if features_shape_check < seq_len + horizon:
            print(f"  Skipping {symbol}: only {features_shape_check} events")
            continue

        mid = df["mid_px"].to_numpy(dtype=np.float64)
        anchors_preview = np.arange(seq_len - 1, len(mid) - horizon, stride)
        p_now = mid[anchors_preview]
        p_future_avg = np.array(
            [mid[a + 1 : a + 1 + horizon].mean() for a in anchors_preview]
        )
        l_pct = (p_future_avg - p_now) / p_now

        theta = pick_theta_for_balance(l_pct, target_stable_frac)

        # Now build for real
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

        # Class distribution
        unique, counts = np.unique(result["labels"], return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        down_frac = class_dist.get(0, 0) / result["n_samples"]
        stable_frac = class_dist.get(1, 0) / result["n_samples"]
        up_frac = class_dist.get(2, 0) / result["n_samples"]

        per_contract_stats.append({
            "symbol": symbol,
            "n_samples": result["n_samples"],
            "theta": theta,
            "l_pct_median": result["l_pct_stats"]["median"],
            "l_pct_std": result["l_pct_stats"]["std"],
            "down_frac": down_frac,
            "stable_frac": stable_frac,
            "up_frac": up_frac,
        })

    # Concatenate across contracts
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    tte_days = np.concatenate(all_tte, axis=0)
    contract_ids = np.concatenate(all_contract_ids, axis=0)

    # Save
    np.savez(
        out / "sequences.npz",
        sequences=sequences,
        labels=labels,
        tte_days=tte_days,
        contract_ids=contract_ids,
    )

    stats_df = pd.DataFrame(per_contract_stats)
    stats_df.to_parquet(out / "_build_stats.parquet")

    # Summary
    print(f"Total samples: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Label distribution: down={np.mean(labels == 0):.2%}, "
          f"stable={np.mean(labels == 1):.2%}, "
          f"up={np.mean(labels == 2):.2%}")
    print(f"TTE range: {tte_days.min():.2f} to {tte_days.max():.2f} days")
    print()
    print("Per-contract stats:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    build_all_contracts(
        processed_dir="data/processed/spx_smoke_2025-06-17",
        out_dir="data/sequences/spx_smoke_2025-06-17_h50",
        seq_len=128,
        horizon=50,
        stride=1,
        target_stable_frac=0.60,  # match FI-2010's roughly 60% stable
    )