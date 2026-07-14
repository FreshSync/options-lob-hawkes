"""
Build training sequences with TTE, event types, and inter-event times.

Adds two arrays beyond build_sequences_tte.py:
- next_event_type: int64 array of {0=bid_change, 1=ask_change, 2=trade}
- next_event_dt: float32 array of time gap in seconds to the next event after sequence end
"""

import numpy as np
import pandas as pd
from pathlib import Path


def classify_event(row, prev_bid_px, prev_ask_px, prev_bid_sz, prev_ask_sz):
    """
    Classify an event as one of {0: bid change, 1: ask change, 2: trade}.
    We use action if it's a trade (T); otherwise infer from which NBBO side changed.
    Priority: trade > ask change > bid change (arbitrary but consistent).
    """
    if row["action"] == "T":
        return 2
    ask_changed = (row["ask_px_00"] != prev_ask_px) or (row["ask_sz_00"] != prev_ask_sz)
    bid_changed = (row["bid_px_00"] != prev_bid_px) or (row["bid_sz_00"] != prev_bid_sz)
    if ask_changed:
        return 1
    if bid_changed:
        return 0
    # If nothing seems changed (shouldn't happen since we filtered), default to bid
    return 0


def classify_events_vectorized(df):
    """
    Vectorized event classification.
    Returns array of event types [0=bid_change, 1=ask_change, 2=trade].
    """
    n = len(df)
    event_types = np.zeros(n, dtype=np.int64)

    # Trades directly
    is_trade = (df["action"] == "T").to_numpy()

    # Compute changes vs previous row
    prev_bid_px = df["bid_px_00"].shift(1)
    prev_ask_px = df["ask_px_00"].shift(1)
    prev_bid_sz = df["bid_sz_00"].shift(1)
    prev_ask_sz = df["ask_sz_00"].shift(1)

    ask_changed = (df["ask_px_00"] != prev_ask_px) | (df["ask_sz_00"] != prev_ask_sz)
    bid_changed = (df["bid_px_00"] != prev_bid_px) | (df["bid_sz_00"] != prev_bid_sz)

    # Priority: trade > ask > bid
    event_types[bid_changed.to_numpy()] = 0
    event_types[ask_changed.to_numpy()] = 1
    event_types[is_trade] = 2

    return event_types


def build_for_contract(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    theta: float,
    stride: int,
) -> dict:
    n_events = len(df)
    if n_events < seq_len + horizon + 1:
        return None

    # LOB features
    features = df[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].to_numpy(
        dtype=np.float32
    )
    mid = df["mid_px"].to_numpy(dtype=np.float64)
    tte = df["tte_days"].to_numpy(dtype=np.float32)
    log_tte = np.log(tte + 1.0).astype(np.float32)

    combined = np.concatenate([features, log_tte.reshape(-1, 1)], axis=1)  # (N, 5)

    # Event types across the whole contract
    event_types_all = classify_events_vectorized(df)

    # Inter-event times in seconds
    ts_recv = df["ts_recv"].to_numpy()  # datetime64
    ts_recv_s = ts_recv.astype("datetime64[ns]").astype(np.int64) / 1e9  # to seconds
    dt_all = np.diff(ts_recv_s, prepend=ts_recv_s[0]).astype(np.float32)  # dt[i] = ts[i] - ts[i-1]
    # dt[0] is 0 (no prior event); use small positive value to avoid log(0)
    dt_all[0] = 1e-6

    # Anchors: end index of each sequence
    anchors = np.arange(seq_len - 1, n_events - horizon - 1, stride)
    n_samples = len(anchors)
    if n_samples == 0:
        return None

    seq_starts = anchors - seq_len + 1
    sequences = np.stack([combined[start : start + seq_len] for start in seq_starts])

    # Direction labels (same as before)
    p_now = mid[anchors]
    p_future_avg = np.array([mid[a + 1 : a + 1 + horizon].mean() for a in anchors])
    l_pct = (p_future_avg - p_now) / p_now
    labels = np.full(n_samples, 1, dtype=np.int64)
    labels[l_pct > theta] = 2
    labels[l_pct < -theta] = 0

    # Hawkes targets:
    # next_event_type = event type at (anchor + 1), which is the event immediately after sequence end
    # next_event_dt = time gap from anchor to (anchor + 1) in seconds
    next_event_type = event_types_all[anchors + 1]
    next_event_dt = dt_all[anchors + 1]

    # Clamp minimum dt to avoid degenerate loss (some events may have zero gap due to sub-microsecond ties)
    next_event_dt = np.maximum(next_event_dt, 1e-6).astype(np.float32)

    tte_at_anchor = tte[anchors]

    return {
        "sequences": sequences,
        "labels": labels,
        "tte_days": tte_at_anchor,
        "next_event_type": next_event_type,
        "next_event_dt": next_event_dt,
        "n_samples": n_samples,
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
    print(f"Building Hawkes-annotated sequences from {len(manifest)} contracts")
    print(f"  seq_len={seq_len}, horizon={horizon}, stride={stride}")
    print()

    all_sequences = []
    all_labels = []
    all_tte = []
    all_contract_ids = []
    all_next_event_type = []
    all_next_event_dt = []

    for contract_id, row in manifest.iterrows():
        symbol = row["symbol"]
        file_path = proc / row["file"]
        df = pd.read_parquet(file_path)

        if len(df) < seq_len + horizon + 1:
            continue

        mid = df["mid_px"].to_numpy(dtype=np.float64)
        anchors_preview = np.arange(seq_len - 1, len(mid) - horizon - 1, stride)
        p_now = mid[anchors_preview]
        p_future_avg = np.array([mid[a + 1 : a + 1 + horizon].mean() for a in anchors_preview])
        l_pct = (p_future_avg - p_now) / p_now
        theta = pick_theta_for_balance(l_pct, target_stable_frac)

        result = build_for_contract(df, seq_len=seq_len, horizon=horizon, theta=theta, stride=stride)
        if result is None:
            continue

        all_sequences.append(result["sequences"])
        all_labels.append(result["labels"])
        all_tte.append(result["tte_days"])
        all_contract_ids.append(np.full(result["n_samples"], contract_id, dtype=np.int32))
        all_next_event_type.append(result["next_event_type"])
        all_next_event_dt.append(result["next_event_dt"])

    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    tte_days = np.concatenate(all_tte, axis=0)
    contract_ids = np.concatenate(all_contract_ids, axis=0)
    next_event_type = np.concatenate(all_next_event_type, axis=0)
    next_event_dt = np.concatenate(all_next_event_dt, axis=0)

    np.savez(
        out / "sequences.npz",
        sequences=sequences,
        labels=labels,
        tte_days=tte_days,
        contract_ids=contract_ids,
        next_event_type=next_event_type,
        next_event_dt=next_event_dt,
    )

    print(f"Total samples: {len(sequences):,}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Label distribution: down={np.mean(labels == 0):.2%}, "
          f"stable={np.mean(labels == 1):.2%}, "
          f"up={np.mean(labels == 2):.2%}")
    print(f"Event type distribution:")
    for etype, name in enumerate(["bid_change", "ask_change", "trade"]):
        frac = np.mean(next_event_type == etype)
        print(f"  {name}: {frac:.2%}")
    print(f"Inter-event dt: min={next_event_dt.min():.6f}s, "
          f"median={np.median(next_event_dt):.4f}s, "
          f"max={next_event_dt.max():.2f}s")


if __name__ == "__main__":
    build_all_contracts(
        processed_dir="data/processed/spx_smoke_2025-06-17",
        out_dir="data/sequences/spx_smoke_2025-06-17_hawkes",
        seq_len=128,
        horizon=50,
        stride=1,
        target_stable_frac=0.60,
    )