"""
Load raw Databento parquet, filter to NBBO-changing events, save per-contract.

Input:  data/raw/spx_smoke_2025-06-17_quotes.parquet
Output: data/processed/spx_smoke_2025-06-17/<contract_symbol>.parquet (one per contract)
        data/processed/spx_smoke_2025-06-17/_manifest.parquet (contract metadata)
"""

import pandas as pd
from pathlib import Path


def load_and_clean(
    raw_path: str,
    basket_path: str,
    out_dir: str,
) -> None:
    """
    Load raw quotes, filter to NBBO-changing events per contract, save.
    """
    # Load raw quote data (ts_recv is stored as the index, reset to column)
    quotes = pd.read_parquet(raw_path).reset_index()
    print(f"Loaded {len(quotes):,} raw events from {raw_path}")

    # Load basket metadata (has expiry, strike, class per symbol)
    basket = pd.read_parquet(basket_path)
    print(f"Loaded basket metadata: {len(basket)} contracts")

    # Prepare output directory
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nProcessing contracts:")
    manifest_rows = []
    for symbol, group in quotes.groupby("symbol"):
        symbol_clean = symbol.strip()

        # Sort by receive timestamp (already sorted but explicit is safer)
        g = group.sort_values("ts_recv").copy()

        # Keep only rows that changed the NBBO
        # NBBO consists of bid_px_00, ask_px_00, bid_sz_00, ask_sz_00
        nbbo_cols = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]

        # Drop rows where NBBO is entirely NaN (e.g. pure trade events)
        g_nbbo = g.dropna(subset=nbbo_cols, how="all").copy()

        # Compare each row to the previous NBBO; keep rows that changed
        prev = g_nbbo[nbbo_cols].shift(1)
        changed = (g_nbbo[nbbo_cols] != prev).any(axis=1)
        # Keep first row (previous is NaN so nothing to compare)
        changed.iloc[0] = True
        g_clean = g_nbbo[changed].copy()

        # Compute mid-price and spread
        g_clean["mid_px"] = (g_clean["bid_px_00"] + g_clean["ask_px_00"]) / 2
        g_clean["spread"] = g_clean["ask_px_00"] - g_clean["bid_px_00"]

        # Diagnostic: negative/zero spreads (locked/crossed markets)
        n_negative = (g_clean["spread"] < 0).sum()
        n_zero = (g_clean["spread"] == 0).sum()
        if n_negative > 0 or n_zero > 0:
            pct_neg = n_negative / len(g_clean) * 100
            pct_zero = n_zero / len(g_clean) * 100
            print(f"  {symbol_clean}: {n_negative} negative spreads ({pct_neg:.2f}%), "
                  f"{n_zero} zero spreads ({pct_zero:.2f}%)")

        # Drop rows with non-positive spreads (locked/crossed markets)
        g_clean = g_clean[g_clean["spread"] > 0].copy()

        assert (g_clean["mid_px"] > 0).all(), f"Non-positive mid in {symbol_clean}"

        # Attach contract metadata as columns (constant per contract)
        meta_row = basket[basket["raw_symbol"] == symbol_clean].iloc[0]
        g_clean["strike_price"] = meta_row["strike_price"]
        g_clean["expiration"] = meta_row["expiration"]
        g_clean["instrument_class"] = meta_row["instrument_class"]

        # Compute time-to-expiry (TTE) at each event, in days
        g_clean["tte_days"] = (
            meta_row["expiration"] - g_clean["ts_recv"]
        ).dt.total_seconds() / 86400

        # Keep only useful columns
        keep_cols = [
            "ts_recv", "ts_event",
            "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
            "mid_px", "spread",
            "action", "side",
            "strike_price", "expiration", "instrument_class",
            "tte_days",
        ]
        g_out = g_clean[keep_cols].reset_index(drop=True)

        # Save per-contract cleaned file
        safe_name = symbol_clean.replace(" ", "_")
        out_path = out / f"{safe_name}.parquet"
        g_out.to_parquet(out_path)

        manifest_rows.append({
            "symbol": symbol_clean,
            "n_raw_events": len(g),
            "n_clean_events": len(g_clean),
            "strike_price": meta_row["strike_price"],
            "expiration": meta_row["expiration"],
            "instrument_class": meta_row["instrument_class"],
            "mid_min": g_clean["mid_px"].min(),
            "mid_max": g_clean["mid_px"].max(),
            "spread_median": g_clean["spread"].median(),
            "tte_days_min": g_clean["tte_days"].min(),
            "tte_days_max": g_clean["tte_days"].max(),
            "file": str(out_path.name),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(out / "_manifest.parquet")

    # Summary
    print(f"\nProcessed {len(manifest_rows)} contracts")
    print(f"Total raw events: {manifest['n_raw_events'].sum():,}")
    print(f"Total clean (NBBO-changing) events: {manifest['n_clean_events'].sum():,}")
    print(f"Reduction: {(1 - manifest['n_clean_events'].sum() / manifest['n_raw_events'].sum()) * 100:.1f}%")
    print(f"\nPer-contract:")
    print(manifest[[
        "symbol", "n_clean_events", "strike_price", "instrument_class",
        "tte_days_min", "tte_days_max", "spread_median",
    ]].to_string(index=False))


if __name__ == "__main__":
    load_and_clean(
        raw_path="data/raw/spx_smoke_2025-06-17_quotes.parquet",
        basket_path="data/raw/spx_smoke_2025-06-17_basket.parquet",
        out_dir="data/processed/spx_smoke_2025-06-17",
    )