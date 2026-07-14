# SPX Options Data Format (from Databento OPRA.PILLAR)

Dataset: OPRA.PILLAR
Schema: cmbp-1 (consolidated NBBO, 1 level)
Symbology: raw_symbol for specific contracts (e.g. "SPX 250620C06000000"),
ALL_SYMBOLS for definitions

Format: "SPX YYMMDDCXXXXXXXX" where:
YYMMDD = expiry date
C or P = call or put
XXXXXXXX = strike price × 1000 (right-padded)

Example: "SPX 250620C06000000" = SPX 2025-06-20 expiry, Call, $6000 strike

Columns in cmbp-1:
ts_recv, ts_event: nanosecond timestamps
action: A=add, T=trade, M=modify, C=cancel, F=fill, R=clear, N=none
side: B=bid, A=ask, N=trade/non-directional
price, size: event-specific
bid_px_00, ask_px_00, bid_sz_00, ask_sz_00: resulting NBBO
symbol: contract identifier

One hour of 18 ATM SPX contracts = ~660k events, ~14 MB parquet
