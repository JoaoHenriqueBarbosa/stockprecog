"""Features mínimas pro loop end-to-end. Point-in-time (só passado), cross-sectional ranks."""
from __future__ import annotations

import numpy as np
import pandas as pd

# features de série temporal por ticker (todas usam só passado)
TS_FEATURES = ["mom_5", "mom_10", "mom_21", "vol_10", "vol_21", "dist_sma21", "rsi_14"]
# versões rankeadas cross-section (dentro de cada dia) — o que dá poder ao cross-section
CS_FEATURES = [f"{f}_rank" for f in TS_FEATURES]
FEATURES = TS_FEATURES + CS_FEATURES


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ts_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"]
    logret = np.log(c / c.shift(1))
    g["mom_5"] = c.pct_change(5)
    g["mom_10"] = c.pct_change(10)
    g["mom_21"] = c.pct_change(21)
    g["vol_10"] = logret.rolling(10).std()
    g["vol_21"] = logret.rolling(21).std()
    g["dist_sma21"] = c / c.rolling(21).mean() - 1.0
    g["rsi_14"] = _rsi(c, 14)
    return g


def make_features(labeled: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat(
        [_ts_feats(g) for _, g in labeled.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    # ranks cross-section dentro de cada data (pct rank em [0,1])
    for f in TS_FEATURES:
        df[f"{f}_rank"] = df.groupby("date")[f].rank(pct=True)
    return df
