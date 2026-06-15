"""Triple-barrier labeling (López de Prado, AFML cap. 3). Vol-scaled, point-in-time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def daily_vol(close: pd.Series, span: int = C.VOL_SPAN) -> pd.Series:
    ret = np.log(close / close.shift(1))
    return ret.ewm(span=span).std()


def _label_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    close = g["close"].to_numpy()
    high = g["high"].to_numpy()
    low = g["low"].to_numpy()
    sigma = daily_vol(g["close"]).to_numpy()
    n = len(g)

    dates = g["date"].to_numpy().astype("datetime64[ns]")
    label = np.full(n, np.nan)
    barrier = np.empty(n, dtype=object)
    fwd_ret = np.full(n, np.nan)
    t1 = np.full(n, np.datetime64("NaT", "ns"))  # data em que o label resolve (p/ purge)

    for t in range(n):
        s = sigma[t]
        if not np.isfinite(s) or s == 0:
            continue
        end = min(t + C.HORIZON, n - 1)
        if end <= t:
            continue
        up = close[t] * (1.0 + C.PT_MULT * s)
        dn = close[t] * (1.0 - C.SL_MULT * s)

        lab, why, res = 0, "vert", end
        for k in range(t + 1, end + 1):
            hit_up = high[k] >= up
            hit_dn = low[k] <= dn
            if hit_up and hit_dn:
                lab, why, res = 0, "both->sl", k
                break
            if hit_up:
                lab, why, res = 1, "pt", k
                break
            if hit_dn:
                lab, why, res = 0, "sl", k
                break
        label[t] = lab
        barrier[t] = why
        fwd_ret[t] = close[res] / close[t] - 1.0
        t1[t] = dates[res]

    g["sigma"] = sigma
    g["label"] = label
    g["barrier"] = barrier
    g["fwd_ret"] = fwd_ret
    g["t1"] = t1
    return g


def make_labels(panel: pd.DataFrame) -> pd.DataFrame:
    # pandas 3.0: iteração explícita preserva 'ticker' como coluna
    out = pd.concat(
        [_label_ticker(g) for _, g in panel.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    cache = C.PROC_DIR / "panel_labeled.parquet"
    out.to_parquet(cache, index=False)
    return out
