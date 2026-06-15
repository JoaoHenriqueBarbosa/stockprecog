"""Fixtures e helpers compartilhados. Dados sintéticos pequenos com invariantes
conhecidos — o painel real só entra no smoke de integração."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ticker(closes, highs=None, lows=None, ticker="AAA0", start="2020-01-01"):
    """Constrói um DataFrame de barras (date,ticker,open,high,low,close,volume).

    Datas são dias ÚTEIS consecutivos (bdate_range) para casar com a semântica de
    'dias-de-pregão' do pipeline. highs/lows default = close (sem range intrabar).
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    highs = closes.copy() if highs is None else np.asarray(highs, dtype=float)
    lows = closes.copy() if lows is None else np.asarray(lows, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 1_000_000.0),
    })
