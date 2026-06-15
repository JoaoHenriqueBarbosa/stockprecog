"""Features de iliquidez/microestrutura em frequência diária — eixos ortogonais a
momentum/vol. Point-in-time (só passado), por ticker. O rank cross-section é
aplicado depois (em features.make_features); aqui só produzimos as TIME-SERIES."""
from __future__ import annotations

import numpy as np
import pandas as pd

# features time-series por ticker (cross-section rank é aplicado fora)
MICRO_FEATURES = ["amihud", "roll_spread", "dollar_vol_z", "turnover_accel"]


def amihud(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Iliquidez de Amihud (2002): média móvel de |logret_t| / volume_financeiro_t.
    Em log — a razão crua tem escala ~1e-9 (cauda extrema), log estabiliza."""
    dollar_vol = close * volume
    logret = np.log(close / close.shift(1))
    ratio = logret.abs() / dollar_vol.replace(0, np.nan)
    illiq = ratio.rolling(window).mean()
    return np.log(illiq.replace(0, np.nan))


def roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    """Estimador de spread efetivo de Roll = 2*sqrt(-cov(dP_t, dP_{t-1})).
    0 quando cov >= 0 (estimador indefinido — sem reversão de bid-ask bounce)."""
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def dollar_vol_z(close: pd.Series, volume: pd.Series, window: int = 63) -> pd.Series:
    """Z-score do log do volume financeiro vs média/desvio móvel — regime de liquidez."""
    log_dv = np.log((close * volume).replace(0, np.nan))
    mu = log_dv.rolling(window).mean()
    sd = log_dv.rolling(window).std()
    return (log_dv - mu) / sd.replace(0, np.nan)


def turnover_accel(volume: pd.Series, window: int = 21) -> pd.Series:
    """Aceleração de volume: média curta vs longa (sinal de atenção/fluxo)."""
    short = max(1, window // 4)
    fast = volume.rolling(short).mean()
    slow = volume.rolling(window).mean()
    return fast / slow.replace(0, np.nan) - 1.0


def _micro_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c, v = g["close"], g["volume"]
    g["amihud"] = amihud(c, v, 21)
    g["roll_spread"] = roll_spread(c, 21)
    g["dollar_vol_z"] = dollar_vol_z(c, v, 63)
    g["turnover_accel"] = turnover_accel(v, 21)
    return g


def make_micro_features(panel_or_group: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as colunas MICRO_FEATURES ao painel (point-in-time, por ticker)."""
    return pd.concat(
        [_micro_feats(g) for _, g in panel_or_group.groupby("ticker", sort=False)],
        ignore_index=True,
    )
