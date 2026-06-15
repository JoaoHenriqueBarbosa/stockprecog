"""Fractional differentiation de janela fixa (FFD) — López de Prado, AFML cap. 5.

Retorno simples (d=1) torna a série estacionária mas APAGA quase toda a memória de
longo prazo. Preço cru (d=0) tem memória mas é não-estacionário (inútil pra ML).
A diferenciação fracionária acha o d* mínimo (geralmente 0.3–0.6) que passa no
teste de estacionariedade ADF PRESERVANDO o máximo de memória. Esse é o ponto
doce que features de retorno jogam fora.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    """Pesos da FFD: w_0=1, w_k = -w_{k-1}*(d-k+1)/k, corta quando |w_k|<thresh."""
    w = [1.0]
    k = 1
    while True:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1])  # ordem pra convolução (mais antigo primeiro)


def fracdiff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Aplica FFD a uma série (janela fixa). Retorna série alinhada (NaN no warm-up)."""
    w = ffd_weights(d, thresh)
    width = len(w)
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.dot(w, window)
    return pd.Series(out, index=series.index)


def min_ffd_d(series: pd.Series, ds=None, thresh: float = 1e-4,
              pvalue: float = 0.05) -> float:
    """Menor d em `ds` cuja FFD passa no ADF (estacionária). Default: grade fina."""
    if ds is None:
        ds = np.round(np.arange(0.0, 1.01, 0.05), 2)
    for d in ds:
        fd = fracdiff_ffd(series, d, thresh).dropna()
        if len(fd) < 50:
            continue
        try:
            p = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        except Exception:  # noqa: BLE001
            continue
        if p < pvalue:
            return float(d)
    return 1.0  # fallback: retorno (sempre estacionário)
