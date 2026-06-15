"""Features de REGIME / quebra estrutural / explosividade — eixos ORTOGONAIS a momentum.

Momentum/vol/rsi/fracdiff captam NÍVEL e direção da tendência. Aqui medimos MUDANÇA
de regime: o quanto a dinâmica recente rompe com a passada (CUSUM), se o preço está em
trajetória explosiva/bolha (SADF-lite) e se a volatilidade está expandindo ou contraindo
(vol_regime). São sinais de "o mundo mudou", não de "o preço subiu" — daí a ortogonalidade.

Regras duras (idênticas a features.py):
- POINT-IN-TIME: toda janela termina em t e usa só [.., t]. Nenhum bfill, nenhuma janela
  centrada, nenhum lookahead. diff() usa t e t-1 (passado), ok.
- POR TICKER: calculado dentro da timeline de cada ticker (groupby('ticker')).
- pandas 3.0: nada de fillna(method=). Vetorizado em numpy (sem statsmodels por janela).

make_regime_features produz as features TIME-SERIES por ticker; o rank cross-section
(groupby('date').rank) é aplicado depois em features.make_features, fora daqui.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# features de série temporal por ticker (o rank cross-section é aplicado a jusante)
REGIME_FEATURES = ["cusum_vol", "explosivity", "vol_regime"]


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Soma móvel trailing de janela fixa w via cumsum: out[t] = sum(x[t-w+1 .. t]).
    NaN nas primeiras w-1 posições (warm-up). Trata NaN em x como 0 (chamador mascara)."""
    cs = np.concatenate([[0.0], np.nancumsum(x)])
    out = np.full(x.shape[0], np.nan)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def cusum_vol_robust(close: pd.Series, window: int = 63) -> pd.Series:
    """CUSUM Chu-Stinchcombe-White de quebra na MÉDIA dos log-retornos, VOL-ROBUSTO.

    O teste CSW clássico assume vol constante; séries B3 têm vol time-varying, então
    sem normalizar a estatística estoura toda vez que a vol sobe (falso positivo de
    regime). Aqui cada retorno é primeiro padronizado pela vol móvel local (z-score
    trailing), e o supremo CSW é calculado sobre os retornos padronizados:

        z_t = (r_t - mean_w(r)) / std_w(r)            (demean + vol-robusto, só passado)
        C_t = sum_{i<=t} z_i
        S_t = max_{j=1..window} |C_t - C_{t-j}| / sqrt(j)

    S_t grande => a soma acumulada dos retornos padronizados desviou demais de algum
    ponto de referência recente => quebra na média. Rolling, supremo só sobre o passado.
    """
    c = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(c), prepend=np.nan)

    rs = pd.Series(r)
    mu = rs.rolling(window).mean().to_numpy()
    sd = rs.rolling(window).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r - mu) / sd
    z[~np.isfinite(z)] = np.nan

    finite = np.isfinite(z).astype(float)
    cz = np.where(np.isfinite(z), z, 0.0)
    C = np.concatenate([[0.0], np.cumsum(cz)])  # C[k] = sum z[0..k-1]; len = L+1

    L = c.shape[0]
    out = np.full(L, np.nan)
    if L >= window + 1:
        # janelas deslizantes de C de comprimento window+1: [C_{t-window} .. C_t]
        win = np.lib.stride_tricks.sliding_window_view(C, window + 1)  # (L+1-window, w+1)
        Ct = win[:, -1][:, None]
        Cn = win[:, :-1]                       # referências n = t-window .. t-1
        denom = np.sqrt(np.arange(window, 0, -1, dtype=float))  # t-n = window..1
        stat = np.nanmax(np.abs(Ct - Cn) / denom, axis=1)
        # C[k]=sum z[0..k-1]; linha i termina em C[i+window] => preço index t = i+window-1
        out[window - 1:] = stat

    # mascara onde a janela CSW não tem os `window` z's todos válidos (warm-up de z+janela)
    valid = _rolling_sum(finite, window) == window
    out[~valid] = np.nan
    return pd.Series(out, index=close.index)


def explosivity(close: pd.Series, min_w: int = 20, max_w: int = 80) -> pd.Series:
    """SADF-lite: supremo, sobre janelas recursivas terminando em t, da estatística ADF.

    Para cada comprimento de janela w em [min_w, max_w], roda a regressão ADF na janela
    [t-w+1, t] sobre y = log(close):

        Dy_s = alpha + rho * y_{s-1} + eps           (s na janela)
        ADF_w(t) = rho_hat / SE(rho_hat)

    explosivity_t = sup_w ADF_w(t). rho>0 (t-stat à direita) => raiz > 1 => preço em
    trajetória explosiva/bolha. É ortogonal a momentum: mede CURVATURA/aceleração
    auto-sustentada do nível, não o sinal do retorno acumulado.

    Vetorização: a OLS de 2 regressores tem forma fechada via somas móveis (cumsum),
    então cada w é O(L) em numpy e o loop externo tem só (max_w-min_w+1) iterações —
    NADA de statsmodels por janela. Mantém o painel inteiro bem abaixo de ~180s.
    """
    c = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log(c)
    L = y.shape[0]
    if L < min_w + 2:
        return pd.Series(np.full(L, np.nan), index=close.index)

    dy = np.diff(y, prepend=np.nan)          # Dy_s = y_s - y_{s-1}, alinhado em s
    ylag = np.concatenate([[np.nan], y[:-1]])  # y_{s-1}, alinhado em s
    # regressão usa pares (ylag_s, dy_s) válidos a partir de s=1
    Lr = ylag                                # regressor (nível defasado)
    D = dy                                   # dependente (diferença)

    # somas móveis dos termos da OLS (NaN no s=0 vira 0; janela só começa após warm-up)
    Lr0 = np.where(np.isfinite(Lr), Lr, 0.0)
    D0 = np.where(np.isfinite(D), D, 0.0)
    fin = (np.isfinite(Lr) & np.isfinite(D)).astype(float)

    best = np.full(L, -np.inf)
    for w in range(min_w, max_w + 1):
        n = _rolling_sum(fin, w)
        Sx = _rolling_sum(Lr0, w)
        Sy = _rolling_sum(D0, w)
        Sxx = _rolling_sum(Lr0 * Lr0, w)
        Sxy = _rolling_sum(Lr0 * D0, w)
        Syy = _rolling_sum(D0 * D0, w)

        with np.errstate(invalid="ignore", divide="ignore"):
            den = n * Sxx - Sx * Sx
            rho = (n * Sxy - Sx * Sy) / den
            alpha = (Sy - rho * Sx) / n
            ssr = Syy - alpha * Sy - rho * Sxy
            sigma2 = ssr / (n - 2.0)
            var_rho = sigma2 * n / den
            tstat = rho / np.sqrt(var_rho)

        # só onde a janela está cheia de pares válidos e a OLS é bem-posta
        ok = (n == w) & np.isfinite(tstat) & (den > 0) & (sigma2 > 0)
        cand = np.where(ok, tstat, -np.inf)
        best = np.maximum(best, cand)

    out = np.where(np.isfinite(best), best, np.nan)
    return pd.Series(out, index=close.index)


def vol_regime(close: pd.Series, short: int = 10, long: int = 63) -> pd.Series:
    """Razão vol curta / vol longa dos log-retornos: expansão (>1) vs contração (<1)
    de regime de volatilidade. Ambas trailing (só passado). Ortogonal a direção."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = pd.Series(np.diff(np.log(close.to_numpy(dtype=float)), prepend=np.nan),
                      index=close.index)
    vs = r.rolling(short).std()
    vl = r.rolling(long).std()
    return (vs / vl.replace(0, np.nan)).rename(None)


def _regime_feats(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"]
    g["cusum_vol"] = cusum_vol_robust(c, window=63).to_numpy()
    g["explosivity"] = explosivity(c, min_w=20, max_w=80).to_numpy()
    g["vol_regime"] = vol_regime(c, short=10, long=63).to_numpy()
    return g


def make_regime_features(panel_or_group: pd.DataFrame) -> pd.DataFrame:
    """Adiciona REGIME_FEATURES (time-series por ticker). Aceita o painel inteiro
    (groupby ticker) ou um único grupo de ticker. Não faz rank cross-section — isso
    é responsabilidade de features.make_features a jusante."""
    if "ticker" in panel_or_group.columns and panel_or_group["ticker"].nunique() > 1:
        return pd.concat(
            [_regime_feats(g) for _, g in panel_or_group.groupby("ticker", sort=False)],
            ignore_index=True,
        )
    return _regime_feats(panel_or_group)
