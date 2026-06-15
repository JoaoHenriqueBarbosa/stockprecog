"""Sample uniqueness e sample weights (López de Prado, AFML cap. 4).

Labels de triple-barrier se sobrepõem no tempo ([t_i, t1_i]), violando IID.
Este módulo mede concorrência/unicidade e deriva pesos de amostra (atribuição de
retorno + decaimento temporal) — computados POR TICKER e concatenados no painel.
Não edita nenhum módulo existente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def num_concurrent_events(close_dates: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Concorrência por barra (AFML 4.1): nº de labels [t_i, t1_i] que cobrem cada barra.

    close_dates: timeline (barras) do ticker. t1: Series indexada por t_i com a
    data de resolução de cada label. Retorna count por barra.
    """
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(0, index=close_dates, dtype="int64")
    # janela útil: do 1º início até a maior resolução
    idx = close_dates[(close_dates >= t1.index[0]) & (close_dates <= t1.max())]
    count = pd.Series(0, index=idx, dtype="int64")
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count.reindex(close_dates, fill_value=0)


def average_uniqueness(t1: pd.Series, num_co: pd.Series) -> pd.Series:
    """Unicidade média por label (AFML 4.2): média de 1/concorrência em [t_i, t1_i].

    Range (0, 1]: 1 = label isolado, ->0 = muito sobreposto. Indexada por t_i.
    """
    t1 = t1.dropna()
    inv = (1.0 / num_co.replace(0, np.nan))
    out = pd.Series(index=t1.index, dtype="float64")
    for t_in, t_out in t1.items():
        out.loc[t_in] = inv.loc[t_in:t_out].mean()
    return out


def _ind_matrix(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
    """Indicator matrix (AFML 4.3): barras (linhas) x labels (colunas), 1 se a barra
    pertence a [t_i, t1_i]."""
    t1 = t1.dropna()
    m = pd.DataFrame(0, index=bar_index, columns=range(len(t1)), dtype="int8")
    for j, (t_in, t_out) in enumerate(t1.items()):
        m.loc[t_in:t_out, j] = 1
    return m


def sequential_bootstrap(ind_matrix: pd.DataFrame, size: int | None = None) -> list[int]:
    """Sequential bootstrap (AFML 4.5): amostra labels favorecendo baixa sobreposição.

    A cada draw, a prob. de um label é proporcional à sua unicidade média dado o que
    já foi amostrado. ind_matrix: barras x labels (0/1). Retorna índices sorteados.
    """
    cols = ind_matrix.columns
    if size is None:
        size = len(cols)
    arr = ind_matrix.to_numpy(dtype="float64")  # barras x labels
    col_len = arr.sum(axis=0)
    col_len[col_len == 0] = 1.0  # evita div/0
    cum = np.zeros(arr.shape[0])  # concorrência acumulada das colunas já escolhidas
    phi: list[int] = []
    for _ in range(size):
        # unicidade média de cada coluna candidata dado 'cum'
        denom = cum[:, None] + arr  # barras x labels
        u = np.where(arr > 0, arr / denom, 0.0)
        avg_u = u.sum(axis=0) / col_len
        prob = avg_u / avg_u.sum()
        choice = np.random.choice(len(cols), p=prob)
        phi.append(int(cols[choice]))
        cum += arr[:, choice]
    return phi


def return_attribution_weights(t1: pd.Series, num_co: pd.Series,
                               close: pd.Series) -> pd.Series:
    """Pesos por atribuição de retorno (AFML 4.10): |Σ log-ret em [t_i,t1_i] / conc.|.

    Retornos repartidos pela concorrência (cada barra contribui 1/num_co). Indexada
    por t_i. Não-normalizado.
    """
    t1 = t1.dropna()
    ret = np.log(close / close.shift(1))
    inv = (1.0 / num_co.replace(0, np.nan))
    w = pd.Series(index=t1.index, dtype="float64")
    for t_in, t_out in t1.items():
        w.loc[t_in] = (ret.loc[t_in:t_out] * inv.loc[t_in:t_out]).sum()
    return w.abs()


def time_decay(weights: pd.Series, last_w: float = 1.0) -> pd.Series:
    """Decaimento linear por idade (AFML 4.11): pondera unicidade acumulada.

    last_w em [0,1]: peso da obs. mais antiga (1=sem decaimento; 0=linear até zero;
    <0 trunca as mais antigas). weights deve estar ordenada por tempo (t_i crescente).
    """
    clf = weights.sort_index().cumsum()
    if clf.empty:
        return weights
    total = clf.iloc[-1]
    if last_w >= 0:
        slope = (1.0 - last_w) / total
    else:
        slope = 1.0 / ((last_w + 1.0) * total)
    const = 1.0 - slope * total
    decay = const + slope * clf
    decay[decay < 0] = 0.0
    return weights.sort_index() * decay


def _ticker_weights(g: pd.DataFrame, decay_last_w: float | None,
                    mode: str = "return_attr") -> pd.Series:
    """Pesos brutos p/ um ticker. mode='return_attr' (4.10, |Σret/conc|, enfatiza
    movimentos grandes) ou 'uniqueness' (4.2, só corrige sobreposição). Retorna
    Series alinhada ao índice original de g (NaN-t1 -> 0)."""
    g = g.sort_values("date")
    bars = pd.DatetimeIndex(g["date"])
    t1 = pd.Series(g["t1"].to_numpy(), index=bars)
    close = pd.Series(g["close"].to_numpy(), index=bars)

    valid = t1.notna()
    raw = pd.Series(0.0, index=g.index)
    if not valid.any():
        return raw

    t1v = t1[valid]
    num_co = num_concurrent_events(bars, t1v)
    if mode == "uniqueness":
        w = average_uniqueness(t1v, num_co)  # 1/conc média, sem magnitude de retorno
    else:
        w = return_attribution_weights(t1v, num_co, close)
    if decay_last_w is not None:
        w = time_decay(w, last_w=decay_last_w)

    # mapeia de volta pro índice de linhas de g via a data
    date_to_w = w
    aligned = pd.Series(bars.map(lambda d: date_to_w.get(d, 0.0)),
                        index=g.index, dtype="float64")
    return aligned.fillna(0.0)


def panel_sample_weights(labeled_df: pd.DataFrame,
                         decay_last_w: float | None = None,
                         mode: str = "return_attr") -> pd.Series:
    """Orquestra os pesos no painel multi-ticker (AFML cap. 4).

    Espera colunas date, ticker, t1, close. Computa por ticker (concorrência só
    dentro da timeline de cada um) e concatena. Labels sem t1 -> peso 0. Retorna
    Series alinhada ao índice de labeled_df, normalizada a média 1 (sobre os > 0).
    mode: 'return_attr' (4.10) ou 'uniqueness' (4.2).
    """
    parts = [
        _ticker_weights(g, decay_last_w, mode)
        for _, g in labeled_df.groupby("ticker", sort=False)
    ]
    w = pd.concat(parts).reindex(labeled_df.index).fillna(0.0)

    pos = w > 0
    mean_pos = w[pos].mean()
    if mean_pos and np.isfinite(mean_pos) and mean_pos > 0:
        w = w / mean_pos
    return w.rename("sample_weight")
