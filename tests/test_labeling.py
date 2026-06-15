"""Invariantes do triple-barrier (R1: fwd_ret medido no toque da barreira, não na
vertical) e o empate hit_up & hit_dn."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockprecog import config as C
from stockprecog.labeling import _label_ticker
from tests.conftest import make_ticker


def _oscillating_closes(n: int, base: float = 100.0, amp: float = 0.004):
    """Preços com micro-oscilação determinística: logret ~±amp → sigma finita e
    pequena, e (com high=low=close) NENHUM toque natural de barreira — todo label
    resolve na vertical, deixando o palco limpo pra injetar um toque controlado."""
    return base * (1.0 + amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0))


def test_fwd_ret_usa_barra_de_resolucao_nao_a_vertical():
    n = 40
    closes = _oscillating_closes(n)
    highs = closes.copy()
    lows = closes.copy()

    t0 = 25                      # sigma bem aquecida
    res = t0 + 3                 # toque ANTES da vertical (end = t0 + HORIZON)
    assert res < t0 + C.HORIZON
    # spike de high só nesta barra: força profit-take em k=res p/ a obs em t0
    highs[res] = closes[res] * 1.5

    g = make_ticker(closes, highs, lows)
    out = _label_ticker(g).sort_values("date").reset_index(drop=True)
    close = out["close"].to_numpy()
    dates = out["date"].to_numpy()

    # label de t0: profit-take, resolvido em 'res'
    assert out["label"].iloc[t0] == 1
    assert out["barrier"].iloc[t0] == "pt"
    assert dates[t0 + 3] == np.datetime64(out["t1"].iloc[t0])

    # fwd_ret = close[res]/close[t0]-1  (R1) — NÃO a vertical close[end]
    expected_res = close[res] / close[t0] - 1.0
    vertical = close[t0 + C.HORIZON] / close[t0] - 1.0
    got = out["fwd_ret"].iloc[t0]
    assert got == np.float64(expected_res)
    # e os dois valores DIVERGEM (senão o teste não discriminaria o bug)
    assert not np.isclose(expected_res, vertical)


def test_fwd_ret_consistente_com_t1_em_todas_as_linhas():
    """Invariante global: para toda obs com label, fwd_ret casa com close na data
    de resolução (t1), nunca com a barra vertical quando elas diferem."""
    n = 40
    closes = _oscillating_closes(n)
    highs = closes.copy()
    lows = closes.copy()
    highs[20] = closes[20] * 1.5   # alguns toques antecipados espalhados
    lows[31] = closes[31] * 0.5

    out = _label_ticker(make_ticker(closes, highs, lows))
    out = out.sort_values("date").reset_index(drop=True)
    close = out["close"].to_numpy()
    date_to_idx = {d: i for i, d in enumerate(out["date"].to_numpy())}

    labeled = out[out["label"].notna()]
    assert len(labeled) > 0
    for i, row in labeled.iterrows():
        res = date_to_idx[np.datetime64(row["t1"])]
        assert row["fwd_ret"] == np.float64(close[res] / close[i] - 1.0)


def test_empate_hit_up_e_hit_dn_vira_stop():
    """Barra que estoura PT e SL ao mesmo tempo → conservador: label 0 (stop)."""
    n = 40
    closes = _oscillating_closes(n)
    highs = closes.copy()
    lows = closes.copy()

    t0 = 25
    res = t0 + 2
    highs[res] = closes[res] * 1.5   # estoura up
    lows[res] = closes[res] * 0.5    # estoura dn na MESMA barra

    out = _label_ticker(make_ticker(closes, highs, lows))
    out = out.sort_values("date").reset_index(drop=True)

    assert out["label"].iloc[t0] == 0
    assert out["barrier"].iloc[t0].startswith("both")
    # resolve nessa barra, fwd_ret medido no close dela (não na vertical)
    close = out["close"].to_numpy()
    assert out["fwd_ret"].iloc[t0] == np.float64(close[res] / close[t0] - 1.0)


def test_sigma_nan_nao_gera_label():
    """Primeira barra (sigma NaN no warm-up do EWMA) não recebe label."""
    closes = _oscillating_closes(15)
    out = _label_ticker(make_ticker(closes))
    assert pd.isna(out["label"].iloc[0])
