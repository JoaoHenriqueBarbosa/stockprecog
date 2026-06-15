"""Invariantes de sample uniqueness / weights (AFML cap. 4):
average_uniqueness em (0,1], menor sob mais sobreposição; caso construído com
overlap conhecido fecha o número."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockprecog.weights import (
    average_uniqueness,
    num_concurrent_events,
    panel_sample_weights,
)


def _bars(n: int):
    return pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n))


def test_label_isolado_tem_uniqueness_1():
    bars = _bars(6)
    # dois labels que NÃO se sobrepõem: cada um resolve na própria barra
    t1 = pd.Series([bars[0], bars[2]], index=[bars[0], bars[2]])
    num_co = num_concurrent_events(bars, t1)
    u = average_uniqueness(t1, num_co)
    assert np.allclose(u.to_numpy(), 1.0)


def test_uniqueness_no_intervalo_0_1_e_menor_com_overlap():
    bars = _bars(6)
    # label A: bar0..bar3 ; label B: bar1..bar3  → sobrepõem bars 1..3
    t1 = pd.Series([bars[3], bars[3]], index=[bars[0], bars[1]])
    num_co = num_concurrent_events(bars, t1)
    # concorrência esperada: bar0=1, bar1=2, bar2=2, bar3=2
    assert num_co.loc[bars[0]] == 1
    assert num_co.loc[bars[1]] == 2
    assert num_co.loc[bars[2]] == 2
    assert num_co.loc[bars[3]] == 2

    u = average_uniqueness(t1, num_co)
    # range (0,1]
    assert (u > 0).all() and (u <= 1.0 + 1e-12).all()
    # A = mean(1/[1,2,2,2]) = 0.625 ; B = mean(1/[2,2,2]) = 0.5
    assert np.isclose(u.loc[bars[0]], 0.625)
    assert np.isclose(u.loc[bars[1]], 0.5)
    # ambos < 1 (há overlap) e o mais sobreposto (B) é o menor
    assert (u < 1.0).all()
    assert u.loc[bars[1]] < u.loc[bars[0]]


def test_mais_overlap_reduz_uniqueness():
    """Monotonicidade: aumentar a sobreposição (mesmo t_in, t1 mais longo) só pode
    DIMINUIR a unicidade média do label."""
    bars = _bars(8)
    # cenário leve: A resolve em bar2 (sobrepõe B em bar1..bar2)
    t1_leve = pd.Series([bars[2], bars[2]], index=[bars[0], bars[1]])
    u_leve = average_uniqueness(t1_leve, num_concurrent_events(bars, t1_leve))
    # cenário pesado: A resolve em bar5 (sobrepõe B por muito mais tempo)
    t1_pesado = pd.Series([bars[5], bars[5]], index=[bars[0], bars[1]])
    u_pesado = average_uniqueness(t1_pesado, num_concurrent_events(bars, t1_pesado))
    assert u_pesado.loc[bars[0]] < u_leve.loc[bars[0]]


def test_panel_sample_weights_normalizado_e_nao_negativo():
    """Pesos do painel: não-negativos, média 1 sobre os positivos, label sem t1→0."""
    bars = _bars(10)
    df = pd.DataFrame({
        "date": list(bars) * 2,
        "ticker": ["AAA"] * 10 + ["BBB"] * 10,
        "close": np.r_[100 + np.arange(10), 50 + np.arange(10)].astype(float),
        "t1": list(bars[np.minimum(np.arange(10) + 3, 9)]) * 2,
    })
    # injeta um label sem resolução → peso 0
    df.loc[0, "t1"] = pd.NaT

    w = panel_sample_weights(df)
    assert (w >= 0).all()
    assert w.iloc[0] == 0.0
    pos = w[w > 0]
    assert np.isclose(pos.mean(), 1.0)
    assert len(w) == len(df)
