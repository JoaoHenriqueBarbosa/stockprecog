"""Invariantes do modelo de custo: round-trip monotônico crescente na participação
e com escala raiz-quadrada (lei de impacto de Toth et al.)."""
from __future__ import annotations

import numpy as np

from stockprecog.costs import (
    DEFAULT_HALF_SPREAD_BPS,
    apply_costs,
    impact_bps,
    round_trip_cost_bps,
)
import pandas as pd


def test_round_trip_monotonico_crescente_na_participacao():
    sigma = 0.02
    parts = np.linspace(0.0, 0.2, 20)
    costs = [round_trip_cost_bps(p, sigma) for p in parts]
    assert all(b >= a for a, b in zip(costs, costs[1:]))   # não-decrescente
    assert costs[-1] > costs[0]                            # estritamente cresce


def test_impacto_escala_com_raiz_quadrada():
    """impacto(4p) = 2*impacto(p): sqrt(4)=2. Confirma o expoente 1/2."""
    sigma = 0.03
    base = impact_bps(0.01, sigma)
    quad = impact_bps(0.04, sigma)
    assert np.isclose(quad, 2.0 * base)
    # e 9x participação → 3x impacto
    assert np.isclose(impact_bps(0.09, sigma), 3.0 * base)


def test_participacao_zero_e_so_spread():
    """Sem ordem (participation=0) o impacto é 0; round-trip = 2 pernas de spread."""
    assert impact_bps(0.0, 0.02) == 0.0
    rt = round_trip_cost_bps(0.0, 0.02, half_spread_bps=DEFAULT_HALF_SPREAD_BPS)
    assert np.isclose(rt, 2.0 * DEFAULT_HALF_SPREAD_BPS)


def test_participacao_negativa_clampeada():
    """|Q|/ADV não pode ser <0; participação negativa é clampada (impacto 0)."""
    assert impact_bps(-0.5, 0.02) == 0.0


def test_round_trip_dobra_a_perna():
    """round_trip = 2 * (spread + impacto da perna)."""
    p, sigma = 0.05, 0.02
    one_leg = DEFAULT_HALF_SPREAD_BPS + impact_bps(p, sigma)
    assert np.isclose(round_trip_cost_bps(p, sigma), 2.0 * one_leg)


def test_apply_costs_desconta_proporcional_ao_turnover():
    rets = pd.Series([0.01, 0.02, -0.005], index=[0, 1, 2])
    turnover = pd.Series([0.0, 0.5, 1.0], index=[0, 1, 2])
    net = apply_costs(rets, turnover, participation=0.05, sigma=0.02)
    # turnover 0 → retorno intacto
    assert np.isclose(net.iloc[0], rets.iloc[0])
    # turnover > 0 → líquido < bruto
    assert net.iloc[1] < rets.iloc[1]
    assert net.iloc[2] < rets.iloc[2]
    # custo proporcional: turnover 1.0 desconta o dobro do turnover 0.5
    cost_half = rets.iloc[1] - net.iloc[1]
    cost_full = rets.iloc[2] - net.iloc[2]
    assert np.isclose(cost_full, 2.0 * cost_half)
