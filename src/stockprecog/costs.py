"""Modelo de custo de transação para estratégias cross-section na B3.

Combina dois componentes por ida-e-volta:

  custo_total = spread/corretagem (linear)  +  market impact (raiz quadrada)

O termo de impacto segue a *lei da raiz quadrada* (Toth et al. 2011,
"Anomalous price impact and the critical nature of liquidity"):

    impacto ~ eta * sigma * sqrt(Q / ADV)

onde Q é o tamanho da ordem, ADV é o volume médio diário NEGOCIADO EM R$ e
sigma é a volatilidade diária do ativo. A raiz quadrada (côncava em Q) é
empírica e robusta entre mercados; é mais realista que o impacto linear do
modelo de Almgren-Chriss (1999), que aqui serve só como contexto histórico —
AC usa impacto linear/temporário e é calibrado pra execução agendada, não pra
o custo agregado que queremos descontar do retorno da estratégia.

ATENÇÃO — calibração para mid-caps B3:
  - `half_spread_bps` e `eta` são PISO CONSERVADOR. Em ativos de ADV baixo o
    impacto é MAIS convexo do que a raiz quadrada captura (o expoente sobe e o
    spread alarga sob estresse de liquidez), então estes números SUBESTIMAM o
    custo real na cauda. Trate como limite inferior, não como estimativa central.
  - Enquanto o scrape (brapi_ingest) não trouxer volume financeiro real em R$,
    o `participation = |Q|/ADV` é um PLACEHOLDER: sem ADV real não há como
    ancorar a participação. Os bps retornados são ilustrativos até o volume
    em R$ entrar no painel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pisos conservadores para mid-caps B3 (ver nota de calibração no módulo).
DEFAULT_HALF_SPREAD_BPS = 10.0  # meia-corretagem + meio-spread, por perna
DEFAULT_ETA = 1.0               # coeficiente de impacto (adimensional)


def impact_bps(participation: float, sigma: float, eta: float = DEFAULT_ETA) -> float:
    """Custo de market impact em bps pela lei da raiz quadrada.

    impacto_bps = eta * sigma * sqrt(participation) * 1e4

    Args:
        participation: |Q| / ADV — fração do volume diário (R$) consumida pela
            ordem. Adimensional, tipicamente 1e-3 a 1e-1.
        sigma: volatilidade diária do ativo (ex.: 0.02 = 2% ao dia).
        eta: coeficiente de impacto. Piso conservador = 1.0.

    Returns:
        Custo de impacto em basis points (1 bps = 0.01%).
    """
    participation = np.maximum(participation, 0.0)
    return eta * sigma * np.sqrt(participation) * 1e4


def round_trip_cost_bps(
    participation: float,
    sigma: float,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    eta: float = DEFAULT_ETA,
) -> float:
    """Custo de ida-e-volta em bps: spread/corretagem + impacto, nas duas pernas.

    Cada perna (entrada e saída) paga `half_spread_bps` de spread/corretagem
    mais um impacto. Como o impacto da raiz quadrada já é por execução, a
    ida-e-volta dobra ambos os componentes.

    Returns:
        Custo total de round-trip em basis points.
    """
    one_leg = half_spread_bps + impact_bps(participation, sigma, eta)
    return 2.0 * one_leg


def apply_costs(
    daily_returns: pd.Series,
    turnover: pd.Series,
    participation: float | pd.Series,
    sigma: float | pd.Series,
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS,
    eta: float = DEFAULT_ETA,
) -> pd.Series:
    """Desconta custo de transação dos retornos da estratégia por período.

    O custo cobrado em cada período é proporcional ao turnover (mudança de
    pesos, em [0, 1] onde 1 = troca total do book naquele período). Aplica-se
    apenas a meia-corretagem/spread + impacto da *perna* executada — não a
    ida-e-volta — porque o turnover já contabiliza cada lado da rotação ao
    longo do tempo (vender hoje e recomprar amanhã aparece como turnover nos
    dois períodos).

    Args:
        daily_returns: retornos brutos da estratégia, indexados por período.
        turnover: fração do portfólio rotacionada no período (>= 0).
        participation: |Q|/ADV; escalar ou série alinhada a `daily_returns`.
        sigma: vol diária; escalar ou série alinhada.
        half_spread_bps: spread/corretagem por perna (piso conservador).
        eta: coeficiente de impacto (piso conservador).

    Returns:
        Série de retornos líquidos (mesmo índice de `daily_returns`).
    """
    cost_per_leg_bps = half_spread_bps + impact_bps(participation, sigma, eta)
    cost_frac = turnover * cost_per_leg_bps / 1e4
    return daily_returns - cost_frac
