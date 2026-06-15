"""Combinatorial Purged Cross-Validation (López de Prado, AFML cap. 7 e 12) +
Deflated / Probabilistic Sharpe Ratio (cap. 14).

CPCV: divide a linha do tempo em N grupos contíguos, escolhe k grupos pra teste
em cada combinação (C(N,k) splits), purga do treino observações cujo label
(janela [date, t1]) vaza pro período de teste, e aplica embargo após cada bloco
de teste. Gera C(N,k) estimativas independentes → distribuição de performance, em
vez de um único número que mente.

DSR/PSR: corrigem o Sharpe pela quantidade de tentativas (multiple testing) e pela
não-normalidade dos retornos. É a régua anti-autoengano.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.stats import norm

from . import config as C

_EULER = 0.5772156649015329


def cpcv_splits(dates, t1, n_groups: int = 6, n_test: int = 2,
                embargo_days: int = C.EMBARGO_DAYS):
    """Gera (train_idx, test_idx) pra cada combinação de k grupos de teste.

    dates, t1: arrays datetime64 (observação e resolução do label).
    Retorna lista de C(n_groups, n_test) tuplas. Nº de paths = C(N,k)*k/N.
    """
    dates = np.asarray(dates, dtype="datetime64[ns]")
    t1 = np.asarray(t1, dtype="datetime64[ns]")
    uniq = np.sort(np.unique(dates))
    blocks = np.array_split(uniq, n_groups)

    date_to_group = {}
    ranges = []
    for gi, block in enumerate(blocks):
        ranges.append((block[0], block[-1]))
        for d in block:
            date_to_group[d] = gi
    grp = np.array([date_to_group[d] for d in dates])
    emb = np.timedelta64(embargo_days, "D")

    splits = []
    for test_groups in combinations(range(n_groups), n_test):
        test_mask = np.isin(grp, test_groups)
        train_mask = ~test_mask
        for tg in test_groups:
            tstart, tend = ranges[tg]
            # purge: treino cujo label resolve (t1) depois do início do teste e que
            # começou antes do fim do teste -> vaza, remove
            overlap = (t1 >= tstart) & (dates <= tend)
            # embargo: treino logo após o bloco de teste
            embargoed = (dates > tend) & (dates <= tend + emb)
            train_mask &= ~(overlap | embargoed)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def n_paths(n_groups: int = 6, n_test: int = 2) -> int:
    return comb(n_groups, n_test) * n_test // n_groups


def probabilistic_sharpe(sr, sr_benchmark, n, skew=0.0, kurt=3.0):
    """PSR: P(SR verdadeiro > benchmark) dado SR observado, n amostras, skew/kurt.
    sr e sr_benchmark NÃO anualizados (mesma frequência de n)."""
    if n < 2:
        return np.nan
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom == 0:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(sr, sr_trials, n, skew=0.0, kurt=3.0):
    """DSR: PSR com benchmark = Sharpe máximo ESPERADO sob N tentativas nulas.
    sr_trials: array de Sharpes das várias tentativas (pra estimar a variância)."""
    sr_trials = np.asarray(sr_trials, dtype=float)
    n_trials = len(sr_trials)
    if n_trials < 2:
        return np.nan
    var_sr = np.var(sr_trials, ddof=1)
    if var_sr == 0:
        return np.nan
    # Sharpe máximo esperado de N tentativas ~N(0, var_sr) (Bailey & LdP 2014)
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    sr_star = np.sqrt(var_sr) * ((1 - _EULER) * z1 + _EULER * z2)
    return probabilistic_sharpe(sr, sr_star, n, skew, kurt)


def sharpe(returns) -> float:
    """Sharpe NÃO anualizado de uma série de retornos."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))
