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

import json
from itertools import combinations
from math import comb
from pathlib import Path

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

    splits = []
    for test_groups in combinations(range(n_groups), n_test):
        test_mask = np.isin(grp, test_groups)
        train_mask = ~test_mask
        for tg in test_groups:
            tstart, tend = ranges[tg]
            # purge: treino cujo label resolve (t1) depois do início do teste e que
            # começou antes do fim do teste -> vaza, remove
            overlap = (t1 >= tstart) & (dates <= tend)
            # embargo em dias-DE-PREGÃO (não calendário): pega o índice de tend no
            # array 'uniq' de datas-de-pregão e embarga as próximas embargo_days
            # entradas (clamp no fim). HORIZON dias-úteis ~= 14 calendário; usar
            # timedelta 'D' embargaria curto demais e vazaria o label sobreposto.
            end_idx = int(np.searchsorted(uniq, tend))
            emb_dates = uniq[end_idx + 1 : end_idx + 1 + embargo_days]
            if len(emb_dates):
                embargoed = (dates > tend) & (dates <= emb_dates[-1])
            else:
                embargoed = np.zeros(len(dates), dtype=bool)
            train_mask &= ~(overlap | embargoed)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def effective_n(test_dates, horizon: int = C.HORIZON) -> int:
    """n EFETIVO (aprox. não-sobreposto) pra PSR/DSR.

    Cada data de teste aparece em ~k/N * C(N,k) splits (duplicada) e os retornos do
    triple-barrier são sobrepostos por HORIZON dias. Usar len(pooled) infla
    maciçamente sqrt(n-1) e mente sobre a confiança. n_eff = (nº de datas de teste
    ÚNICAS) / horizon aproxima a contagem de observações independentes.
    """
    d = np.asarray(test_dates, dtype="datetime64[ns]")
    n_unique = int(len(np.unique(d)))
    return max(n_unique // max(int(horizon), 1), 1)


def n_paths(n_groups: int = 6, n_test: int = 2) -> int:
    """Nº de paths de backtest gerados pelo CPCV: C(N,k)*k/N."""
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


def deflated_sharpe(sr, sr_trials, n, skew=0.0, kurt=3.0, n_trials=None):
    """DSR: PSR com benchmark = Sharpe máximo ESPERADO sob N tentativas nulas.

    sr_trials: Sharpes das TENTATIVAS DE RESEARCH (configs distintos já testados —
    multiple testing), NÃO os paths CPCV de uma única config. A dispersão dos paths
    é variância do ESTIMADOR de uma config; usá-la aqui confunde ruído de estimação
    com viés de seleção. Veja o trials log (append_trial/load_trials).

    n_trials: nº REAL de tentativas de research. Default = len(sr_trials) pra
    backward-compat; passe explícito quando sr_trials já for o log de trials.
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    if len(sr_trials) < 2:
        return np.nan
    n_trials = len(sr_trials) if n_trials is None else int(n_trials)
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


# --- trials log (multiple-testing real, não dispersão de paths) ---------------
# Registra (config_name -> sharpe) num JSON. Cada config testado é UMA tentativa de
# research; a variância ENTRE esses Sharpes é o insumo correto do DSR. Não confundir
# com a variância entre paths CPCV (essa é dispersão do estimador de UMA config e
# deve ser reportada como tal, separadamente).

def _trials_path(path=None) -> Path:
    return Path(path) if path is not None else C.PROC_DIR / "trials_log.json"


def load_trials(path=None) -> dict:
    """Carrega o log {config_name: sharpe}. {} se não existe."""
    p = _trials_path(path)
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, ValueError):
        return {}


def append_trial(config_name: str, sharpe, path=None) -> dict:
    """Registra/atualiza o Sharpe de uma config no log e persiste."""
    trials = load_trials(path)
    trials[config_name] = float(sharpe)
    _trials_path(path).write_text(json.dumps(trials, indent=2, sort_keys=True))
    return trials


def seed_trials(seed: dict, path=None) -> dict:
    """Semeia o log com tentativas conhecidas sem sobrescrever as já presentes."""
    trials = load_trials(path)
    changed = False
    for k, v in seed.items():
        if k not in trials:
            trials[k] = float(v)
            changed = True
    if changed:
        _trials_path(path).write_text(json.dumps(trials, indent=2, sort_keys=True))
    return trials


def trial_sharpes(path=None) -> np.ndarray:
    """Array dos Sharpes de todas as tentativas no log."""
    return np.asarray(list(load_trials(path).values()), dtype=float)


def sharpe(returns) -> float:
    """Sharpe NÃO anualizado de uma série de retornos."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))
