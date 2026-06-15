"""Invariantes do CPCV: embargo em dias-DE-PREGÃO (R2), purga via t1 sem vazamento,
n efetivo << len(pooled), e PSR/DSR no range correto."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockprecog import cpcv


def _trading_dates(n: int, start="2020-01-01"):
    return pd.bdate_range(start=start, periods=n).to_numpy().astype("datetime64[ns]")


def test_embargo_remove_numero_certo_de_barras_de_pregao():
    """Embargo conta POSIÇÕES no array de datas-de-pregão (não dias calendário).
    Com 1 ticker, 1 data por barra e t1=date (sem overlap de purga), o treino logo
    após cada bloco de teste deve perder exatamente `embargo_days` barras."""
    n = 60
    dates = _trading_dates(n)
    t1 = dates.copy()  # resolve na própria barra → purga não remove nada extra
    embargo = 5

    splits = cpcv.cpcv_splits(dates, t1, n_groups=6, n_test=1, embargo_days=embargo)
    uniq = np.sort(np.unique(dates))
    blocks = np.array_split(uniq, 6)

    # split do grupo de teste 0 (primeiro bloco): logo após o fim do teste devem
    # faltar exatamente `embargo` barras no treino.
    tr, te = splits[0]
    tend = blocks[0][-1]
    end_idx = int(np.searchsorted(uniq, tend))
    embargoed_expected = uniq[end_idx + 1: end_idx + 1 + embargo]

    train_dates = dates[tr]
    # nenhuma data embargada no treino
    assert not np.isin(embargoed_expected, train_dates).any()
    # e a barra imediatamente após o embargo VOLTA ao treino (clamp correto)
    after = uniq[end_idx + 1 + embargo]
    assert np.isin(after, train_dates).any()
    assert len(embargoed_expected) == embargo


def test_purge_t1_nao_deixa_label_vazar_no_teste():
    """Toda obs de treino cujo label resolve (t1) DENTRO da janela de teste deve
    ser purgada: não pode existir treino com date<=tend e t1>=tstart."""
    n = 60
    dates = _trading_dates(n)
    uniq = np.sort(np.unique(dates))
    # labels com horizonte longo (10 barras) → muito overlap pra purgar
    idx = np.searchsorted(uniq, dates)
    res_idx = np.minimum(idx + 10, n - 1)
    t1 = uniq[res_idx]

    splits = cpcv.cpcv_splits(dates, t1, n_groups=6, n_test=2, embargo_days=0)
    blocks = np.array_split(uniq, 6)

    for tr, te in splits:
        test_groups = sorted({int(np.searchsorted(
            [b[-1] for b in blocks], d, side="left")) for d in dates[te]})
        for tg in test_groups:
            tstart, tend = blocks[tg][0], blocks[tg][-1]
            leak = (dates[tr] <= tend) & (t1[tr] >= tstart)
            assert not leak.any(), "label de treino vaza dentro do teste"


def test_embargo_dia_de_pregao_difere_de_calendario():
    """Em fim de semana/feriado, N barras-de-pregão cobrem MAIS que N dias-D de
    calendário. O embargo posicional deve refletir isso (não subestimar)."""
    n = 40
    dates = _trading_dates(n)
    t1 = dates.copy()
    embargo = 10
    splits = cpcv.cpcv_splits(dates, t1, n_groups=4, n_test=1, embargo_days=embargo)
    uniq = np.sort(np.unique(dates))
    blocks = np.array_split(uniq, 4)

    tr, _ = splits[0]
    tend = blocks[0][-1]
    end_idx = int(np.searchsorted(uniq, tend))
    last_embargoed = uniq[end_idx + embargo]   # 10ª barra-de-pregão após o teste
    span_days = (last_embargoed - tend) / np.timedelta64(1, "D")
    # 10 pregões abarcam >10 dias de calendário (fins de semana no meio)
    assert span_days > embargo
    assert not np.isin(last_embargoed, dates[tr]).any()


def test_effective_n_muito_menor_que_pooled():
    """n_eff = datas únicas // HORIZON, bem abaixo do nº de observações empilhadas
    (que duplica datas entre splits e sobrepõe retornos)."""
    n = 200
    dates = _trading_dates(n)
    # pooled simula empilhamento: cada data repetida ~3x entre paths
    pooled = np.concatenate([dates, dates, dates])
    n_eff = cpcv.effective_n(dates, horizon=10)
    assert n_eff == len(np.unique(dates)) // 10
    assert n_eff < len(pooled)
    assert n_eff < len(np.unique(dates))


def test_n_paths_formula():
    assert cpcv.n_paths(6, 2) == 5     # C(6,2)*2/6 = 15*2/6
    assert cpcv.n_paths(4, 1) == 1


def test_probabilistic_sharpe_no_intervalo_0_1():
    assert np.isnan(cpcv.probabilistic_sharpe(0.1, 0.0, n=1))
    for sr in (-0.5, 0.0, 0.05, 0.3):
        p = cpcv.probabilistic_sharpe(sr, 0.0, n=200)
        assert 0.0 <= p <= 1.0
    # SR muito acima do benchmark com n grande → prob alta
    hi = cpcv.probabilistic_sharpe(0.5, 0.0, n=500)
    lo = cpcv.probabilistic_sharpe(-0.5, 0.0, n=500)
    assert hi > 0.5 > lo


def test_deflated_sharpe_range_e_n_trials():
    """DSR é uma PSR contra o Sharpe-máximo-esperado das tentativas → fica em [0,1].
    Menos de 2 trials finitos → NaN."""
    trials = [0.0, -0.022, 0.028, -0.088, 0.03]
    dsr = cpcv.deflated_sharpe(0.03, trials, n=300, n_trials=len(trials))
    assert 0.0 <= dsr <= 1.0
    assert np.isnan(cpcv.deflated_sharpe(0.03, [0.01], n=300))
    assert np.isnan(cpcv.deflated_sharpe(0.03, [np.nan, np.nan], n=300))


def test_sharpe_basico():
    assert np.isnan(cpcv.sharpe([1.0]))
    assert np.isnan(cpcv.sharpe([2.0, 2.0, 2.0]))  # std 0
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    assert np.isfinite(cpcv.sharpe(r))
