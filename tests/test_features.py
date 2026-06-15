"""Causalidade da fractional differentiation (R1: ffill no warm-up da vol, nunca
bfill) — nenhuma saída pode depender de valor FUTURO da série.

Nota: a janela da FFD (d=0.4, thresh=1e-4) tem ~282 termos, então as séries aqui
precisam ser bem mais longas que isso pra existir qualquer saída não-NaN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockprecog.features import FD_D, FD_THRESH, _ts_feats
from stockprecog.fracdiff import ffd_weights, fracdiff_ffd
from tests.conftest import make_ticker

WIDTH = len(ffd_weights(FD_D, FD_THRESH))  # ~55 com thresh=1e-3


def test_fracdiff_ffd_nao_usa_futuro():
    """Mexer numa observação t NÃO pode alterar nenhuma saída em índice < t.
    (FFD é convolução causal de janela fixa: out[i] depende só de [i-w+1 .. i].)"""
    n = WIDTH + 300
    rng = np.random.default_rng(0)
    base = pd.Series(np.log(100 + np.cumsum(rng.normal(0, 1, n))))
    a = fracdiff_ffd(base, FD_D, FD_THRESH)

    perturbed = base.copy()
    t = WIDTH + 150
    perturbed.iloc[t] += 5.0          # choque grande numa barra futura
    b = fracdiff_ffd(perturbed, FD_D, FD_THRESH)

    # tudo antes de t é idêntico; de t em diante muda (a barra t entra na janela)
    assert np.allclose(a.iloc[:t].to_numpy(), b.iloc[:t].to_numpy(), equal_nan=True)
    assert not np.allclose(a.iloc[t:].to_numpy(), b.iloc[t:].to_numpy(),
                           equal_nan=True)


def test_ffd_primeira_linha_valida_independe_de_posterior():
    """A 1ª saída não-NaN da FFD usa só a janela que termina nela — perturbar
    qualquer barra DEPOIS dela não a altera."""
    n = WIDTH + 200
    s = pd.Series(np.log(np.linspace(100, 130, n)))
    out = fracdiff_ffd(s, FD_D, FD_THRESH)
    first_valid = out.first_valid_index()
    assert first_valid is not None
    assert first_valid == WIDTH - 1   # 1ª janela completa

    s2 = s.copy()
    s2.iloc[first_valid + 1:] += 3.0   # mexe em tudo depois da 1ª válida
    out2 = fracdiff_ffd(s2, FD_D, FD_THRESH)
    assert out.loc[first_valid] == out2.loc[first_valid]


def test_fd_vol21_ffill_nao_injeta_futuro_no_warmup():
    """fd_vol21 preenche a vol_21 com ffill antes da FFD. bfill puxaria o 1º valor
    de vol pra trás (futuro) e contaminaria o warm-up; ffill propaga só o passado,
    então o início permanece NaN até existir vol observada o bastante."""
    n = WIDTH + 120
    closes = 100 + np.cumsum(np.sin(np.arange(n) / 3.0))
    g = make_ticker(closes)
    feat = _ts_feats(g).sort_values("date").reset_index(drop=True)

    first_vol = feat["vol_21"].first_valid_index()
    assert first_vol >= 21

    # Com ffill, fd_vol21 só pode existir DEPOIS de acumular WIDTH valores de vol
    # (a partir de first_vol). bfill teria preenchido o início com o futuro e
    # produzido fd_vol21 cedo demais.
    fd = feat["fd_vol21"]
    fd_first = fd.first_valid_index()
    assert fd_first is not None
    assert fd_first >= first_vol + WIDTH - 1
    assert fd.iloc[:first_vol].isna().all()


def test_fd_vol21_causal_contra_choque_futuro():
    """Recalcular o ramo fd_vol21 com um choque numa barra futura não muda as
    saídas anteriores — prova de que ffill+FFD continua causal de ponta a ponta."""
    n = WIDTH + 150
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    g = make_ticker(closes)
    a = _ts_feats(g)["fd_vol21"].to_numpy()

    g2 = g.copy()
    t = n - 15
    g2.loc[t, ["open", "high", "low", "close"]] = closes[t] * 1.2
    b = _ts_feats(g2)["fd_vol21"].to_numpy()

    # alguma saída válida existe antes de t, e ela não muda
    assert np.isfinite(a[:t]).any()
    assert np.allclose(a[:t], b[:t], equal_nan=True)
