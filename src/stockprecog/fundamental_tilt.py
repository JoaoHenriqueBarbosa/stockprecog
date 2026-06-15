"""Teste PRÉ-REGISTRADO (uma vez): tilt fundamental value+quality, low-turnover.

Hipótese H1 (registrada antes de rodar): long-short value+quality rebalanceado
TRIMESTRALMENTE (63 pregões) tem net Sharpe > 0 após custos one-way (5bps, R$2M),
porque o turnover baixo mantém o pedágio abaixo do edge fundamental fraco-mas-real.

Construção FIXA (sem tuning):
- score = média dos ranks cross-section de {earnings_yield, book_to_price, profit_margin}
- long tercil superior, short tercil inferior, equal-weight, market-neutral
- rebalance a cada REBAL=63 pregões; retorno = forward-return do nome até o próximo rebal
- net = bruto - turnover * custo one-way (5bps + impacto lei-sqrt @R$2M)

Sucesso: net Sharpe > 0. Falha: <= 0 -> kill-criterion confirmado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import cpcv
from .brapi_ingest import load_for_pipeline
from .costs import impact_bps
from .features_fundamental import make_fundamental_features

REBAL = 63          # pregões entre rebalances (~trimestre)
CAPITAL = 2e6
HALF_SPREAD_BPS = 5.0
SIGMA = 0.025
SCORE_FACTORS = ["earnings_yield", "book_to_price", "profit_margin"]


def run() -> dict:
    """Teste pré-registrado do tilt fundamental value/quality, long-short tercil, rebal trimestral."""
    panel = load_for_pipeline()
    adv_map = (panel.assign(dv=panel["close"] * panel["volume"])
               .groupby("ticker")["dv"].median().to_dict())
    ff = make_fundamental_features(panel)

    # score = média dos ranks cross-section dos fatores (point-in-time: fatores já lagados)
    for f in SCORE_FACTORS:
        ff[f"{f}_r"] = ff.groupby("date")[f].rank(pct=True)
    ff["score"] = ff[[f"{f}_r" for f in SCORE_FACTORS]].mean(axis=1)

    # retorno forward até o próximo rebal, por ticker
    ff = ff.sort_values(["ticker", "date"]).reset_index(drop=True)
    ff["fwd"] = ff.groupby("ticker")["close"].shift(-REBAL) / ff["close"] - 1.0

    dates = np.sort(ff["date"].unique())[::REBAL]
    prev_w: dict = {}
    gross, net = [], []
    for d in dates:
        day = ff[(ff["date"] == d) & ff["score"].notna() & ff["fwd"].notna()]
        if len(day) < 12:
            continue
        hi = day["score"].quantile(2 / 3); lo = day["score"].quantile(1 / 3)
        L = day[day["score"] >= hi]; S = day[day["score"] <= lo]
        if len(L) == 0 or len(S) == 0:
            continue
        w = {t: 0.5 / len(L) for t in L["ticker"]}
        w.update({t: -0.5 / len(S) for t in S["ticker"]})
        fwd = dict(zip(day["ticker"], day["fwd"]))
        gr = sum(w[t] * fwd[t] for t in w if np.isfinite(fwd.get(t, np.nan)))
        cost = 0.0
        for t in set(w) | set(prev_w):
            dw = w.get(t, 0.0) - prev_w.get(t, 0.0)
            if dw == 0:
                continue
            adv = adv_map.get(t, np.nan)
            part = abs(dw) * CAPITAL / adv if (np.isfinite(adv) and adv > 0) else 0.01
            cost += abs(dw) * (HALF_SPREAD_BPS + impact_bps(part, SIGMA)) / 1e4
        gross.append(gr); net.append(gr - cost); prev_w = w

    gross = np.array(gross); net = np.array(net)
    sr_g, sr_n = cpcv.sharpe(gross), cpcv.sharpe(net)
    # turnover médio
    cpcv.append_trial("fundamental_tilt_quarterly", sr_n)
    sr_tr = cpcv.trial_sharpes()
    psr = cpcv.probabilistic_sharpe(sr_n, 0.0, len(net),
                                    skew=float(pd.Series(net).skew()),
                                    kurt=float(pd.Series(net).kurt() + 3)) if len(net) > 2 else np.nan
    dsr = cpcv.deflated_sharpe(sr_n, sr_tr, len(net), n_trials=len(sr_tr)) if len(sr_tr) >= 2 else np.nan

    print("=== TESTE PRÉ-REGISTRADO: tilt fundamental trimestral ===")
    print(f"rebalances: {len(net)}  ({REBAL} pregões cada, ~trimestral)")
    print(f"Sharpe BRUTO   = {sr_g:+.3f}")
    print(f"Sharpe LÍQUIDO = {sr_n:+.3f}   <- critério: >0 ?  {'SUCESSO' if sr_n > 0 else 'FALHA'}")
    print(f"PSR(net>0)     = {psr:.3f}")
    print(f"DSR (deflac. {len(sr_tr)} trials) = {dsr:.3f}")
    ann = sr_n * np.sqrt(252 / REBAL) if np.isfinite(sr_n) else np.nan
    print(f"Sharpe líq anualizado ~ {ann:+.3f}")
    return {"sharpe_gross": sr_g, "sharpe_net": sr_n, "psr": psr, "dsr": dsr,
            "n_rebal": len(net), "success": bool(sr_n > 0)}


if __name__ == "__main__":
    run()
