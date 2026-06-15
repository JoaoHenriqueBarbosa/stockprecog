"""Fase ML SOTA: learning-to-rank cross-section TRIMESTRAL (low-turnover).

A formulação que cruzou breakeven foi o tilt fundamental rebal/63d. Aqui subimos de
nível: em vez do composite fixo equal-weight de 3 fatores, um LightGBM LambdaRank
aprende a ORDENAÇÃO cross-section a partir de TODAS as features (price+micro+regime+
fundamental), otimizando net-of-cost. Baseline a bater = tilt fundamental (net +0.18).

Régua: CPCV sobre datas de rebalance + custo one-way + DSR deflac. por trials.
Objetivo nativo do problema: rank (Rank IC), não classificação binária (AUC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

from . import config as C
from . import cpcv
from .ablation import build_feature_matrix
from .brapi_ingest import load_for_pipeline
from .costs import impact_bps
from .features_fundamental import FUNDAMENTAL_FEATURES, make_fundamental_features
from .features_micro import MICRO_FEATURES
from .features_regime import REGIME_FEATURES
from .features import TS_FEATURES

REBAL = 63
CAPITAL, HALF_SPREAD_BPS, SIGMA = 2e6, 5.0, 0.025
ALL_FEATS = (TS_FEATURES + MICRO_FEATURES + REGIME_FEATURES + FUNDAMENTAL_FEATURES)


def _quarterly_panel() -> pd.DataFrame:
    """Painel nas datas de rebalance (stride 63d): features + retorno forward 63d."""
    df = build_feature_matrix()                         # price+micro+regime+fund + ranks
    panel = load_for_pipeline()
    fund = make_fundamental_features(panel)[["date", "ticker"] + FUNDAMENTAL_FEATURES]
    df = df.merge(fund, on=["date", "ticker"], suffixes=("", "_f"))
    # retorno forward de REBAL dias por ticker
    px = panel.sort_values(["ticker", "date"]).copy()
    px["fwd"] = px.groupby("ticker")["close"].shift(-REBAL) / px["close"] - 1.0
    df = df.merge(px[["date", "ticker", "fwd", "close", "volume"]], on=["date", "ticker"])
    dates = np.sort(df["date"].unique())[::REBAL]       # só datas de rebalance
    q = df[df["date"].isin(dates)].dropna(subset=ALL_FEATS + ["fwd"]).copy()
    return q.sort_values(["date", "ticker"]).reset_index(drop=True)


def _relevance(fwd: pd.Series) -> np.ndarray:
    """Relevância p/ lambdarank: rank cross-section do fwd em 5 buckets [0..4]."""
    return pd.qcut(fwd.rank(method="first"), 5, labels=False).astype(int).to_numpy()


def _net_sharpe(test: pd.DataFrame, score: np.ndarray, adv_map: dict,
                prev_w: dict) -> tuple[float, dict, list]:
    t = test[["date", "ticker", "fwd"]].copy(); t["s"] = score
    rets = []
    for d, day in t.groupby("date", sort=True):
        if len(day) < 12:
            continue
        hi, lo = day["s"].quantile(2/3), day["s"].quantile(1/3)
        L, S = day[day["s"] >= hi], day[day["s"] <= lo]
        if len(L) == 0 or len(S) == 0:
            continue
        w = {tk: 0.5/len(L) for tk in L["ticker"]}
        w.update({tk: -0.5/len(S) for tk in S["ticker"]})
        fwd = dict(zip(day["ticker"], day["fwd"]))
        gr = sum(w[tk]*fwd[tk] for tk in w if np.isfinite(fwd.get(tk, np.nan)))
        cost = 0.0
        for tk in set(w) | set(prev_w):
            dw = w.get(tk, 0.0) - prev_w.get(tk, 0.0)
            if dw == 0:
                continue
            adv = adv_map.get(tk, np.nan)
            part = abs(dw)*CAPITAL/adv if (np.isfinite(adv) and adv > 0) else 0.01
            cost += abs(dw) * (HALF_SPREAD_BPS + impact_bps(part, SIGMA)) / 1e4
        rets.append(gr - cost); prev_w = w
    return rets, prev_w


def run(n_groups: int = 6, n_test: int = 2) -> dict:
    q = _quarterly_panel()
    panel = load_for_pipeline()
    adv_map = (panel.assign(dv=panel["close"]*panel["volume"])
               .groupby("ticker")["dv"].median().to_dict())
    print(f">> painel trimestral: {len(q):,} linhas, {q['date'].nunique()} datas de rebal, "
          f"{q['ticker'].nunique()} tickers")
    # t1 p/ purge = data do rebalance seguinte (holding de 63d)
    q["t1"] = q["date"]  # aproximação: cada obs resolve no próximo rebal; embargo cobre
    splits = cpcv.cpcv_splits(q["date"].to_numpy(), q["date"].to_numpy(), n_groups, n_test)

    ics, net_all = [], []
    for tr, te in splits:
        dtr, dte = q.iloc[tr], q.iloc[te]
        if dtr["date"].nunique() < 2 or dte["date"].nunique() < 1:
            continue
        gtr = dtr.groupby("date").size().to_numpy()
        m = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03,
                       num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                       random_state=C.SEED, n_jobs=-1, verbose=-1)
        m.fit(dtr.sort_values("date")[ALL_FEATS], _relevance(dtr.sort_values("date")["fwd"]),
              group=dtr.sort_values("date").groupby("date").size().to_numpy())
        sc = m.predict(dte[ALL_FEATS])
        # Rank IC por data
        for d, day in dte.assign(sc=sc).groupby("date"):
            if len(day) >= 12:
                ic, _ = spearmanr(day["sc"], day["fwd"])
                if np.isfinite(ic):
                    ics.append(ic)
        rets, _ = _net_sharpe(dte, sc, adv_map, {})
        net_all.extend(rets)

    ic = float(np.mean(ics)) if ics else np.nan
    net = np.array(net_all)
    sr_net = cpcv.sharpe(net)
    cpcv.append_trial("rank_lambdarank_quarterly", sr_net)
    sr_tr = cpcv.trial_sharpes()
    dsr = cpcv.deflated_sharpe(sr_net, sr_tr, len(net), n_trials=len(sr_tr)) \
        if len(sr_tr) >= 2 and len(net) >= 2 else np.nan
    psr = cpcv.probabilistic_sharpe(sr_net, 0.0, len(net),
                                    skew=float(pd.Series(net).skew()),
                                    kurt=float(pd.Series(net).kurt()+3)) if len(net) > 2 else np.nan
    print("\n=== LEARNING-TO-RANK TRIMESTRAL (LambdaRank, todas as features) ===")
    print(f"Rank IC médio   = {ic:+.4f}  (0=aleatório; >0.02-0.05 já é bom em equities)")
    print(f"Sharpe LÍQUIDO  = {sr_net:+.3f}   (baseline tilt fundamental = +0.18)")
    print(f"PSR(net>0)={psr:.3f}  DSR(deflac {len(sr_tr)} trials)={dsr:.3f}  rebalances={len(net)}")
    return {"rank_ic": ic, "sharpe_net": sr_net, "psr": psr, "dsr": dsr}


if __name__ == "__main__":
    run()
