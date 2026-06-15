"""Fase ML SOTA: TabPFN v2 (foundation tabular small-N) no ranking trimestral.

A pergunta afiada (mata o fantasma "será que faltava ML poderoso"): o tilt fundamental
que venceu é um composite LINEAR equal-weight de 3 fatores. TabPFN sobre os MESMOS
fatores fundamentais lentos testa se há estrutura NÃO-LINEAR que o linear deixou na
mesa — com turnover controlado por construção (mesmas features lentas). Também rodamos
sobre TODAS as features como diagnóstico do churn que features price-derived causam.

DISCIPLINA (advertência registrada): turnover PRIMEIRO (prediz o líquido), depois net
CPCV, depois DSR deflacionado por TODOS os trials. TabPFN não tem hiperparâmetro pra
tunar (in-context) -> 1 trial por config, sem grid search silencioso. Bater +0.18 só
conta OOS, net, com turnover medido e DSR deflacionado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor

from . import cpcv
from .brapi_ingest import load_for_pipeline
from .features import TS_FEATURES
from .features_fundamental import FUNDAMENTAL_FEATURES
from .features_micro import MICRO_FEATURES
from .features_regime import REGIME_FEATURES
from .rank_model import _net_sharpe, _quarterly_panel

_CKPT = hf_hub_download(repo_id="Prior-Labs/TabPFN-v2-reg",
                        filename="tabpfn-v2-regressor.ckpt")

FEATSETS = {
    "fundamental": FUNDAMENTAL_FEATURES,                                   # lento (sharp test)
    "all": TS_FEATURES + MICRO_FEATURES + REGIME_FEATURES + FUNDAMENTAL_FEATURES,
}


def _eval(q, feats, adv_map, n_groups=6, n_test=2):
    d0 = q.dropna(subset=feats + ["fwd"]).copy()
    splits = cpcv.cpcv_splits(d0["date"].to_numpy(), d0["date"].to_numpy(), n_groups, n_test)
    ics, net_all, turns_all = [], [], []
    for tr, te in splits:
        dtr, dte = d0.iloc[tr], d0.iloc[te]
        if dtr["date"].nunique() < 2 or dte["date"].nunique() < 1:
            continue
        m = TabPFNRegressor(model_path=_CKPT, device="cuda", ignore_pretraining_limits=True)
        m.fit(dtr[feats].to_numpy(), dtr["fwd"].to_numpy())
        sc = m.predict(dte[feats].to_numpy())
        for _, day in dte.assign(sc=sc).groupby("date"):
            if len(day) >= 12:
                ic, _ = spearmanr(day["sc"], day["fwd"])
                if np.isfinite(ic):
                    ics.append(ic)
        rets, _, turns = _net_sharpe(dte, sc, adv_map, {})
        net_all.extend(rets); turns_all.extend(turns)
    return np.mean(ics), np.array(net_all), float(np.mean(turns_all)) if turns_all else np.nan


def run() -> dict:
    """Ranking trimestral com TabPFN v2 (foundation small-N) vs baseline tilt linear (+0.18)."""
    q = _quarterly_panel()
    panel = load_for_pipeline()
    adv_map = (panel.assign(dv=panel["close"] * panel["volume"])
               .groupby("ticker")["dv"].median().to_dict())
    print(f">> painel trimestral: {len(q):,} linhas, {q['date'].nunique()} rebal | TabPFN v2 GPU")
    print(f">> baseline tilt fundamental (linear equal-weight) = net +0.18, turnover ~0.4\n")

    out = {}
    for name, feats in FEATSETS.items():
        ic, net, turn = _eval(q, feats, adv_map)
        sr = cpcv.sharpe(net)
        cpcv.append_trial(f"tabpfn_{name}_quarterly", sr)
        sr_tr = cpcv.trial_sharpes()
        dsr = cpcv.deflated_sharpe(sr, sr_tr, len(net), n_trials=len(sr_tr)) \
            if len(sr_tr) >= 2 and len(net) >= 2 else np.nan
        psr = cpcv.probabilistic_sharpe(sr, 0.0, len(net),
                                        skew=float(pd.Series(net).skew()),
                                        kurt=float(pd.Series(net).kurt()+3)) if len(net) > 2 else np.nan
        print(f"--- TabPFN [{name}] ({len(feats)} features) ---")
        print(f"  TURNOVER/rebal = {turn:.3f}   (price 0.74 | fund-tilt baixo) <- diagnóstico primeiro")
        print(f"  Rank IC = {ic:+.4f}   Sharpe LÍQUIDO = {sr:+.3f}   (bater +0.18?)")
        print(f"  PSR(>0)={psr:.3f}  DSR(deflac {len(sr_tr)} trials)={dsr:.3f}  rebal={len(net)}\n")
        out[name] = {"turnover": turn, "rank_ic": ic, "sharpe_net": sr, "dsr": dsr}
    return out


if __name__ == "__main__":
    run()
