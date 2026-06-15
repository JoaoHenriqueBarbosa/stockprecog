"""Ablation de eixos de feature (passo 18): price vs ortogonais (micro/regime).

Mede CADA conjunto de feature sob a régua honesta: AUC, DSR bruto (deflacionado
pelo trials-log) e Sharpe LÍQUIDO de custos (convenção one-way corrigida, spread
large-cap). Responde: o eixo ortogonal carrega peso próprio ou decora o preço?

DISCIPLINA DE TRIALS: cada config testada é registrada no trials_log — o DSR
deflaciona por TODAS as tentativas, inclusive as que falharem. n_trials honesto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import cpcv
from . import evaluate as E
from .brapi_ingest import load_for_pipeline
from .features import TS_FEATURES, make_features
from .features_fundamental import FUNDAMENTAL_FEATURES, make_fundamental_features
from .features_micro import MICRO_FEATURES, make_micro_features
from .features_regime import REGIME_FEATURES, make_regime_features
from .labeling import make_labels
from .weights import panel_sample_weights

_MATRIX = C.PROC_DIR / "feature_matrix.parquet"


def build_feature_matrix(force: bool = False) -> pd.DataFrame:
    """Computa TODAS as features (price + micro + regime) + ranks cross-section +
    sample weights uniqueness, uma vez, e cacheia."""
    if _MATRIX.exists() and not force:
        return pd.read_parquet(_MATRIX)
    panel = load_for_pipeline()
    labeled = make_labels(panel)
    feat = make_features(labeled)                      # price TS + ranks já prontos
    micro = make_micro_features(panel)[["date", "ticker"] + MICRO_FEATURES]
    regime = make_regime_features(panel)[["date", "ticker"] + REGIME_FEATURES]
    fund = make_fundamental_features(panel)[["date", "ticker"] + FUNDAMENTAL_FEATURES]
    df = (feat.merge(micro, on=["date", "ticker"])
          .merge(regime, on=["date", "ticker"])
          .merge(fund, on=["date", "ticker"]))
    # ranks cross-section (dentro de cada data) das novas features
    for f in MICRO_FEATURES + REGIME_FEATURES + FUNDAMENTAL_FEATURES:
        df[f"{f}_rank"] = df.groupby("date")[f].rank(pct=True)
    # sample weights uniqueness
    s = panel_sample_weights(labeled, mode="uniqueness")
    wm = labeled[["ticker", "date"]].copy(); wm["w"] = s.to_numpy()
    df = df.merge(wm, on=["ticker", "date"], how="left")
    df["w"] = df["w"].fillna(0.0)
    df.to_parquet(_MATRIX, index=False)
    return df


def _featset(name: str) -> list[str]:
    price = TS_FEATURES + [f"{f}_rank" for f in TS_FEATURES]
    micro = MICRO_FEATURES + [f"{f}_rank" for f in MICRO_FEATURES]
    regime = REGIME_FEATURES + [f"{f}_rank" for f in REGIME_FEATURES]
    fund = FUNDAMENTAL_FEATURES + [f"{f}_rank" for f in FUNDAMENTAL_FEATURES]
    amihud = ["amihud", "amihud_rank"]
    return {
        "price": price,
        "amihud_only": amihud,
        "price+amihud": price + amihud,
        "price+micro": price + micro,
        "price+regime": price + regime,
        "price+micro+regime": price + micro + regime,
        "micro+regime": micro + regime,
        "fundamental_only": fund,
        "price+fundamental": price + fund,
        "all": price + micro + regime + fund,
    }[name]


def run_ablation(configs: list[str], n_groups: int = 6, n_test: int = 2) -> pd.DataFrame:
    df = build_feature_matrix()
    adv_map = df.assign(dv=df["close"] * df["volume"]).groupby("ticker")["dv"].median().to_dict()
    cpcv.seed_trials({"L0": 0.0, "L1": -0.022, "L2": 0.028, "L3": -0.088})
    rows = []
    for cfg in configs:
        feats = _featset(cfg)
        d = df.dropna(subset=feats + ["label", "t1"]).copy()
        d["label"] = d["label"].astype(int)
        d = d.sort_values("date").reset_index(drop=True)
        splits = cpcv.cpcv_splits(d["date"].to_numpy(), d["t1"].to_numpy(), n_groups, n_test)
        aucs, srs, net5, tdates, rets = [], [], [], set(), []
        for tr, te in splits:
            ytr, yte = d.iloc[tr]["label"], d.iloc[te]["label"]
            if ytr.nunique() < 2 or yte.nunique() < 2:
                continue
            m = E._model()
            m.fit(d.iloc[tr][feats], ytr, sample_weight=d.iloc[tr]["w"].to_numpy())
            p = m.predict_proba(d.iloc[te][feats])[:, 1]
            aucs.append(roc_auc_score(yte, p))
            sr = cpcv.sharpe(E._strategy_returns(d.iloc[te], p))
            if np.isfinite(sr):
                srs.append(sr); rets.append(E._strategy_returns(d.iloc[te], p))
                tdates.update(d.iloc[te]["date"].to_numpy())
            # net @ large-cap 5bps, capital pequeno (piso realista)
            _, n = E._backtest_costed(d.iloc[te], p, adv_map, capital=2e6, half_spread_bps=5.0)
            sn = cpcv.sharpe(n)
            if np.isfinite(sn):
                net5.append(sn)
        sr_mean = float(np.mean(srs)) if srs else np.nan
        cpcv.append_trial(f"abl_{cfg}", sr_mean)  # DISCIPLINA: registra o trial
        pooled = np.concatenate(rets) if rets else np.array([])
        n_eff = cpcv.effective_n(sorted(tdates)) if tdates else 0
        sr_tr = cpcv.trial_sharpes()
        dsr = cpcv.deflated_sharpe(
            sr_mean, sr_tr, n_eff,
            skew=float(pd.Series(pooled).skew()) if len(pooled) > 2 else 0.0,
            kurt=float(pd.Series(pooled).kurt() + 3) if len(pooled) > 2 else 3.0,
            n_trials=len(sr_tr)) if (np.isfinite(sr_mean) and len(sr_tr) >= 2 and n_eff >= 2) else np.nan
        rows.append({"config": cfg, "n_feat": len(feats), "auc": np.mean(aucs),
                     "sharpe_gross": sr_mean, "dsr_gross": dsr,
                     "net_sharpe_5bps_2M": float(np.mean(net5)) if net5 else np.nan,
                     "n_trials": len(sr_tr)})
        print(f"  {cfg:20} AUC={np.mean(aucs):.4f} DSR={dsr:.3f} net(5bps,2M)={np.mean(net5):+.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run_ablation(["price", "amihud_only", "price+amihud", "price+micro",
                        "price+regime", "price+micro+regime", "micro+regime"])
    print("\n=== ABLATION ===")
    print(res.to_string(index=False))
