"""Avaliação rigorosa sob CPCV: distribuição de AUC + Sharpe de estratégia +
Deflated Sharpe. Substitui o split único do loop.py por múltiplos paths.

Estratégia de teste (pra ter retorno → Sharpe): a cada data do período de teste,
vai LONG nos ativos com proba acima da mediana cross-section e SHORT nos abaixo
(market-neutral simples), retorno = média dos fwd_ret ponderada pelo lado. Não é
a estratégia final — é um sensor honesto de se a proba ordena retorno.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from . import cpcv
from .brapi_ingest import load_for_pipeline
from .features import FEATURES, make_features
from .labeling import make_labels


def _model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=-1, verbose=-1,
    )


def _strategy_returns(test: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Retorno diário de uma long-short market-neutral por proba cross-section."""
    df = test[["date", "fwd_ret"]].copy()
    df["proba"] = proba
    rets = []
    for _, day in df.groupby("date", sort=True):
        if len(day) < 4:
            continue
        med = day["proba"].median()
        longs = day[day["proba"] > med]["fwd_ret"].mean()
        shorts = day[day["proba"] < med]["fwd_ret"].mean()
        if np.isfinite(longs) and np.isfinite(shorts):
            rets.append(longs - shorts)
    return np.asarray(rets, dtype=float)


def run(n_groups: int = 6, n_test: int = 2) -> dict:
    print(">> dados (brapi adjustedClose)")
    panel = load_for_pipeline()
    print(">> labeling triple-barrier (+t1)")
    labeled = make_labels(panel)
    print(">> features")
    feat = make_features(labeled)

    data = feat.dropna(subset=FEATURES + ["label", "t1"]).copy()
    data["label"] = data["label"].astype(int)
    data = data.sort_values("date").reset_index(drop=True)

    splits = cpcv.cpcv_splits(
        data["date"].to_numpy(), data["t1"].to_numpy(),
        n_groups=n_groups, n_test=n_test,
    )
    print(f">> CPCV: {len(splits)} splits (N={n_groups}, k={n_test}), "
          f"{cpcv.n_paths(n_groups, n_test)} paths")

    aucs, sharpes, all_rets = [], [], []
    for i, (tr, te) in enumerate(splits, 1):
        Xtr, ytr = data.iloc[tr][FEATURES], data.iloc[tr]["label"]
        Xte, yte = data.iloc[te][FEATURES], data.iloc[te]["label"]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        m = _model()
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        rets = _strategy_returns(data.iloc[te], proba)
        sr = cpcv.sharpe(rets)
        aucs.append(auc)
        if np.isfinite(sr):
            sharpes.append(sr)
            all_rets.append(rets)
        print(f"   split {i:2d}/{len(splits)}: AUC={auc:.4f}  Sharpe={sr:.3f}  "
              f"(treino={len(tr):,} teste={len(te):,})")

    aucs = np.array(aucs)
    sharpes = np.array(sharpes)
    pooled = np.concatenate(all_rets) if all_rets else np.array([])
    sk = float(pd.Series(pooled).skew()) if len(pooled) > 2 else 0.0
    ku = float(pd.Series(pooled).kurt() + 3) if len(pooled) > 2 else 3.0
    sr_mean = float(np.mean(sharpes)) if len(sharpes) else np.nan
    dsr = cpcv.deflated_sharpe(sr_mean, sharpes, len(pooled), skew=sk, kurt=ku) \
        if len(sharpes) >= 2 else np.nan

    print("\n=== AVALIAÇÃO CPCV ===")
    print(f"AUC      : média={aucs.mean():.4f}  desvio={aucs.std():.4f}  "
          f"min={aucs.min():.4f}  max={aucs.max():.4f}")
    print(f"  P(AUC>0.5) empírica: {(aucs > 0.5).mean():.2%}  "
          f"({(aucs > 0.5).sum()}/{len(aucs)} splits)")
    print(f"Sharpe   : média={sr_mean:.3f}  desvio={sharpes.std():.3f}  "
          f"(por path, não anualizado)")
    print(f"Deflated Sharpe (P(skill real)) : {dsr:.4f}")
    print(f"  skew={sk:.2f} kurt={ku:.2f} n_ret={len(pooled):,}")

    return {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
            "sharpe_mean": sr_mean, "dsr": float(dsr) if np.isfinite(dsr) else None}


if __name__ == "__main__":
    run()
