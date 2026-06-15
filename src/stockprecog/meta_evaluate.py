"""Meta-labeling sob CPCV (López de Prado, AFML cap. 3).

Primário: classifica P(PT bate) — define o LADO da aposta (long se proba>0.5).
Meta: aprende P(o primário acerta) e DIMENSIONA a aposta (size = confiança meta).
O valor não está em melhorar AUC, e sim em FILTRAR — só apostar quando o meta tem
confiança, elevando precisão/Sharpe mesmo com AUC primário fraco.

Anti-vazamento: dentro de cada fold de treino do CPCV, split interno temporal —
primário treina na 1ª metade, gera meta-labels (acertou/errou) na 2ª, meta treina
nessa 2ª. No teste: primário dá lado, meta dá tamanho.
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
        n_estimators=300, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=-1, verbose=-1,
    )


def _ls_returns(test: pd.DataFrame, side: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Retorno diário long-short com tamanho variável (size em [0,1])."""
    df = test[["date", "fwd_ret"]].copy()
    df["w"] = np.where(side > 0, 1.0, -1.0) * size
    rets = []
    for _, day in df.groupby("date", sort=True):
        wsum = day["w"].abs().sum()
        if wsum < 1e-9 or len(day) < 4:
            continue
        rets.append(float((day["w"] * day["fwd_ret"]).sum() / wsum))
    return np.asarray(rets, dtype=float)


def run(n_groups: int = 6, n_test: int = 2) -> dict:
    """Compara Sharpe flat vs meta-labeling (cap. 3) sob CPCV — testa se o meta agrega valor."""
    print(">> dados + labeling + features")
    panel = load_for_pipeline()
    feat = make_features(make_labels(panel))
    data = feat.dropna(subset=FEATURES + ["label", "t1"]).copy()
    data["label"] = data["label"].astype(int)
    data = data.sort_values("date").reset_index(drop=True)

    splits = cpcv.cpcv_splits(data["date"].to_numpy(), data["t1"].to_numpy(),
                              n_groups=n_groups, n_test=n_test)
    print(f">> CPCV {len(splits)} splits, {cpcv.n_paths(n_groups, n_test)} paths\n")

    aucs, sr_flat, sr_meta, rets_flat, rets_meta = [], [], [], [], []
    for i, (tr, te) in enumerate(splits, 1):
        tr_df = data.iloc[tr]
        te_df = data.iloc[te]
        # split interno temporal no treino (1ª metade primário, 2ª meta)
        cut = tr_df["date"].quantile(0.5)
        p_df = tr_df[(tr_df["date"] <= cut) & (tr_df["t1"] <= cut)]
        m_df = tr_df[tr_df["date"] > cut]
        if p_df["label"].nunique() < 2 or len(m_df) < 200:
            continue

        prim = _model().fit(p_df[FEATURES], p_df["label"])

        # meta-labels: o primário acertou no conjunto meta?
        m_proba = prim.predict_proba(m_df[FEATURES])[:, 1]
        m_side = (m_proba >= 0.5).astype(int)
        m_correct = (m_side == m_df["label"].to_numpy()).astype(int)
        if len(np.unique(m_correct)) < 2:
            continue
        meta = _model().fit(m_df[FEATURES], m_correct)

        # teste
        te_proba = prim.predict_proba(te_df[FEATURES])[:, 1]
        te_side = (te_proba >= 0.5).astype(int)
        te_conf = meta.predict_proba(te_df[FEATURES])[:, 1]
        if te_df["label"].nunique() < 2:
            continue

        aucs.append(roc_auc_score(te_df["label"], te_proba))
        # flat: tamanho 1 sempre; meta: tamanho = confiança (filtra incerteza)
        rf = _ls_returns(te_df, te_side, np.ones(len(te_df)))
        rm = _ls_returns(te_df, te_side, te_conf)
        sf, sm = cpcv.sharpe(rf), cpcv.sharpe(rm)
        if np.isfinite(sf):
            sr_flat.append(sf); rets_flat.append(rf)
        if np.isfinite(sm):
            sr_meta.append(sm); rets_meta.append(rm)
        print(f"   split {i:2d}: AUC={aucs[-1]:.4f}  Sharpe flat={sf:+.3f}  meta={sm:+.3f}")

    def report(name, srs, rets_list):
        srs = np.array(srs)
        pooled = np.concatenate(rets_list) if rets_list else np.array([])
        psr = cpcv.probabilistic_sharpe(cpcv.sharpe(pooled), 0.0, len(pooled),
                                        skew=float(pd.Series(pooled).skew()),
                                        kurt=float(pd.Series(pooled).kurt() + 3)) \
            if len(pooled) > 2 else np.nan
        tstat = srs.mean() / (srs.std(ddof=1) / np.sqrt(len(srs))) if len(srs) > 1 else np.nan
        print(f"{name:6}: Sharpe médio={srs.mean():+.3f}±{srs.std():.3f}  "
              f"t={tstat:+.2f}  PSR(>0)={psr:.3f}  (n_ret={len(pooled):,})")
        return srs.mean(), psr

    aucs = np.array(aucs)
    print(f"\n=== META-LABELING CPCV ===")
    print(f"AUC primário: média={aucs.mean():.4f} ±{aucs.std():.4f}  "
          f"P(>0.5)={(aucs > 0.5).mean():.0%}")
    sf_m, psr_f = report("flat", sr_flat, rets_flat)
    sm_m, psr_m = report("meta", sr_meta, rets_meta)
    print(f"\nmeta melhora Sharpe? {sm_m > sf_m}  (Δ={sm_m - sf_m:+.3f})")
    return {"auc": float(aucs.mean()), "sharpe_flat": float(sf_m),
            "sharpe_meta": float(sm_m), "psr_meta": float(psr_m) if np.isfinite(psr_m) else None}


if __name__ == "__main__":
    run()
