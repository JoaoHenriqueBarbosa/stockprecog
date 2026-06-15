"""Loop end-to-end mínimo: dados -> labels -> features -> treino -> inferência -> métrica.

Split temporal PURGADO: treino e teste separados por um embargo >= horizonte do
triple-barrier, pra que nenhum label de treino enxergue preços do período de teste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import config as C
from .data import ingest
from .features import FEATURES, make_features
from .labeling import make_labels


def temporal_purged_split(df: pd.DataFrame):
    """Divide por data: treino no início, teste no fim, com embargo no meio."""
    dates = np.sort(df["date"].unique())
    cut = int(len(dates) * (1 - C.TEST_FRACTION))
    train_end = dates[cut]
    # embargo: remove do treino as últimas EMBARGO_DAYS barras antes do teste
    embargo_start = dates[max(0, cut - C.EMBARGO_DAYS)]

    train = df[df["date"] < embargo_start]
    test = df[df["date"] >= train_end]
    return train, test, pd.Timestamp(train_end)


def run() -> dict:
    print(">> ingestão")
    panel = ingest()
    print(">> labeling triple-barrier")
    labeled = make_labels(panel)
    print(">> features")
    feat = make_features(labeled)

    data = feat.dropna(subset=FEATURES + ["label"]).copy()
    data["label"] = data["label"].astype(int)

    train, test, cut = temporal_purged_split(data)
    print(f">> split: treino={len(train):,} (até <{cut.date()} c/ embargo) | "
          f"teste={len(test):,} (>= {cut.date()})")

    Xtr, ytr = train[FEATURES], train["label"]
    Xte, yte = test[FEATURES], test["label"]

    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    base_rate = yte.mean()
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, pred)

    print("\n=== RESULTADO (loop end-to-end) ===")
    print(f"base rate teste      : {base_rate:.4f}")
    print(f"accuracy             : {acc:.4f}  (acima do base rate? {acc > max(base_rate, 1-base_rate)})")
    print(f"ROC AUC              : {auc:.4f}  (0.5 = aleatório)")

    # top features por ganho
    imp = (pd.Series(model.feature_importances_, index=FEATURES)
           .sort_values(ascending=False).head(8))
    print("\ntop features (ganho):")
    print(imp.to_string())

    return {"auc": auc, "acc": acc, "base_rate": base_rate}


if __name__ == "__main__":
    run()
