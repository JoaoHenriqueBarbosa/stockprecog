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
from .costs import round_trip_cost_bps
from .features import FEATURES, make_features
from .labeling import make_labels
from .weights import panel_sample_weights


def _sample_weights(labeled: pd.DataFrame, mode: str = "return_attr") -> pd.DataFrame:
    """Pesos de amostra (cap.4), cacheados por modo ('return_attr' ou 'uniqueness')."""
    cache = C.PROC_DIR / f"sample_weights_{mode}.parquet"
    if cache.exists():
        w = pd.read_parquet(cache)
        if len(w) == len(labeled):
            return w
    print(f">> sample weights mode={mode} (cap.4 — pode levar 1-2min)")
    s = panel_sample_weights(labeled, mode=mode)
    w = labeled[["ticker", "date"]].copy()
    w["w"] = s.to_numpy()
    w.to_parquet(cache, index=False)
    return w


def _model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=-1, verbose=-1,
    )


def _strategy_returns(test: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Sensor de sinal: long-short market-neutral diário por proba (sobreposto)."""
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


def _backtest_costed(test: pd.DataFrame, proba: np.ndarray, adv_map: dict,
                     capital: float = 2e7, half_spread_bps: float = 10.0,
                     sigma: float = 0.025, frac: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Backtest econômico NÃO-SOBREPOSTO: rebalanceia a cada HORIZON dias (holding
    casa com o horizonte do label), long-short equal-weight por proba. frac=fração
    de cada perna (0.5=split na mediana; 0.1=decis, menos nomes/turnover). Aplica
    custo lei-sqrt por nome (participation = |Δw|*capital/ADV). Retorna (bruto, líq)."""
    df = test[["date", "ticker", "fwd_ret"]].copy()
    df["proba"] = proba
    days = np.sort(df["date"].unique())[::C.HORIZON]  # stride = HORIZON (não-sobreposto)
    prev_w: dict = {}
    gross, net = [], []
    for d in days:
        day = df[df["date"] == d]
        if len(day) < 4:
            continue
        hi = day["proba"].quantile(1 - frac); lo_q = day["proba"].quantile(frac)
        lo = day[day["proba"] >= hi]; sh = day[day["proba"] <= lo_q]
        if len(lo) == 0 or len(sh) == 0:
            continue
        w = {t: 0.5 / len(lo) for t in lo["ticker"]}
        w.update({t: -0.5 / len(sh) for t in sh["ticker"]})
        gr = float(sum(w[t] * r for t, r in zip(day["ticker"], day["fwd_ret"])
                       if t in w and np.isfinite(r)))
        # custo de rotação vs book anterior
        cost = 0.0
        for t in set(w) | set(prev_w):
            dw = w.get(t, 0.0) - prev_w.get(t, 0.0)
            if dw == 0:
                continue
            adv = adv_map.get(t, np.nan)
            part = abs(dw) * capital / adv if (np.isfinite(adv) and adv > 0) else 0.01
            cost += abs(dw) * round_trip_cost_bps(part, sigma, half_spread_bps) / 1e4
        gross.append(gr); net.append(gr - cost)
        prev_w = w
    return np.asarray(gross), np.asarray(net)


def run(n_groups: int = 6, n_test: int = 2, weight_mode: str | None = "uniqueness") -> dict:
    print(">> dados (brapi adjustedClose)")
    panel = load_for_pipeline()
    adv_map = (panel.assign(dv=panel["close"] * panel["volume"])
               .groupby("ticker")["dv"].median().to_dict())  # ADV em R$ por nome
    print(">> labeling triple-barrier (+t1)")
    labeled = make_labels(panel)
    print(">> features")
    feat = make_features(labeled)

    data = feat.dropna(subset=FEATURES + ["label", "t1"]).copy()
    data["label"] = data["label"].astype(int)
    if weight_mode:
        wdf = _sample_weights(labeled, mode=weight_mode)
        data = data.merge(wdf, on=["ticker", "date"], how="left")
        data["w"] = data["w"].fillna(0.0)
    else:
        data["w"] = 1.0
    data = data.sort_values("date").reset_index(drop=True)

    splits = cpcv.cpcv_splits(
        data["date"].to_numpy(), data["t1"].to_numpy(),
        n_groups=n_groups, n_test=n_test,
    )
    print(f">> CPCV: {len(splits)} splits (N={n_groups}, k={n_test}), "
          f"{cpcv.n_paths(n_groups, n_test)} paths")

    aucs, sharpes, all_rets = [], [], []
    sr_gross_no, sr_net_no = [], []  # backtest não-sobreposto bruto/líquido
    test_dates: set = set()
    for i, (tr, te) in enumerate(splits, 1):
        Xtr, ytr = data.iloc[tr][FEATURES], data.iloc[tr]["label"]
        Xte, yte = data.iloc[te][FEATURES], data.iloc[te]["label"]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        m = _model()
        m.fit(Xtr, ytr, sample_weight=data.iloc[tr]["w"].to_numpy())
        proba = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        rets = _strategy_returns(data.iloc[te], proba)
        sr = cpcv.sharpe(rets)
        gross_no, net_no = _backtest_costed(data.iloc[te], proba, adv_map)
        aucs.append(auc)
        if np.isfinite(sr):
            sharpes.append(sr)
            all_rets.append(rets)
            test_dates.update(data.iloc[te]["date"].to_numpy())
        sg, sn = cpcv.sharpe(gross_no), cpcv.sharpe(net_no)
        if np.isfinite(sg):
            sr_gross_no.append(sg)
        if np.isfinite(sn):
            sr_net_no.append(sn)
        print(f"   split {i:2d}/{len(splits)}: AUC={auc:.4f}  Sharpe={sr:.3f}  "
              f"| não-sobrep bruto={sg:+.3f} líq={sn:+.3f}  "
              f"(treino={len(tr):,} teste={len(te):,})")

    aucs = np.array(aucs)
    sharpes = np.array(sharpes)
    pooled = np.concatenate(all_rets) if all_rets else np.array([])
    sk = float(pd.Series(pooled).skew()) if len(pooled) > 2 else 0.0
    ku = float(pd.Series(pooled).kurt() + 3) if len(pooled) > 2 else 3.0
    sr_mean = float(np.mean(sharpes)) if len(sharpes) else np.nan

    # n EFETIVO: datas de teste únicas / HORIZON. NÃO len(pooled) — pooled duplica
    # cada data em ~k/N*C(N,k) splits e sobrepõe retornos por HORIZON dias, o que
    # inflaria sqrt(n-1) maciçamente e mentiria sobre a confiança do PSR/DSR.
    n_eff = cpcv.effective_n(sorted(test_dates)) if test_dates else 0

    # DSR via LOG DE TRIALS (multiple testing real), não variância dos paths.
    # Semeia os levers conhecidos L0..L3 e registra o Sharpe DESTA run; cada config
    # é uma tentativa de research. A variância ENTRE configs é o insumo do DSR.
    cpcv.seed_trials({"L0_baseline": 0.0, "L1": -0.022, "L2": 0.028, "L3": -0.088})
    if np.isfinite(sr_mean):
        cfg = "L5_universo111+sample_weights" if "w" in data and data["w"].nunique() > 1 \
            else "L4_universo111"
        cpcv.append_trial(cfg, sr_mean)
    sr_trials = cpcv.trial_sharpes()
    dsr = cpcv.deflated_sharpe(sr_mean, sr_trials, n_eff, skew=sk, kurt=ku,
                               n_trials=len(sr_trials)) \
        if np.isfinite(sr_mean) and len(sr_trials) >= 2 and n_eff >= 2 else np.nan

    print("\n=== AVALIAÇÃO CPCV ===")
    print(f"AUC      : média={aucs.mean():.4f}  desvio={aucs.std():.4f}  "
          f"min={aucs.min():.4f}  max={aucs.max():.4f}")
    print(f"  P(AUC>0.5) empírica: {(aucs > 0.5).mean():.2%}  "
          f"({(aucs > 0.5).sum()}/{len(aucs)} splits)")
    print(f"Sharpe   : média={sr_mean:.3f}  (por path, não anualizado)")
    print(f"  dispersão entre paths (estimador, não tentativas): "
          f"±{sharpes.std():.3f}  ({len(sharpes)} paths)")
    print(f"Deflated Sharpe (P(skill real)) : {dsr:.4f}")
    print(f"  n_eff={n_eff} (datas únicas={len(test_dates):,}/HORIZON={C.HORIZON}; "
          f"n_pooled bruto={len(pooled):,})  "
          f"trials no log={len(sr_trials)}  skew={sk:.2f} kurt={ku:.2f}")
    g_no = float(np.mean(sr_gross_no)) if sr_gross_no else np.nan
    n_no = float(np.mean(sr_net_no)) if sr_net_no else np.nan
    print(f"\nBacktest NÃO-SOBREPOSTO (rebal/{C.HORIZON}d, custos lei-sqrt @cap R$20M/perna):")
    print(f"  Sharpe bruto={g_no:+.3f}  ->  LÍQUIDO de custos={n_no:+.3f}  "
          f"(drag={g_no-n_no:+.3f})")

    return {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
            "sharpe_mean": sr_mean, "n_eff": int(n_eff),
            "n_trials": int(len(sr_trials)),
            "dsr": float(dsr) if np.isfinite(dsr) else None}


if __name__ == "__main__":
    run()
