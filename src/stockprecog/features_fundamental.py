"""Fatores FUNDAMENTAIS (value/quality) — o eixo genuinamente NÃO-PREÇO.

Momentum/microestrutura/regime são todos derivados de preço/volume. Fundamentos
(lucro, patrimônio, margem) vêm da contabilidade — eixo ortogonal de verdade.

DISCIPLINA POINT-IN-TIME (o pecado capital aqui é lookahead de fundamento):
- A brapi dá `endDate` = FIM do trimestre de referência, NÃO a data de divulgação.
- Resultados trimestrais (ITR) têm prazo CVM de 45 dias; anuais (DFP) 90. Aplicamos
  lag CONSERVADOR de 90 dias corridos: o fundamento do trimestre que termina em
  endDate só fica "conhecível" em endDate+90. merge_asof garante que cada (ticker,
  date) vê só o último fundamento já divulgado.
- Fatores combinam FUNDAMENTO DEFASADO com PREÇO VIVO (earnings_yield = EPS_lag /
  close_t), não os ratios congelados da API (que usam preço velho do endDate).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C

STATS_CACHE = C.RAW_DIR / "brapi_api" / "statistics"
LAG_DAYS = 90  # buffer conservador sobre o prazo CVM (ITR 45d / DFP 90d)

FUNDAMENTAL_FEATURES = ["earnings_yield", "book_to_price", "profit_margin", "div_yield"]


def parse_statistics() -> pd.DataFrame:
    """Lê os JSONs de statistics e monta long (ticker, endDate, fundamentos)."""
    rows = []
    for f in sorted(STATS_CACHE.glob("*.json")):
        r = json.loads(f.read_text())
        tk = r.get("symbol")
        for q in r.get("data") or []:
            ed = q.get("endDate")
            if not ed:
                continue
            rows.append({
                "ticker": tk,
                "end_date": pd.Timestamp(ed[:10]),
                "book_value": q.get("bookValue"),
                "trailing_eps": q.get("trailingEps") or q.get("earningsPerShare"),
                "profit_margin": q.get("profitMargins"),
                "div_yield_raw": q.get("dividendYield") or q.get("yield"),
            })
    df = pd.DataFrame(rows)
    # data em que o fundamento fica conhecível (PIT)
    df["available"] = df["end_date"] + pd.Timedelta(days=LAG_DAYS)
    return df.sort_values(["ticker", "available"]).reset_index(drop=True)


def make_fundamental_features(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Anexa fatores fundamentais PIT ao painel de preços (merge_asof por ticker).
    Retorna o painel + FUNDAMENTAL_FEATURES. Linhas sem fundamento divulgado -> NaN."""
    fund = parse_statistics()
    out = []
    fund = fund.copy()
    fund["available"] = fund["available"].astype("datetime64[ns]")
    for tk, g in price_panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["date"] = g["date"].astype("datetime64[ns]")
        fg = fund[fund["ticker"] == tk].sort_values("available")
        if fg.empty:
            for col in FUNDAMENTAL_FEATURES:
                g[col] = np.nan
            out.append(g)
            continue
        merged = pd.merge_asof(
            g, fg[["available", "book_value", "trailing_eps",
                   "profit_margin", "div_yield_raw"]],
            left_on="date", right_on="available", direction="backward",
        )
        close = merged["close"]
        # fatores: fundamento DEFASADO / PREÇO VIVO
        merged["earnings_yield"] = merged["trailing_eps"] / close
        merged["book_to_price"] = merged["book_value"] / close
        merged["profit_margin"] = merged["profit_margin"]   # ratio PIT-estável
        merged["div_yield"] = merged["div_yield_raw"]
        out.append(merged.drop(columns=["available", "book_value", "trailing_eps",
                                        "div_yield_raw"]))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    f = parse_statistics()
    print(f"fundamentos: {f['ticker'].nunique()} tickers, {len(f)} trimestres, "
          f"{f['end_date'].min().date()} -> {f['end_date'].max().date()}")
