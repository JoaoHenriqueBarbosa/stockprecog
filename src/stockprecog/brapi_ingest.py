"""Ingestão via brapi MCP — parseia os JSONs logados em data/raw/brapi_log/ e
monta o painel de preços AJUSTADOS (adjustedClose embute dividendos+splits+bonif).

As chamadas MCP em si são feitas pelo agente (não há cliente HTTP aqui, pra não
duplicar auth/limites). Cada resposta é salva em brapi_log/ e este módulo a consome.
Fonte de produção validada contra reconstrução COTAHIST (VALE3: 0.02% em 2024).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from . import config as C

LOG_DIR = C.RAW_DIR / "brapi_log"


def _parse_one_file(path: Path) -> list[pd.DataFrame]:
    """Extrai DataFrames (1 por symbol) de um JSON de resposta do MCP."""
    raw = json.loads(path.read_text())
    # formato: lista de blocos {type,text}; o bloco JSON é o último 'text' parseável
    payload = None
    for blk in raw if isinstance(raw, list) else [raw]:
        t = blk.get("text", "") if isinstance(blk, dict) else ""
        if t.strip().startswith("{"):
            payload = json.loads(t)
    if not payload:
        return []
    frames = []
    for res in payload.get("results", []):
        sym = res.get("symbol")
        hp = res.get("historicalDataPrice") or []
        if not hp:
            continue
        df = pd.DataFrame(hp)
        df["date"] = df["date"].map(lambda s: pd.Timestamp(dt.date.fromtimestamp(s)))
        df["ticker"] = res.get("requestedSymbol", sym)
        df = df[["date", "ticker", "open", "high", "low", "close",
                 "adjustedClose", "volume"]]
        frames.append(df)
    return frames


def build_panel() -> pd.DataFrame:
    """Consolida os historical_*.json do log num painel ajustado (adjustedClose).
    Exclui C.UNRELIABLE (ajuste não-confiável) — só entram tickers com brapi."""
    frames: list[pd.DataFrame] = []
    for f in sorted(LOG_DIR.glob("historical_*.json")):
        frames.extend(_parse_one_file(f))
    if not frames:
        raise SystemExit("Nenhum JSON de histórico em brapi_log/.")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[~panel["ticker"].isin(C.UNRELIABLE)]
    panel = (
        panel.drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    out = C.RAW_DIR / "panel_brapi_adj.parquet"
    panel.to_parquet(out, index=False)
    miss = [t for t in C.USABLE_UNIVERSE if t not in set(panel["ticker"].unique())]
    print(f"painel final: {len(panel):,} linhas, {panel['ticker'].nunique()}/"
          f"{len(C.USABLE_UNIVERSE)} tickers usáveis, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"  excluídos (não-confiáveis): {C.UNRELIABLE}")
    if miss:
        print(f"  AINDA FALTANDO: {miss}")
    print(f"salvo: {out}")
    return panel


def load_for_pipeline() -> pd.DataFrame:
    """Painel no schema do pipeline (date,ticker,open,high,low,close,volume) com
    OHLC TOTALMENTE ajustado: close=adjustedClose; OHL escalados pelo fator
    adjustedClose/close (ajuste de dividendo; o close brapi já vem split/bonif-aj).
    Filtra USABLE_UNIVERSE e [START, END]. Constrói o painel se faltar."""
    path = C.RAW_DIR / "panel_brapi_adj.parquet"
    panel = pd.read_parquet(path) if path.exists() else build_panel()

    panel = panel[panel["ticker"].isin(C.USABLE_UNIVERSE)].copy()
    fac = panel["adjustedClose"] / panel["close"]
    out = pd.DataFrame({
        "date": pd.to_datetime(panel["date"]),
        "ticker": panel["ticker"],
        "open": panel["open"] * fac,
        "high": panel["high"] * fac,
        "low": panel["low"] * fac,
        "close": panel["adjustedClose"],
        "volume": panel["volume"],
    })
    out = out[(out["date"] >= C.START) & (out["date"] <= C.END)]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    build_panel()
