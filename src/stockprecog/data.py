"""Ingestão de EOD ajustado da B3 via yfinance, com retry/backoff e cache parquet."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from . import config as C

_COLS = ["date", "open", "high", "low", "close", "volume"]


def _fetch_one(ticker: str, retries: int = 3, backoff: float = 3.0) -> pd.DataFrame | None:
    sym = f"{ticker}.SA"
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                sym, start=C.START, end=C.END,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df.columns = _COLS
                df["ticker"] = ticker
                return df
        except Exception as e:  # noqa: BLE001
            print(f"  {ticker} tentativa {attempt}: {e}")
        if attempt < retries:
            time.sleep(backoff * attempt)
    return None


def ingest(force: bool = False) -> pd.DataFrame:
    """Baixa o universo e cacheia em parquet. Retorna o painel long."""
    cache = C.RAW_DIR / "panel_eod.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, t in enumerate(C.UNIVERSE, 1):
        print(f"[{i:2d}/{len(C.UNIVERSE)}] {t} ...", end=" ", flush=True)
        df = _fetch_one(t)
        if df is None:
            print("FALHOU")
            failed.append(t)
        else:
            print(f"{len(df)} barras")
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise SystemExit("Nenhum dado baixado.")

    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    panel.to_parquet(cache, index=False)
    print(f"\ningeridos {panel['ticker'].nunique()}/{len(C.UNIVERSE)} tickers, "
          f"{len(panel):,} linhas. falhas: {failed or 'nenhuma'}")
    return panel


if __name__ == "__main__":
    ingest(force=True)
