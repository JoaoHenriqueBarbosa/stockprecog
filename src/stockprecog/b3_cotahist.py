"""Scraper + parser do COTAHIST (série histórica oficial da B3).

Fonte autoritativa e gratuita de EOD de TODAS as ações listadas em cada ano —
inclui as que foram deslistadas, então NÃO tem survivorship bias (ao contrário
do yfinance com composição atual).

CAVEAT IMPORTANTE: COTAHIST é PREÇO BRUTO (não ajustado por proventos/splits).
Ajuste por eventos corporativos é um passo separado, ainda a fazer.

Layout posicional (registro tipo 01), posições 1-indexed conforme manual B3:
  TIPREG  01-02 | DATA 03-10 (YYYYMMDD) | CODBDI 11-12 | CODNEG 13-24 (ticker)
  TPMERC  25-27 | PREABE 57-69 | PREMAX 70-82 | PREMIN 83-95 | PREULT 109-121
  QUATOT  153-170 (qtd negociada) | VOLTOT 171-188 (volume R$)
Preços vêm *100 (2 casas implícitas).
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config as C

BASE = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
UA = {"User-Agent": "Mozilla/5.0 (stockprecog research)"}

# filtros: CODBDI 02 = lote padrão; TPMERC 010 = mercado à vista
_CODBDI_LOTE_PADRAO = "02"
_TPMERC_VISTA = "010"


def download_year(year: int, force: bool = False) -> Path:
    """Baixa o ZIP COTAHIST do ano em RAW_DIR (com retry). Reusa cache se já existe."""
    dest = C.RAW_DIR / f"COTAHIST_A{year}.ZIP"
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    url = BASE.format(year=year)
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=UA, timeout=180)
            r.raise_for_status()
            if len(r.content) > 1000:
                dest.write_bytes(r.content)
                return dest
        except Exception as e:  # noqa: BLE001
            print(f"  {year} tentativa {attempt}: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"falha ao baixar COTAHIST {year}")


def parse_year(zip_path: Path, tickers: set[str] | None = None) -> pd.DataFrame:
    """Parseia o layout posicional do COTAHIST (à vista, lote padrão) -> OHLC+volume.

    `tickers` opcional filtra o universo. Preços já divididos por 100 (preço BRUTO).
    """
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.upper().endswith(".TXT")][0]
        raw = z.read(name)

    rows = []
    for line in io.BytesIO(raw):
        if line[0:2] != b"01":  # só registros de cotação
            continue
        codbdi = line[10:12].decode()
        tpmerc = line[24:27].decode()
        if codbdi != _CODBDI_LOTE_PADRAO or tpmerc != _TPMERC_VISTA:
            continue
        ticker = line[12:24].decode().strip()
        if tickers is not None and ticker not in tickers:
            continue
        date = line[2:10].decode()
        o = int(line[56:69]) / 100.0
        h = int(line[69:82]) / 100.0
        lo = int(line[82:95]) / 100.0
        c = int(line[108:121]) / 100.0
        qty = int(line[152:170])
        rows.append((date, ticker, o, h, lo, c, qty))

    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def build_panel(years: range, tickers: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """Baixa+parseia os anos, concatena e salva o painel COTAHIST em parquet."""
    tset = set(tickers) if tickers else None
    frames = []
    for y in years:
        print(f"[{y}] download...", end=" ", flush=True)
        zp = download_year(y, force=force)
        print(f"parse...", end=" ", flush=True)
        df = parse_year(zp, tset)
        print(f"{len(df):,} linhas, {df['ticker'].nunique()} tickers")
        frames.append(df)
    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    out = C.RAW_DIR / "panel_cotahist.parquet"
    panel.to_parquet(out, index=False)
    print(f"\npainel: {len(panel):,} linhas, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()}→{panel['date'].max().date()}\nsalvo: {out}")
    return panel


if __name__ == "__main__":
    # validação: só 2024, universo do projeto
    build_panel(range(2024, 2025), tickers=C.UNIVERSE)
