"""Cliente REST direto da brapi.dev (token) — scrape rápido com backoff e cache.

Mais rápido que o MCP (um-a-um): bate no endpoint v2 direto, em lotes, e cacheia
cada ticker em disco pra reruns serem instantâneos. Educado com rate limit:
sleep base + backoff exponencial com jitter em 429/5xx, respeita Retry-After.

Token: lido de BRAPI_TOKEN (env) ou do arquivo .env do repo (gitignored).
Cache: data/raw/brapi_api/{historical,dividends}/{TICKER}.json
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config as C

BASE = "https://brapi.dev/api"
CACHE = C.RAW_DIR / "brapi_api"
_UA = "Mozilla/5.0 (stockprecog research scraper)"
_BASE_SLEEP = 0.25  # pausa base entre requisições (educado)


def _token() -> str:
    t = os.environ.get("BRAPI_TOKEN")
    if t:
        return t
    envf = Path(__file__).resolve().parents[2] / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("BRAPI_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("BRAPI_TOKEN não encontrado (env ou .env).")


def _get(path: str, params: dict, retries: int = 6) -> dict:
    """GET com backoff exponencial + jitter. Respeita Retry-After em 429."""
    q = {**params, "token": _token()}
    url = f"{BASE}{path}?{urllib.parse.urlencode(q)}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                time.sleep(_BASE_SLEEP)
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra else _BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.5)
                wait = min(wait, 30.0)
                print(f"  [{e.code}] backoff {wait:.1f}s (tentativa {attempt+1})")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(_BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.5))
    raise SystemExit(f"GET {path} falhou após {retries} tentativas: {last}")


# ---------------------------------------------------------------- descoberta
def list_stocks() -> list[dict]:
    """Lista TODAS as ações subType=stock (pagina), com close/volume/market_cap/sector."""
    out, page = [], 1
    while True:
        d = _get("/quote/list", {
            "type": "stock", "subType": "stock",
            "sortBy": "market_cap_basic", "sortOrder": "desc",
            "limit": "200", "page": str(page),
        })
        out.extend(d.get("stocks", []))
        if not d.get("hasNextPage"):
            break
        page += 1
    return out


def select_universe(stocks: list[dict], n: int = 120, min_adv_brl: float = 5e6) -> list[str]:
    """Seleciona ~n tickers líquidos: ADV-proxy (close*volume) acima do piso,
    ordenado por market_cap. Exclui o que não tem close/volume."""
    cand = []
    for s in stocks:
        tk, close, vol = s.get("stock"), s.get("close"), s.get("volume")
        if not tk or not close or not vol:
            continue
        if tk.endswith("F"):  # mercado fracionário — duplicata do lote padrão
            continue
        adv = close * vol
        if adv >= min_adv_brl:
            cand.append((tk, s.get("market_cap") or 0, adv))
    cand.sort(key=lambda x: x[1], reverse=True)
    return [tk for tk, _, _ in cand[:n]]


# ------------------------------------------------------------------- scrape
def _cache_path(kind: str, ticker: str) -> Path:
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}.json"


def fetch_historical(symbols: list[str], rng: str = "max") -> dict[str, dict]:
    """Histórico de 1+ symbols num request. Retorna {requestedSymbol: result_normalizado}.
    O REST v2 aninha historicalDataPrice em result['data'] — achata pro topo aqui,
    mesmo schema que o brapi_ingest._parse_one_file espera."""
    d = _get("/v2/stocks/historical", {
        "symbols": ",".join(symbols), "range": rng,
        "interval": "1d", "sortOrder": "desc",
    })
    out = {}
    for r in d.get("results", []):
        key = r.get("requestedSymbol", r.get("symbol"))
        data = r.get("data") or {}
        out[key] = {
            "requestedSymbol": key,
            "symbol": r.get("symbol"),
            "changed": r.get("changed"),
            "historicalDataPrice": data.get("historicalDataPrice")
            or r.get("historicalDataPrice"),
        }
    return out


def scrape_historical(tickers: list[str], batch: int = 4, rng: str = "max",
                      force: bool = False) -> dict[str, str]:
    """Puxa histórico em lotes, cacheia por ticker. Re-tenta individual o que cair.
    Retorna {ticker: status}."""
    status = {}
    todo = [t for t in tickers if force or not _cache_path("historical", t).exists()]
    skipped = [t for t in tickers if t not in todo]
    for t in skipped:
        status[t] = "cache"
    for i in range(0, len(todo), batch):
        lote = todo[i:i + batch]
        print(f"[hist {i+1}-{i+len(lote)}/{len(todo)}] {','.join(lote)}")
        res = fetch_historical(lote, rng)
        for t in lote:
            r = res.get(t)
            if r and r.get("historicalDataPrice"):
                _cache_path("historical", t).write_text(json.dumps(r))
                status[t] = f"ok({len(r['historicalDataPrice'])})"
            else:
                # re-tenta individual (drops de lote acontecem)
                r1 = fetch_historical([t], rng).get(t)
                if r1 and r1.get("historicalDataPrice"):
                    _cache_path("historical", t).write_text(json.dumps(r1))
                    status[t] = f"ok-solo({len(r1['historicalDataPrice'])})"
                else:
                    status[t] = "VAZIO"
    return status


def fetch_statistics_history(symbol: str) -> dict | None:
    """Histórico trimestral de estatísticas-chave (value/quality factors)."""
    d = _get("/v2/stocks/statistics",
             {"symbols": symbol, "mode": "history", "period": "quarterly"})
    res = d.get("results") or []
    if not res:
        return None
    r = res[0]
    data = r.get("data")
    return {"symbol": r.get("requestedSymbol", symbol), "data": data} if data else None


def scrape_statistics(tickers: list[str], force: bool = False) -> dict[str, str]:
    """Puxa estatísticas-chave trimestrais por ticker (1 req/ticker), cacheia."""
    status = {}
    todo = [t for t in tickers if force or not _cache_path("statistics", t).exists()]
    for i, t in enumerate(todo, 1):
        print(f"[stats {i}/{len(todo)}] {t}")
        r = fetch_statistics_history(t)
        if r and r.get("data"):
            _cache_path("statistics", t).write_text(json.dumps(r))
            status[t] = f"ok({len(r['data'])})"
        else:
            status[t] = "vazio"
    return status


def scrape_dividends(tickers: list[str], batch: int = 8, force: bool = False) -> dict[str, str]:
    """Puxa dividendos/JCP por ticker, cacheia."""
    status = {}
    todo = [t for t in tickers if force or not _cache_path("dividends", t).exists()]
    for i in range(0, len(todo), batch):
        lote = todo[i:i + batch]
        print(f"[div {i+1}-{i+len(lote)}/{len(todo)}] {','.join(lote)}")
        d = _get("/v2/stocks/dividends", {"symbols": ",".join(lote)})
        by = {r.get("requestedSymbol", r.get("symbol")): r for r in d.get("results", [])}
        for t in lote:
            _cache_path("dividends", t).write_text(json.dumps(by.get(t, {})))
            status[t] = "ok" if t in by else "vazio"
    return status


if __name__ == "__main__":
    stocks = list_stocks()
    print(f"universo bruto: {len(stocks)} ações")
    uni = select_universe(stocks)
    print(f"selecionados: {len(uni)} — {uni}")
