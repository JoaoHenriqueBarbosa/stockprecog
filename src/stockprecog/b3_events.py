"""Eventos corporativos da B3 (API interna gratuita) + fator de ajuste retroativo.

COTAHIST é preço bruto. Pra reconstruir série total-return (sem gaps falsos em
ex-dates), aplicamos o fator de ajuste clássico:

  no ex-date, fator_evento = (close_prior_ex - provento_por_acao) / close_prior_ex

O fator de ajuste de uma data t é o PRODUTO de todos os fatores de eventos com
ex-date > t (acumulado retroativo). Preço ajustado = preço bruto * fator.

IMPORTANTE (idiossincrasia B3): a API de proventos indexa por NOME DE PREGÃO da
época, casado por prefixo — NÃO por ticker. Empresas reorganizadas têm o histórico
partido entre nomes (ex: AMBV4 sob "AMBEV", ABEV3 sob "AMBEV S.A."). Por isso NÃO
há resolução automática confiável: o nome (ou nomes) de cada ticker vem do mapa
validado manualmente, ativo por ativo. Ver docs/b3_data_idiosyncrasies.md.
"""
from __future__ import annotations

import base64
import json
import time

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_BASE = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall"
_URL_DIV = _BASE + "/GetListedCashDividends/{payload}"
_URL_SEARCH = _BASE + "/GetInitialCompanies/{payload}"
_UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _b64(d: dict) -> str:
    return base64.b64encode(json.dumps(d).encode()).decode()


def _to_float(s: str) -> float:
    return float(s.replace(".", "").replace(",", "."))


def search_candidates(ticker: str) -> list[str]:
    """Sugere nomes de pregão candidatos pra um ticker (ponto de partida da
    inspeção manual). Casa issuingCompany == raiz do ticker."""
    code = "".join(ch for ch in ticker if ch.isalpha())[:4]
    payload = _b64({"language": "pt-br", "pageNumber": 1, "pageSize": 50, "company": code})
    names: list[str] = []
    try:
        r = requests.get(_URL_SEARCH.format(payload=payload), headers=_UA, timeout=30, verify=False)
        for it in r.json().get("results", []):
            nm = it.get("tradingName")
            if nm and (it.get("issuingCompany", "").upper() == code or nm not in names):
                names.append(nm)
    except Exception as e:  # noqa: BLE001
        print(f"  {ticker}: erro na busca: {e}")
    return names


def fetch_dividends(name: str, max_pages: int = 10) -> pd.DataFrame:
    """Proventos de UM nome de pregão. Retorna df cru com todos os campos úteis.
    name é explícito (do mapa validado), nunca adivinhado."""
    rows: list[dict] = []
    page = 1
    while page <= max_pages:
        payload = _b64({"language": "pt-br", "pageNumber": page, "pageSize": 120, "tradingName": name})
        try:
            r = requests.get(_URL_DIV.format(payload=payload), headers=_UA, timeout=30, verify=False)
            data = r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  '{name}' erro pág {page}: {e}")
            break
        for it in data.get("results", []) or []:
            try:
                ex = pd.to_datetime(it["lastDatePriorEx"], format="%d/%m/%Y") + pd.Timedelta(days=1)
                cp = _to_float(it["closingPricePriorExDate"])
                val = _to_float(it["valueCash"])
                if cp > 0 and val > 0:
                    rows.append({"ex_date": ex, "value": val, "close_prior": cp,
                                 "type_stock": it.get("typeStock", ""),
                                 "action": it.get("corporateAction", "")})
            except (KeyError, ValueError):
                continue
        total_pages = (data.get("page") or {}).get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    return df.sort_values("ex_date").reset_index(drop=True) if not df.empty else df


def fetch_dividends_multi(names: list[str], type_stock: str = "ON") -> pd.DataFrame:
    """Emenda proventos de múltiplos nomes (eras pré/pós reorganização) e filtra
    o tipo de ação (ON pra tickers terminados em 3). Dedup por (ex_date, value)."""
    parts = [fetch_dividends(nm) for nm in names]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["ex_date", "value", "close_prior", "type_stock", "action"])
    div = pd.concat(parts, ignore_index=True)
    if type_stock:
        div = div[div["type_stock"] == type_stock]
    div = (div.drop_duplicates(subset=["ex_date", "value"])
              .sort_values("ex_date").reset_index(drop=True))
    return div


def adjustment_factor(dates: pd.Series, divs: pd.DataFrame) -> pd.Series:
    """Fator de ajuste retroativo por data (1 ticker)."""
    dvec = pd.to_datetime(dates).to_numpy()
    factor = pd.Series(1.0, index=range(len(dvec)))
    for _, ev in divs.iterrows():
        f = max(0.0, 1.0 - ev["value"] / ev["close_prior"])
        factor.iloc[(dvec < ev["ex_date"].to_datetime64()).nonzero()[0]] *= f
    return factor
