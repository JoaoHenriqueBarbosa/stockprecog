# %% [markdown]
# # Reconstrução de preços ajustados da B3 — ativo por ativo
#
# Script estilo notebook (células `# %%`). Roda no terminal (`uv run python
# notebooks/reconstruct_adjusted_prices.py`) ou abre como notebook no VSCode/Jupyter.
#
# **Por que ativo por ativo:** a API de proventos da B3 indexa por NOME DE PREGÃO
# da época (casado por prefixo), não por ticker. Empresas reorganizadas têm o
# histórico partido entre nomes (AMBV4→ABEV3). A resolução automática de nome
# falha em ~27% dos ativos, então NÃO confiamos nela — cada ativo é descoberto e
# validado manualmente aqui. Ver `docs/b3_data_idiosyncrasies.md`.
#
# **Não commitamos o dataset** — este script É a fonte da verdade reproduzível.
# Cada célula de ativo mostra: candidatos de nome → proventos → ajuste → validação.

# %%
import pandas as pd

from stockprecog import config as C
from stockprecog.b3_cotahist import build_panel
from stockprecog.b3_events import (
    adjustment_factor,
    fetch_dividends,
    fetch_dividends_multi,
    search_candidates,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# MAPA VALIDADO ativo->nomes de pregão. Preenchido célula a célula, abaixo.
# Cada entrada é confirmada por inspeção + validação antes de entrar aqui.
DIVIDEND_NAMES: dict[str, list[str]] = {}


# %% [markdown]
# ## Helpers de inspeção e validação
# `inspect(ticker)` — mostra candidatos e o que cada nome retorna, pra decidir o
# mapeamento. `validate(ticker, names)` — aplica o ajuste e compara com yfinance
# (quando disponível) ou reporta sanidade (quando não há referência).

# %%
def inspect(ticker: str) -> None:
    """Mostra candidatos de nome e o resumo de proventos de cada um."""
    print(f"=== {ticker}: candidatos de tradingName ===")
    cands = search_candidates(ticker)
    print("  busca automática:", cands or "(nada)")
    for nm in cands:
        d = fetch_dividends(nm)
        if d.empty:
            print(f"  '{nm}': 0 proventos")
        else:
            print(f"  '{nm}': {len(d)} proventos | {d.ex_date.min().date()} -> "
                  f"{d.ex_date.max().date()} | tipos={sorted(d.type_stock.unique())}")


def _load_cotahist() -> pd.DataFrame:
    p = C.RAW_DIR / "panel_cotahist.parquet"
    return pd.read_parquet(p)


def _load_yf() -> pd.DataFrame | None:
    p = C.RAW_DIR / "panel_eod.parquet"
    return pd.read_parquet(p) if p.exists() else None


def validate(ticker: str, names: list[str], type_stock: str = "ON",
             check_dates=("2024-01-02", "2020-01-02", "2016-01-04")) -> pd.DataFrame:
    """Aplica ajuste e compara com yfinance. Retorna a série ajustada do ticker."""
    co = _load_cotahist()
    g = co[co.ticker == ticker].sort_values("date").reset_index(drop=True)
    if g.empty:
        print(f"  {ticker}: SEM dados COTAHIST")
        return g
    divs = fetch_dividends_multi(names, type_stock=type_stock)
    fac = adjustment_factor(g["date"], divs)
    g = g.copy()
    g["adj_factor"] = fac.to_numpy()
    for col in ("open", "high", "low", "close"):
        g[f"{col}_adj"] = g[col].to_numpy() * fac.to_numpy()

    print(f"  {ticker}: {len(divs)} proventos {type_stock} "
          f"({divs.ex_date.min().date() if not divs.empty else '-'} -> "
          f"{divs.ex_date.max().date() if not divs.empty else '-'}), "
          f"fator min={fac.min():.3f}")

    yf = _load_yf()
    if yf is not None and ticker in set(yf.ticker.unique()):
        ref = yf[yf.ticker == ticker][["date", "close"]].rename(columns={"close": "yf"})
        m = g[["date", "close_adj"]].merge(ref, on="date")
        for d in check_dates:
            row = m[m.date == d]
            if not row.empty:
                x = row.iloc[0]
                print(f"    {d}: adj={x.close_adj:.2f} yf={x.yf:.2f} "
                      f"diff={100*(x.close_adj/x.yf-1):+.1f}%")
    else:
        print(f"    (sem referência yfinance — validação por sanidade: "
              f"fator decai monotonicamente? {(fac.diff().dropna() <= 1e-9).all()})")
    return g


# %% [markdown]
# ## ABEV3 — caso de referência (resolvido)
# AmBev reorganizou em out/2013: AMBV3/AMBV4 → ABEV3. Proventos partidos entre
# "AMBEV" (2000–2013) e "AMBEV S.A." (2013–hoje). Filtra ON. Validado: ~2% vs yf
# (resíduo = JCP/convenção). Este é o template pros demais.

# %%
inspect("ABEV3")

# %%
DIVIDEND_NAMES["ABEV3"] = ["AMBEV", "AMBEV S.A."]
_ = validate("ABEV3", DIVIDEND_NAMES["ABEV3"])
