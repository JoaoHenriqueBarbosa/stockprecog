---
license: other
license_name: mixed-redistribution-restricted
license_link: LICENSE_NOTICE.md
language:
  - pt
pretty_name: stockprecog — B3 cross-sectional reproducible feature panel (code + recipe)
tags:
  - finance
  - quant
  - equities
  - brazil
  - b3
  - lopez-de-prado
  - triple-barrier
  - cross-sectional
  - reproducible-research
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 100K<n<1M
configs:
  - config_name: derived_features
    description: >-
      Non-invertible derived features only (fracdiff, normalized vol/returns,
      cross-section z-scores, Amihud, regime flags, triple-barrier labels + t1).
      Contains NO price levels (no OHLC, no adjustedClose). NOT shipped by
      default — see "What is and is not in this repo".
---

# stockprecog — B3 cross-sectional reproducible panel (code + recipe)

> **TL;DR.** This is a **reproducible-research artifact**, not a raw market-data
> dataset. It ships the **pipeline + reproduction recipe + aggregated results**.
> It deliberately **does NOT ship raw or adjusted B3 prices** (`panel_eod.parquet`,
> `adjustedClose`). You reconstruct the panel locally with **your own brapi key**
> and **CVM Open Data parsers**. See [Licensing & redistribution](#licensing--redistribution).

## What this is

A rigorous López de Prado (AFML) pipeline for predicting **P(a trade works out)**
cross-section on liquid **B3 (Brazilian)** equities. The headline scientific result
is an **honest invalidation study**: a methodologically strict pipeline shows that
price-derived signals on B3 are *statistically real but economically sub-marginal
after transaction costs*, and that the only economically viable configuration is a
**simple linear fundamental tilt** — which **SOTA ML does not beat**.

This card documents the **feature panel and how to regenerate it**. The "model"
side (the linear fundamental tilt champion, plus the LightGBM/TabPFN configs that
fail to beat it) is documented in the companion **[Model Card](README_model.md)**.

## Universe & coverage

| Field | Value |
|---|---|
| Market | B3 (Brasil, Bovespa) |
| Universe | 111 liquid tickers (ADV ≥ R$3M, ≥300 bars) |
| Period | 2010-01-01 → 2026-06-15 |
| Adjustment | total-return (brapi `adjustedClose` EOD), reconstructed locally |
| IPOs | post-2018 IPOs included → survivorship *partially* mitigated |
| Survivorship | **partial** — current liquid composition, not full point-in-time membership (documented limitation) |

The 111-ticker list and the `ADV ≥ R$3M` selection rule are public metadata and are
shipped here (`src/stockprecog/config.py::UNIVERSE`). The **price observations behind
them are not.**

## Data layers & their license regime (per layer)

The raw data behind this project splits into legally distinct layers. The
redistribution decision is made **per layer**:

| Layer | Source | Redistributable here? |
|---|---|---|
| (a) Raw / adjusted prices (`adjustedClose` EOD, OHLC) | B3 via brapi | **NO** — B3 expressly forbids redistribution of market-data bases to third parties; brapi is silent (silence ≠ permission). Highest-risk layer. |
| (b) Derived **non-invertible** features (fracdiff, z-scores, Amihud, regime, labels) | computed | **Grey zone** — low/moderate risk *only if* they do not reconstruct OHLC. Not shipped by default. |
| (c) Fundamentals (`earnings_yield`, `book_to_price`, `profit_margin`) | CVM Open Data | **YES with attribution** — CVM is Open Data (LAI + Open Data Policy). Prefer pointing to the CVM source. |
| (d) Code / pipeline | own work | **YES** — unrestricted. |
| (e) Aggregated results / metrics | computed | **YES** — statistics, not redistributable data. |

> ⚠️ The adjusted-price panel is the layer of **maximum risk**: it stacks the B3
> restriction on top of brapi's contractual silence. It is **excluded** from this repo.

## How to reproduce the panel (the recipe)

You bring the data access; the code does the rest.

```bash
# 1. clone the pipeline (this repo's `src/` is shipped)
git clone <this-repo>
cd stockprecog
uv sync                              # Python 3.12, LightGBM, torch (cu130), etc.

# 2. provide YOUR OWN brapi token (free key at brapi.dev) — never commit it
echo "BRAPI_TOKEN=your_key_here" > .env   # .env is gitignored

# 3. reconstruct the price panel locally (brapi REST scrape, adjustedClose EOD)
uv run python -m stockprecog.brapi_ingest        # writes data/raw/panel_eod.parquet (gitignored)

# 4. fundamentals: CVM Open Data (attribution required) parsed by features_fundamental
#    point-in-time merge_asof, 90d conservative CVM lag

# 5. build features + labels + run the honest evaluation ruler
uv run python -m stockprecog.loop                # end-to-end baseline
uv run python -m stockprecog.fundamental_tilt    # the one viable config
```

The reproduction recipe distributes the **means of obtaining** the data (a scraper
keyed to *your* credentials + open CVM parsers), not the data itself.

## Feature families (computed, point-in-time, leakage-audited)

| Family | Module | Examples |
|---|---|---|
| Price / TA | `features.py` | momentum, EWMA vol, RSI, cross-section ranks |
| Fractional diff | `fracdiff.py` | FFD `d=0.4`, thresh `1e-3`, window ~55 |
| Microstructure | `features_micro.py` | Amihud illiquidity |
| Regime | `features_regime.py` | CUSUM / SADF-lite structural breaks |
| Fundamental (value/quality) | `features_fundamental.py` | earnings_yield, book_to_price, profit_margin, div_yield |
| Labels | `labeling.py` | triple-barrier (PT/SL = 2σ, H=10d) + `t1` |

All features are **point-in-time**: fundamentals are lagged a conservative **90 days**
(CVM ITR 45d / DFP 90d deadline) and combine *lagged fundamental ÷ live price*.
Causality was adversarially audited (zero leak); the fracdiff convolution memory is
covered by an embargo of **55 trading days** in CPCV.

## Evaluation ruler (why the numbers are honest)

This panel is meant to be evaluated under the project's anti-self-deception ruler:

- **CPCV**: 6 groups, 2 test → 15 splits / 5 paths, with **purge + embargo in trading days**.
- **Deflated / Probabilistic Sharpe** (Bailey–LdP) with **effective N** (unique dates / HORIZON × avg uniqueness) and a **trials log** that deflates by *all* ~21 research attempts.
- **Sample uniqueness weights** (AFML ch.4) — uniqueness *preserves* edge; return-attribution would *destroy* it.
- **Costs**: square-root market-impact law (Tóth et al.), **one-way** convention (not round-trip).

## Headline results (real numbers — do not inflate)

| Config | Gross Sharpe | Net Sharpe | PSR | DSR | Verdict |
|---|---|---|---|---|---|
| Price signal (momentum/vol/fracdiff + Amihud + regime), rebal 10d | DSR_gross ~0.41 | **−1.1** | — | — | sub-economic (turnover-killed) |
| **Fundamental tilt (value/quality, tercile L/S, quarterly 63d)** | **+0.19** | **+0.18** | **0.92** | **0.72** (deflated 18 trials) | ✅ **only viable config** (~0.35 annualized) |
| LightGBM LambdaRank (40 feat, quarterly) | — | ~0 | — | — | overfits (Rank IC OOS −0.014) |
| TabPFN v2 (fundamentals) | — | **+0.183** | — | — | **ties** the linear composite |
| TabPFN v2 (all features) | — | worse | — | — | loses (turnover 1.10 churn) |

**Thesis:** the binding constraint is **turnover × cost**, not signal strength; a
zero-parameter 3-factor linear composite **equals** the SOTA foundation model. The
bottleneck is **information + cost, not model capacity**.

## Licensing & redistribution

This repo follows the **"code + recipe"** model standard in reproducible quant finance:
publish the pipeline, reproduction instructions, and results — **not** the raw price panel.

- Raw/adjusted B3 prices → **not redistributed** (B3 express restriction + brapi silence).
- Fundamentals → CVM **Open Data**, redistributable **with attribution**
  (*"data accessed via the CVM Open Data Portal"*); prefer pointing to the CVM source.
- Code/pipeline → own work, unrestricted.
- Results/metrics → free to publish.

See **[LICENSE_NOTICE.md](LICENSE_NOTICE.md)** for the full per-layer analysis and the
**[MANIFEST.md](MANIFEST.md)** for exactly what ships and what is withheld.

> *Not legal advice. For material legal risk, confirm in writing with brapi and cite
> CVM/B3 as sources.*

## Citation

```bibtex
@misc{stockprecog2026,
  title  = {stockprecog: an honest-invalidation study of cross-sectional
            tradeability prediction on B3 equities under a López de Prado ruler},
  author = {Barbosa, João Henrique},
  year   = {2026},
  note   = {Pipeline + reproduction recipe; raw market data not redistributed.}
}
```
