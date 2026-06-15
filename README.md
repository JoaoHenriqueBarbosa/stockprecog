# stockprecog

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Status: research](https://img.shields.io/badge/status-research-orange.svg)

Cross-sectional prediction of **P(trade succeeds)** on Brazilian (B3) equities, built
on a rigorous [López de Prado / *Advances in Financial Machine Learning*](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
methodology. This is deliberately a **study of honest invalidation**: the headline finding
is *modest and partly negative*, and that is the point. The contribution is the
anti-self-deception ruler — not a magic alpha. If a pipeline this careful says a signal is
sub-economic, that verdict is the result.

---

## TL;DR — what we actually found

All numbers below are **real** (measured, not illustrative). The ruler is point-in-time,
causality-audited (zero leak), with 26 pytest tests.

1. **Price-derived signals are statistically real but economically sub-marginal.**
   Momentum / vol / fractional-diff + microstructure (Amihud) + regime (CUSUM / SADF-lite)
   reach **gross DSR ≈ 0.41**, but **NET of costs = −1.1** when rebalancing every 10 days →
   sub-economic. Robust: deciles don't help, longer horizons don't help, a spread-only floor
   sits near 0 even for large caps.

2. **Cost is driven by turnover, not signal strength.** Same ruler, *opposite* verdict for
   high- vs low-turnover. Price high-turnover goes **gross +0.06 → net −1.1**. What flips the
   conclusion is turnover, not the edge.

3. **A simple linear fundamental tilt is the only net-positive config.**
   Value/quality tilt (`earnings_yield`, `book_to_price`, `profit_margin`; fundamentals lagged
   90d for CVM filing delay + live price), long-short tercile, **quarterly** rebalance (63d),
   **pre-registered** test → **Sharpe gross +0.19, net +0.18, PSR 0.92, DSR 0.72** (deflated over
   18 trials), annualized ≈ 0.35. Note: PSR/DSR are **below the conventional 0.95** skill
   threshold — it clears its pre-registered net-positive bar (a weaker criterion) and is
   *promising, not confirmed*; the annualized Sharpe is modest.

4. **SOTA ML does not beat the linear composite.** LightGBM LambdaRank (40 feats) overfits
   (Rank IC OOS −0.014, net ≈ 0, effective sample ≈ 58 quarters). TabPFN v2 (small-N foundation
   model) **ties** the linear composite (+0.183 vs +0.18) on the fundamental factors and **loses**
   on the full feature set (turnover 1.10, churn). **Verdict: the binding constraint is
   information + cost, not model capacity.** A zero-parameter linear composite of 3 factors
   equals the SOTA foundation model.

**Thesis.** A rigorous LdP pipeline on Brazilian equities shows that (a) price-derived signals are
real but economically sub-marginal due to transaction cost; (b) the binding constraint is
turnover × cost, not signal strength; (c) a simple linear fundamental tilt is the only viable
signal; (d) SOTA ML doesn't beat the linear composite — the bottleneck is information, not model
sophistication. A reproducible exercise in honest invalidation.

---

## The methodological ruler (the actual contribution)

This is what makes the negative result trustworthy:

- **CPCV** — Combinatorial Purged Cross-Validation (6 groups, 2 test → 15 splits / 5 paths) with
  **purge + embargo** measured in trading days (embargo covers both label resolution `HORIZON=10`
  and fracdiff memory → 55 days).
- **Triple-barrier labels + `t1`** (profit-take / stop-loss / vertical barrier).
- **Deflated / Probabilistic Sharpe** (Bailey–LdP) with effective `n` (dates / HORIZON) and a
  **trials log** that deflates by *all* ~21 attempts — not just the winner.
- **Sample uniqueness weights** (AFML ch.4 — uniqueness *preserves* edge, return-attribution
  *destroys* it).
- **Fractional differentiation** (FFD `d=0.4`, threshold `1e-3`, window ~55) — stationary while
  preserving memory.
- **Costs**: square-root market-impact law (Toth et al.), **one-way** convention (not round-trip).
- Everything **point-in-time**; causality audited; 26 tests.

---

## Universe & data

- **111 liquid B3 stocks** (ADV ≥ R$3M, 2010–2026), total-return adjusted via brapi `adjustedClose`.
- Includes post-2018 IPOs → survivorship bias *partially* mitigated.
- Fundamentals derived from **CVM Open Data**.

> **Data is NOT redistributed here.** See [Data & licensing](#data--licensing). You reproduce the
> dataset locally with your own brapi key + CVM parsers. The repo ships the **pipeline + recipe +
> results**, never the raw price panel.

---

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JoaoHenriqueBarbosa/stockprecog
cd stockprecog
uv sync                       # installs deps (LightGBM, torch CUDA, statsmodels, ...)
cp .env.example .env          # then put your brapi token in .env
```

Get a free token at [brapi.dev](https://brapi.dev). The token lives only in `.env` (gitignored)
— it is never committed.

---

## How to reproduce

Each stage is a runnable module. Run them in order; the first builds the local panel.

```bash
# 1. Ingest the price panel from brapi (uses YOUR token)
uv run python -m stockprecog.brapi_ingest

# 2. Main CPCV evaluation (LightGBM + DSR/PSR, uniqueness weights)
uv run python -m stockprecog.evaluate

# 3. Ablation across feature blocks / cost regimes
uv run python -m stockprecog.ablation

# 4. Pre-registered fundamental tilt (only net-positive config: net +0.18, DSR 0.72)
uv run python -m stockprecog.fundamental_tilt

# 5. TabPFN v2 ranker — SOTA foundation model that only ties the linear composite
uv run python -m stockprecog.rank_tabpfn
```

Other entrypoints: `loop` (minimal end-to-end smoke run), `meta_evaluate` (meta-labeling CPCV),
`rank_model` (LightGBM LambdaRank). Run the test suite with `uv run pytest`.

---

## Module map (`src/stockprecog/`)

| Module | Role |
|---|---|
| `config` | Universe, barriers, fracdiff/embargo params, paths |
| `brapi_api`, `brapi_ingest` | Scrape brapi REST → local price panel |
| `b3_cotahist`, `b3_events` | B3 COTAHIST parsing, corporate events |
| `labeling` | Triple-barrier labels + `t1` |
| `features`, `features_micro`, `features_regime`, `features_fundamental` | Price, microstructure (Amihud), regime (CUSUM/SADF), CVM fundamentals |
| `fracdiff` | Fractional differentiation (FFD) |
| `cpcv` | Combinatorial Purged CV (purge + embargo) |
| `weights` | Sample uniqueness weights (AFML ch.4) |
| `costs` | Square-root market-impact, one-way convention |
| `evaluate`, `meta_evaluate`, `ablation` | CPCV evaluation, meta-labeling, ablation |
| `fundamental_tilt` | Pre-registered quarterly value/quality tilt |
| `rank_model`, `rank_tabpfn` | LightGBM LambdaRank, TabPFN v2 ranker |
| `data` | Orphan yfinance loader (legacy) |
| `loop` | Minimal end-to-end run |

---

## Data & licensing

**Code is MIT.** The **data is not redistributed** — and that is a deliberate legal posture, not an
oversight.

- **Prices/quotes** descend from **B3**, whose regime *expressly* prohibits redistribution /
  republication / reformatting of market-data bases to third parties without prior consent (with
  audit rights; policy hardened in 2026). On top of that, **brapi.dev is silent** on
  redistribution: it permits commercial use but has no ToS authorizing republication — and
  contractual silence ≠ permission. The adjusted-price panel (`adjustedClose` EOD) carries the
  *highest* risk (B3 restriction + brapi silence) and is **never published here**.
- **Fundamentals** come from **CVM Open Data** (LAI + Open Data Policy) — redistributable *with
  attribution* ("data accessed via CVM's Open Data Portal"). Prefer pointing readers to the CVM source.
- **Code / pipeline** — own work, no restriction (provided the brapi token never enters git history).
- **Results / aggregate metrics** (DSR/PSR, gross/net Sharpe, Rank IC, ablation tables, turnover) —
  statistics, not redistributable data; safe to publish.

This repo adopts the standard **"code + recipe"** model for reproducible quant finance: publish the
**pipeline + instructions to fetch the data with your own key + the results** — *not* the raw prices.
`.gitignore` excludes the parquet panels; the token lives in `.env`.

> Not legal advice. For material legal risk, confirm in writing with brapi and cite CVM/B3 as sources.

---

## Paper & citation

Paper: *(link — to be added)*.

```bibtex
@misc{barbosa_stockprecog,
  author = {Barbosa, João Henrique},
  title  = {stockprecog: honest invalidation of cross-sectional equity signals on B3 under a López de Prado ruler},
  year   = {2026},
  url    = {https://github.com/JoaoHenriqueBarbosa/stockprecog}
}
```
