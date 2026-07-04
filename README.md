# stockprecog

> An **honest-invalidation** study of cross-sectional return prediction on Brazilian (B3) equities.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: research](https://img.shields.io/badge/status-research%20spike-orange.svg)](#status)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JoaoHenriqueBarbosa/stockprecog/main/.github/badges/tests.json)](tests/)
[![Lines of code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JoaoHenriqueBarbosa/stockprecog/main/.github/badges/loc.json)](src/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706701.svg)](https://doi.org/10.5281/zenodo.20706701)

Cross-sectional prediction of **P(trade succeeds)** on Brazilian (B3) equities, built on a rigorous
[López de Prado / *Advances in Financial Machine Learning*](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
(AFML) methodology. This is deliberately a **study of honest invalidation**: the headline finding is
*modest and partly negative*, and that is the point. The contribution is the anti-self-deception
evaluation ruler — not a magic alpha. If a pipeline this careful says a signal is sub-economic, that
verdict *is* the result.

> **This is not a trading system and never claims to be.** No signal here is deployable as-is. Read
> it as a worked example of how to evaluate a strategy without fooling yourself.

---

## TL;DR — what we actually found

All numbers below are **real** (measured, logged to `data/processed/trials_log.json`, not
illustrative). The ruler is point-in-time and causality-audited (zero look-ahead leak).

1. **Price-derived signals are statistically real but economically sub-marginal.**
   Momentum / volatility / fractional-diff + microstructure (Amihud) + regime (CUSUM / SADF-lite)
   reach **gross DSR ≈ 0.41**, but **net of costs ≈ −1.1** when rebalancing every 10 days →
   sub-economic. Robust: deciles don't help, longer horizons don't help, a spread-only floor sits
   near 0 even for large caps.

2. **Cost is driven by turnover, not signal strength.** Same ruler, *opposite* verdict for high- vs
   low-turnover. Price high-turnover goes **gross +0.06 → net −1.1**. What flips the conclusion is
   turnover, not the edge.

3. **A simple linear fundamental tilt is the only net-positive config.**
   Value/quality tilt (`earnings_yield`, `book_to_price`, `profit_margin`; fundamentals lagged 90d
   for CVM filing delay + live price), long-short tercile, **quarterly** rebalance (63d),
   **pre-registered** test → **Sharpe gross +0.19, net +0.18, PSR 0.92, DSR 0.72** (deflated over the
   logged trials), annualized ≈ 0.35. Note: PSR/DSR are **below the conventional 0.95 skill
   threshold** — the config clears its pre-registered net-positive bar (a weaker criterion) and is
   *promising, not confirmed*; the annualized Sharpe is modest.

4. **SOTA ML does not beat the linear composite.** LightGBM LambdaRank overfits (Rank IC OOS
   ≈ −0.014, net ≈ 0). TabPFN v2 (small-N foundation model) **ties** the linear composite (+0.183 vs
   +0.18) on the fundamental factors and **loses** on the full feature set (turnover ~1.10, churn).
   **Verdict: the binding constraint is information + cost, not model capacity.** A zero-parameter
   linear composite of 3 factors equals the SOTA foundation model.

**Thesis.** A rigorous LdP pipeline on Brazilian equities shows that (a) price-derived signals are
real but economically sub-marginal due to transaction cost; (b) the binding constraint is
turnover × cost, not signal strength; (c) a simple linear fundamental tilt is the only viable signal;
(d) SOTA ML doesn't beat the linear composite — the bottleneck is information, not model
sophistication. A reproducible exercise in honest invalidation.

---

## Highlights

The reason to read this repo is the **methodology**, which is implemented carefully enough that the
negative result is trustworthy:

- **CPCV done correctly.** Combinatorial Purged Cross-Validation (6 groups, 2 test → 15 splits /
  5 paths). Training labels whose resolution time `t1` falls inside the test block are **purged**,
  and the **embargo is measured in trading days** (via `searchsorted` on the unique-date array), not
  calendar `timedelta` — the code documents *why* a calendar embargo would leak.
- **Deflation that accounts for multiple testing.** Deflated / Probabilistic Sharpe (Bailey–LdP)
  uses an **effective sample** `n = unique_test_dates // HORIZON` (so overlapping labels don't inflate
  `√(n−1)`), and a **trials log** deflates by *all* research attempts logged — including the failures —
  not just the winner.
- **Sample uniqueness weights** (AFML ch. 4): concurrency, average uniqueness, indicator matrix,
  sequential bootstrap, return attribution, time decay.
- **Fractional differentiation (FFD)** with ADF-based `min_ffd_d`; `d = 0.4` chosen to stay
  stationary while preserving memory. The ~55-day fracdiff window is what drove `EMBARGO_DAYS = 55`
  (a feature-memory leak caught in an adversarial audit).
- **Vectorized regime features.** CUSUM (vol-robust Chu–Stinchcombe–White) and SADF-lite explosivity
  as `O(L)` closed-form rolling OLS in NumPy — no per-window `statsmodels` — to keep the panel build
  under ~180 s.
- **Point-in-time fundamentals.** 90-day conservative lag over CVM ITR/DFP filing deadlines via
  `merge_asof`; factors combine the *lagged* fundamental with the *live* price
  (`earnings_yield = EPS_lag / close_t`).
- **One-way (not round-trip) cost convention** applied consistently on `|Δw|`, with a code comment
  explaining why a round-trip charge would double-count.

---

## Status

This is a completed **research spike**, not an actively maintained product. The engineering
conclusion (the ruler works; the signals it measures are mostly sub-economic net of cost) has been
reached and archived with a DOI. Issues and PRs are welcome, but expect maintenance to be
best-effort. Treat the code as a reference implementation of the AFML evaluation stack rather than a
supported library.

---

## Universe & data

- **111 liquid B3 stocks** (`config.UNIVERSE`; ADV screen, 2010–2026), total-return adjusted via
  brapi `adjustedClose`.
- Includes post-2018 IPOs, so survivorship bias is **partially** mitigated — a full point-in-time
  index composition is *not* yet reconstructed (noted in `config.py`), so the panel is **not**
  survivorship-free.
- Fundamentals derived from **CVM Open Data**.

> **Raw data is NOT redistributed in this repo.** See [Data & licensing](#data--licensing). You
> reproduce the dataset locally with your own brapi key + CVM parsers. The repo ships the **pipeline +
> recipe + aggregate results**, never the raw price panel.

---

## Requirements

- **Python 3.12** (pinned via `.python-version` and `requires-python = "~=3.12.0"`).
- [**uv**](https://docs.astral.sh/uv/) for dependency management and running.
- A free **brapi.dev** API token (for reproducing the data).
- **GPU note:** `torch` is pinned to the **CUDA 13.0** wheel index (`pytorch-cu130`, targeting
  Blackwell `sm_120`), and the TabPFN ranker hardcodes `device="cuda"`. CPU-only machines can run
  everything through the fundamental tilt, but the TabPFN step (5) expects a CUDA GPU.

---

## Installation

```bash
git clone https://github.com/JoaoHenriqueBarbosa/stockprecog
cd stockprecog
uv sync                       # installs declared deps (LightGBM, torch CUDA, statsmodels, ...)
cp .env.example .env          # then put your brapi token in .env
```

Get a free token at [brapi.dev](https://brapi.dev). The token lives only in `.env` (gitignored) — it
is never committed.

> **Reproducibility caveat (TabPFN step).** `tabpfn` and `huggingface_hub` are **imported** by
> `rank_tabpfn.py` but are **not** currently pinned in `uv.lock`, so `uv sync` will not install them.
> To run step 5 below, add them first:
> ```bash
> uv add tabpfn huggingface_hub
> ```
> TabPFN also downloads a model checkpoint from Hugging Face at import time.

---

## How to reproduce

Each stage is a runnable `python -m` module. Run them in order; the data must be fetched **before**
the panel can be built.

```bash
# 0. Fetch the raw B3 price panel + fundamentals with YOUR brapi token.
#    brapi_api discovers the universe and pulls historical/statistics JSON into data/raw/.
uv run python -m stockprecog.brapi_api

# 1. Build the adjusted price panel that the pipeline consumes.
#    (brapi_ingest parses the cached brapi JSON into panel_brapi_adj.parquet;
#     it does NOT fetch — with an empty cache it exits early.)
uv run python -m stockprecog.brapi_ingest

# 2. Main CPCV evaluation (LightGBM + DSR/PSR, uniqueness weights).
uv run python -m stockprecog.evaluate

# 3. Ablation across feature blocks / cost regimes.
uv run python -m stockprecog.ablation

# 4. Pre-registered fundamental tilt (the only net-positive config: net +0.18, DSR 0.72).
uv run python -m stockprecog.fundamental_tilt

# 5. TabPFN v2 ranker — SOTA foundation model that only ties the linear composite.
#    (Requires the extra deps above + a CUDA GPU.)
uv run python -m stockprecog.rank_tabpfn
```

Other entrypoints: `loop` (minimal end-to-end smoke run), `meta_evaluate` (meta-labeling CPCV),
`rank_model` (LightGBM LambdaRank). Run the test suite with `uv run pytest`.

> **Note on `brapi_ingest`.** It is a **parser**, not a downloader: it reads the brapi JSON already
> cached under `data/raw/` and raises `SystemExit` if that cache is empty. The actual network fetch
> lives in `brapi_api` (`scrape_historical` / `scrape_statistics`). Run `brapi_api` first.

---

## A note on feature counts

To keep the claims precise (and because an earlier draft over-stated this):

| Feature block | Module | Count |
|---|---|---:|
| Time-series (momentum, vol, fracdiff, RSI, dist-to-SMA) | `features` | 9 |
| Microstructure (Amihud, Roll spread, dollar-vol z, turnover accel) | `features_micro` | 4 |
| Regime (CUSUM-vol, explosivity, vol-regime) | `features_regime` | 3 |
| Fundamental (earnings yield, book-to-price, profit margin, div yield) | `features_fundamental` | 4 |
| **`ALL_FEATS`** (fed to LambdaRank **and** TabPFN, **unranked**) | — | **20** |

The rankers (`rank_model`, `rank_tabpfn`) consume the **20** raw `ALL_FEATS` columns. Only the
separate LightGBM *classifier* ablation (`ablation._featset("all")`) reaches **40** columns, by
concatenating each raw feature with its cross-sectional rank (raw ⊕ rank). So "40 features" refers to
the classifier ablation, **not** the rankers.

---

## Development

```bash
uv sync                       # install everything (incl. the dev group: pytest)
uv run pytest                 # run the test suite (fast; sub-second)
uv run pytest -q tests/test_cpcv.py::test_embargo_trading_days   # a single test
```

The suite covers the load-bearing parts of the ruler — CPCV purge/embargo, cost accounting,
triple-barrier labeling, feature construction, and uniqueness weights. The test-count and
lines-of-code badges above are refreshed by CI, so they always reflect `main`.

---

## Architecture

Pipeline dataflow (the headline results run entirely through the brapi adjusted-close panel):

```
brapi_api ──► brapi_ingest.load_for_pipeline ──► labeling (triple-barrier + t1)
                                                     │
        ┌────────────────────────────────────────────┤
        ▼               ▼               ▼             ▼
   features      features_micro   features_regime  features_fundamental
        └───────────────┴───────────────┴─────────────┘
                              │
                          weights ──► cpcv
                              │
   ┌──────────┬──────────┬────┴─────┬──────────────┬───────────────────┐
   ▼          ▼          ▼          ▼              ▼                   ▼
evaluate  ablation  meta_evaluate rank_model  rank_tabpfn      fundamental_tilt
```

```
stockprecog/
├── src/stockprecog/          # the package (22 modules)
│   ├── config.py             # universe, barriers, fracdiff/embargo params, paths, seed
│   ├── brapi_api.py          # scrape brapi REST → local JSON caches
│   ├── brapi_ingest.py       # parse cached JSON → adjusted price panel
│   ├── labeling.py           # triple-barrier labels + t1
│   ├── features*.py          # price / microstructure / regime / fundamental features
│   ├── fracdiff.py           # fractional differentiation (FFD)
│   ├── cpcv.py               # combinatorial purged CV (purge + trading-day embargo)
│   ├── weights.py            # sample uniqueness weights (AFML ch. 4)
│   ├── costs.py              # square-root market-impact, one-way convention
│   ├── evaluate.py / meta_evaluate.py / ablation.py
│   ├── fundamental_tilt.py   # pre-registered quarterly value/quality tilt
│   ├── rank_model.py / rank_tabpfn.py   # LightGBM LambdaRank, TabPFN v2 ranker
│   ├── b3_cotahist.py / b3_events.py    # alternate COTAHIST data path — notebook-only (see below)
│   └── loop.py               # minimal end-to-end run
├── tests/                    # pytest suite (cpcv, costs, labeling, features, weights)
├── notebooks/
│   └── reconstruct_adjusted_prices.py   # manual COTAHIST reconstruction (VSCode-cell style)
├── paper/                    # paper.md + compiled paper.pdf + refs.bib (pandoc build)
├── hf/                       # Hugging Face publishing kit (model/dataset cards, license notice)
├── docs/                     # B3 data idiosyncrasies, analysis roadmap
├── data/                     # gitignored: raw caches, parquet panels, trials_log.json
├── pyproject.toml / uv.lock
└── LICENSE
```

> **`b3_cotahist` / `b3_events` are an alternate, currently-unused data path.** They implement B3
> COTAHIST reconstruction and are used **only** by `notebooks/reconstruct_adjusted_prices.py` — no
> `src` module imports them, and the headline numbers do **not** flow through them. `yfinance` is a
> declared dependency but ships **no** loader module; the pipeline runs on the brapi `adjustedClose`
> panel.

---

## Data & licensing

**Code is MIT** (see [`LICENSE`](LICENSE)). The **data is not redistributed** — a deliberate legal
posture, not an oversight.

- **Prices/quotes** descend from **B3**, whose regime prohibits redistribution / republication /
  reformatting of market-data bases to third parties without prior consent. On top of that,
  **brapi.dev is silent** on redistribution (it permits commercial use but has no ToS authorizing
  republication — and contractual silence ≠ permission). The adjusted-price panel (`adjustedClose`
  EOD) carries the *highest* risk and is **never published here**.
- **Fundamentals** come from **CVM Open Data** (LAI + Open Data Policy) — redistributable *with
  attribution* ("data accessed via CVM's Open Data Portal").
- **Code / pipeline** — own work, MIT (provided the brapi token never enters git history).
- **Results / aggregate metrics** (DSR/PSR, gross/net Sharpe, Rank IC, ablation tables, turnover) —
  statistics, not redistributable data; safe to publish.

This repo adopts the standard **"code + recipe"** model for reproducible quant finance: publish the
**pipeline + instructions to fetch the data with your own key + the results** — *not* the raw prices.
Full per-layer notice: [`hf/LICENSE_NOTICE.md`](hf/LICENSE_NOTICE.md).

> **License note (to reconcile before wider publication).** The **repository code is MIT**
> (`LICENSE`, `.zenodo.json`). The Hugging Face **model card** currently declares `apache-2.0` in its
> frontmatter. These should be aligned to a single code license; this repo treats **MIT** as
> authoritative for the source.

> Not legal advice. For material legal risk, confirm in writing with brapi and cite CVM/B3 as sources.

---

## Paper & citation

**Paper (PDF):** [`paper/paper.pdf`](https://github.com/JoaoHenriqueBarbosa/stockprecog/blob/main/paper/paper.pdf)
· **DOI:** [10.5281/zenodo.20706701](https://doi.org/10.5281/zenodo.20706701) (concept DOI — always
resolves to the latest version)

```bibtex
@software{barbosa_stockprecog_2026,
  author    = {Barbosa, João Henrique},
  title     = {stockprecog: An honest-invalidation study of cross-sectional
               return prediction on the Brazilian equity market (B3)},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.20706701},
  url       = {https://doi.org/10.5281/zenodo.20706701}
}
```

---

## Contributing & conduct

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and our
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security reports: [`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE) for the code and methodology. Market data is **not** covered — see
[Data & licensing](#data--licensing).
