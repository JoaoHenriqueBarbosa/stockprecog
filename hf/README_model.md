---
license: apache-2.0
language:
  - pt
pretty_name: stockprecog — fundamental linear tilt (champion) + ML SOTA baselines
tags:
  - finance
  - quant
  - equities
  - brazil
  - b3
  - lopez-de-prado
  - factor-investing
  - lightgbm
  - tabpfn
  - learning-to-rank
library_name: lightgbm
pipeline_tag: tabular-classification
---

# stockprecog — model card

> **The "model" that wins is a zero-parameter 3-factor linear composite.**
> This card documents that champion, the ML baselines that *fail to beat it*, and the
> evaluation ruler that makes the comparison honest. Code is Apache-2.0; the price data
> it trains on is **not** redistributed (see the [Dataset Card](README_dataset.md)).

## Model summary

| | |
|---|---|
| Task | Cross-sectional **P(trade succeeds)** / tradeability ranking on B3 equities |
| Champion | **Linear fundamental tilt** — equal-weight mean of cross-section ranks of `earnings_yield`, `book_to_price`, `profit_margin` |
| Trainable params | **0** (fixed equal-weight composite, no tuning) |
| Construction | long top tercile, short bottom tercile, equal-weight, market-neutral |
| Rebalance | **quarterly (63 trading days)** — turnover is the whole game |
| Costs | square-root market impact (Tóth et al.) + 5bps half-spread, **one-way** |
| Baselines | LightGBM LambdaRank (40 feat), TabPFN v2 (foundation tabular small-N) |

## Why a linear composite is the "model"

The central finding is an **honest invalidation**: across the AFML levers the
*economic* edge (DSR) sat at **0** for every price-derived configuration. The only
configuration that crosses breakeven net of costs is a **pre-registered**, fixed,
no-tuning **fundamental tilt**. SOTA ML does **not** improve on it:

- **LightGBM LambdaRank** (40 features, quarterly groups) **overfits**: Rank IC OOS
  **−0.014**, net ~0, effective sample ~58 quarters. ~28-item queries give listwise
  ranking almost no signal.
- **TabPFN v2** (foundation model designed for small-N) **ties** the linear composite
  on the fundamental factors (**+0.183 vs +0.18**) and **loses** on the full feature set
  (turnover **1.10**, churn-driven cost blowup).

**Verdict:** the bottleneck is **information + cost, not model capacity.** A
zero-parameter linear composite equals the SOTA foundation model.

## Real results (do not invent, do not inflate)

### Champion — fundamental tilt (pre-registered, one-shot)

| Metric | Value |
|---|---|
| Sharpe gross | **+0.19** |
| Sharpe net (one-way costs) | **+0.18** |
| PSR(net>0) | **0.92** |
| DSR (deflated, 18 trials) | **0.72** |
| Annualized (×√(252/63)) | **~0.35** |
| Verdict | **first & only economically viable config** |

### Baselines

| Model | Features | Turnover/rebal | Rank IC OOS | Net Sharpe | Verdict |
|---|---|---|---|---|---|
| Linear composite (champion) | 3 fundamental | low (~0.4) | — | **+0.18** | viable |
| TabPFN v2 | 3 fundamental | low | — | **+0.183** | ties champion |
| TabPFN v2 | all (TS+micro+regime+fund) | **1.10** | — | worse | loses (churn) |
| LightGBM LambdaRank | 40 | high | **−0.014** | ~0 | overfits |

### Price-signal context (why fundamentals win)

The price-derived signal (momentum/vol/fracdiff + Amihud + CUSUM/SADF regime) is
statistically real (gross DSR ~0.41) but **net −1.1** rebalancing every 10 days —
**sub-economic**. Cost is **turnover-driven**: same ruler, *opposite verdict* for high-
vs low-turnover. Price high-turnover gross +0.06 → net −1.1. What changes everything is
turnover, not signal strength.

## Evaluation ruler

The number that counts is **net DSR under the deflated ruler**, never gross Sharpe or
Rank IC alone:

- **CPCV** — 6 groups, 2 test → 15 splits / 5 paths; **purge + embargo in trading days**.
- **Deflated / Probabilistic Sharpe** (Bailey–LdP) with **effective N**
  (unique dates / HORIZON × avg uniqueness) and a **trials log** deflating by *all* ~21
  attempts (`cpcv.append_trial` / `deflated_sharpe(..., n_trials=real_count)`).
- **Sample uniqueness weights** (AFML ch.4) on the LightGBM fit.
- **Costs**: square-root impact, **one-way** convention.
- **Pre-registration**: the fundamental tilt was registered (hypothesis + fixed
  construction + success criterion `net Sharpe > 0`) **before** running — no metric
  shopping. See `fundamental_tilt.py` docstring.

## Intended use & limitations

**Intended use**: methodology reference and reproducible baseline for cost-aware,
deflation-honest cross-sectional equity research. A teaching artifact on
*anti-self-deception* evaluation.

**Out of scope**: live trading. Numbers are research estimates under a conservative
**cost floor** (impact is more convex in low-ADV B3 mid-caps than √ captures).

**Known limitations**
- **Survivorship**: current liquid composition, not full point-in-time membership
  (partially mitigated by including post-2018 IPOs; full PIT membership pending).
- **Small N**: ~58–112k daily bars across 111 tickers; ~28 names/quarter → tiny
  listwise queries; DL is overfit-prone by construction.
- **Cost calibration**: `half_spread_bps` and `eta` are conservative floors, not
  central estimates; participation is a placeholder until real R$ volume lands.
- **Effective N** is ~10× smaller than nominal due to overlapping 10d labels — handled
  by uniqueness weights and effective-N deflation, but it caps statistical power.

## Reproduce

```bash
uv sync
echo "BRAPI_TOKEN=your_key" > .env            # your own key; gitignored
uv run python -m stockprecog.brapi_ingest     # rebuild local panel (not shipped)
uv run python -m stockprecog.fundamental_tilt # champion (pre-registered)
uv run python -m stockprecog.rank_model       # LightGBM LambdaRank baseline
uv run python -m stockprecog.rank_tabpfn      # TabPFN v2 baseline (GPU)
uv run pytest -q                              # 26 tests (ruler invariants)
```

## Citation

```bibtex
@misc{stockprecog2026,
  title  = {stockprecog: an honest-invalidation study of cross-sectional
            tradeability prediction on B3 equities under a López de Prado ruler},
  author = {Barbosa, João Henrique},
  year   = {2026}
}
```
