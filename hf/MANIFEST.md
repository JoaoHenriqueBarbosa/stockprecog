# HuggingFace upload manifest

What ships to HF and what is deliberately withheld, per the per-layer license analysis
in [LICENSE_NOTICE.md](LICENSE_NOTICE.md). `upload_hf.py` enforces this list.

## ✅ SHIPS (safe to publish)

| Item | Layer | Why safe |
|---|---|---|
| `src/stockprecog/*.py` (full pipeline) | (d) code | own work; token never committed |
| `tests/*` (26 pytest) | (d) code | ruler invariants |
| `pyproject.toml`, `uv.lock`, `.python-version` | (d) code | reproducible env |
| `README.md` (→ dataset card) | (e) results | docs |
| `README_model.md` (model card) | (e) results | docs |
| `LICENSE_NOTICE.md`, `MANIFEST.md` | meta | governance |
| `docs/*.md` (roadmap, B3 idiosyncrasies) | (e) results | methodology |
| `config.py::UNIVERSE` (111 tickers) + ADV≥R$3M rule | meta | public selection metadata |
| Aggregated metrics (DSR/PSR/Sharpe/Rank IC/turnover tables) | (e) results | statistics, not data |
| Fundamentals (CVM-derived) **with attribution** — *optional* | (c) CVM Open Data | redistributable w/ attribution; prefer CVM source |

## ❌ DOES NOT SHIP (license-restricted or secret)

| Item | Layer | Why withheld |
|---|---|---|
| `data/raw/panel_eod.parquet` (adjusted EOD prices) | (a) prices | B3 restriction + brapi silence — **max risk** |
| Any OHLC / `adjustedClose` levels | (a) prices | same |
| `.env` / `BRAPI_TOKEN` | secret | credential; never commit |
| `data/processed/*.parquet` | (a)/(b) | may embed price levels |
| Raw brapi JSON caches (`data/raw/brapi_api/*`) | (a) prices | raw vendor payloads |

## ⚠️ CONDITIONAL (not shipped by default)

| Item | Layer | Condition to ship |
|---|---|---|
| Derived non-invertible features (fracdiff, z-scores, Amihud, regime, labels+t1) | (b) grey zone | only if proven not to reconstruct OHLC; ship with explicit "no price levels" note |

## Enforcement

`upload_hf.py` uploads an explicit **allowlist** of paths and refuses if any
denylisted pattern (`*.parquet`, `.env`, `data/`, brapi caches) is present in the
staging folder. It also re-runs the git token audit before upload.
