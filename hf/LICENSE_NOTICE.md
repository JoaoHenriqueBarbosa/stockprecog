# License & redistribution notice

**Not legal advice.** For material legal risk, confirm in writing with brapi and cite
CVM/B3 as primary sources.

## Summary

This project publishes **code + reproduction recipe + aggregated results**. It does
**not** publish the raw or adjusted B3 price panel. This is the same posture the source
repo already adopts (`.gitignore` excludes parquet datasets; the brapi token lives in
`.env`, gitignored, and **was never committed** — audited via `git log`).

## Per-layer analysis

The data behind this project splits into layers with **opposite** license regimes.
The redistribution decision is made per layer, not globally.

### (a) Raw / adjusted prices — `panel_eod.parquet`, `adjustedClose` EOD, OHLC
**Decision: DO NOT redistribute.** This layer stacks two risks:
- **B3**: expressly **prohibits** redistribution / republication / reformatting of
  market-data bases to third parties without prior consent, reserves audit rights, and
  hardened its policy in 2026.
- **brapi**: contractually **silent** — permits commercial use but neither authorizes
  nor forbids republishing the raw dataset. **Silence ≠ permission**; the burden of
  proving the right falls on whoever redistributes.

The adjusted-price panel is the **maximum-risk** layer (B3 restriction + brapi silence).

### (b) Derived non-invertible features — fracdiff, normalized returns/vol, cross-section
z-scores, Amihud, CUSUM/SADF regime, triple-barrier labels + t1
**Decision: grey zone, low–moderate risk.** Publishable **only if** they provably do
**not** reconstruct OHLC / `adjustedClose`. If ever shipped, must carry an explicit note
that they contain no price levels. **Not shipped by default** in this release.

### (c) Fundamentals — `earnings_yield`, `book_to_price`, `profit_margin`, `div_yield`
**Decision: redistributable WITH attribution.** Underlying source is **CVM Open Data**
(LAI + Open Data Policy). Required attribution: *"data accessed via the CVM Open Data
Portal."* Preference: point to the CVM source rather than mirror.

### (d) Code / pipeline — `src/stockprecog/*`
**Decision: redistributable, unrestricted.** Own work. **Condition**: the brapi token
must NEVER appear in git history (`.env` gitignored; `git log` audited clean).

### (e) Results / aggregated metrics — DSR/PSR, gross/net Sharpe per config, Rank IC OOS,
ablation tables, turnover
**Decision: safe.** Statistics, not redistributable data.

## Recommendation

Adopt the **"code + recipe"** model (standard in reproducible quant finance):

1. Publish the **complete pipeline** + **instructions** for the reader to download the
   data with their **own** brapi key + CVM parsers.
2. Publish **results / metrics** freely.
3. For **fundamentals**, prefer pointing to **CVM Open Data** with attribution, or
   republish only those (they are open, unlike prices).
4. If a **feature dataset** on HF is desired, publish **only** derived non-invertible
   features that provably do not reconstruct OHLC, and document that fact.
5. Before any publication including brapi data beyond transformed features, request
   **written authorization** from brapi.
6. **Audit git history** to ensure the token was never committed. ✅ Done — clean.

## Token audit (performed)

```
$ git log --all --oneline -- .env          # → empty (never tracked)
$ git log --all -S "<token>"               # → empty (never committed)
```

The `.env` file holding `BRAPI_TOKEN` is local-only and gitignored.
