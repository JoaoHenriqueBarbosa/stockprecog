# Contributing to stockprecog

Thanks for your interest! This is a research project (a completed spike archived under a DOI), so
contributions tend to be reproductions, methodology fixes, extra tests, or documentation
improvements. All are welcome. Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules specific to this repo

- **Never commit data or secrets.** The `BRAPI_TOKEN` lives only in `.env` (gitignored). Raw price
  panels, brapi/CVM caches, and parquet files under `data/` are **not** committed and are **not**
  redistributable (B3 market data). See [`hf/LICENSE_NOTICE.md`](hf/LICENSE_NOTICE.md) before adding
  or sharing any dataset artifact.
- **Guard against look-ahead leakage.** The whole point of this project is a leak-free evaluation
  ruler. Any change to labeling, CPCV, embargo/purge, weights, or feature timing must preserve
  point-in-time correctness — and should come with a test that would fail if it leaked.
- **Keep claims honest and grounded in code.** If you change a reported number, update the README /
  paper to match what the code actually produces. Don't overstate results.

## Development setup

Requires **Python 3.12** and [**uv**](https://docs.astral.sh/uv/).

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/stockprecog
cd stockprecog

# 2. Install everything, including the dev group (pytest)
uv sync

# 3. Configure your brapi token (only needed if you plan to reproduce the data)
cp .env.example .env    # then edit .env and add your token from https://brapi.dev
```

## Running tests

```bash
uv run pytest                 # full suite (fast; sub-second)
uv run pytest -q              # quiet
uv run pytest tests/test_cpcv.py                          # one file
uv run pytest tests/test_cpcv.py::test_embargo_trading_days   # one test
```

The suite lives in `tests/` and covers the load-bearing parts of the ruler: CPCV purge/embargo, cost
accounting, triple-barrier labeling, feature construction, and uniqueness weights. **Please add or
update tests** for any behavior change, especially anything touching leakage-sensitive code.

## Pull request workflow

1. **Fork** the repository and create a branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. **Make your change.** Keep it focused; one logical change per PR.
3. **Run the tests** (`uv run pytest`) and make sure they pass. Add tests for new behavior.
4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/) (see table below).
5. **Push** to your fork and **open a PR** against `main`. Fill in the PR template and describe *what*
   changed and *why*, including any impact on reported numbers.

### Conventional commit types

| Type | When to use it |
|---|---|
| `feat` | A new capability (new feature block, evaluation mode, entrypoint) |
| `fix` | A bug fix (including leakage or accounting bugs) |
| `exp` | A research experiment / result (matches this repo's history, e.g. an ablation) |
| `docs` | Documentation only (README, paper, cards, docstrings) |
| `test` | Adding or fixing tests, no production-code change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A performance improvement (e.g. vectorizing a panel build) |
| `chore` | Tooling, deps, packaging, CI, metadata |

Example:

```
fix(cpcv): measure embargo in trading days, not calendar days
```

## Code style

- Target **Python 3.12**; keep the existing NumPy/pandas-first, vectorized style.
- Prefer clear, well-named functions and short module-level docstrings that explain *why* (the repo's
  docstrings often record the reasoning behind a choice — keep that habit).
- No enforced formatter is configured; match the surrounding code.

## Reporting bugs & requesting features

Use the issue templates: **Bug Report** and **Feature Request**. For anything security-related, follow
[`SECURITY.md`](SECURITY.md) and email privately instead of opening a public issue.
