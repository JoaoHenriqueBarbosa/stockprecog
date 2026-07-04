# Security Policy

## Supported Versions

This is a research project archived under a DOI. Security fixes, when applicable, are applied to the
latest state of the `main` branch only.

| Version | Supported |
|---|---|
| `main` (latest) | :white_check_mark: |
| older tags / archived DOI versions | :x: |

## Reporting a Vulnerability

Please report security issues **privately** by email to **joaohenriquebarbosa21@gmail.com** — do not
open a public issue for a suspected vulnerability.

We aim to acknowledge your report within **72 hours**.

Please include, where possible:

- a clear description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- any relevant logs, versions, or environment details.

## Process

1. **Acknowledge** — we confirm receipt within 72 hours.
2. **Assess** — we reproduce the issue and determine severity and scope.
3. **Fix** — we develop and test a remediation on `main`.
4. **Release** — we publish the fix and, where relevant, update the archived release.
5. **Disclose** — we credit the reporter (unless anonymity is requested) once a fix is available.

## Scope

Because this repository is a data/ML research pipeline (not a network service), the most relevant
classes of issue are:

- **Injection or unsafe input handling** — e.g. unsafe parsing of the brapi/CVM JSON caches,
  COTAHIST files, or other untrusted inputs consumed by the pipeline.
- **Data or credential exposure** — e.g. a code path that could leak the `BRAPI_TOKEN` (or any
  secret) into logs, artifacts, or git history. The token must stay in `.env` (gitignored) and never
  be committed.
- **Dependency vulnerabilities** — known CVEs in declared dependencies (NumPy, pandas, scikit-learn,
  LightGBM, PyTorch, statsmodels, etc.) or in the ML checkpoints fetched at runtime.

Out of scope: the statistical validity of the research results (that is a scientific, not a security,
matter — use a regular issue), and the redistribution terms of upstream market data (see
`hf/LICENSE_NOTICE.md`).
