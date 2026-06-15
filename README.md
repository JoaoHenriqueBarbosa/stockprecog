# stockprecog

Modelo de ML cross-sectional pra prever **P(trade dá certo)** em ações da B3,
seguindo o protocolo López de Prado (triple-barrier, validação purgada).

## Estado: loop end-to-end fechado (baseline honesta)

- Universo: subconjunto líquido do IBrX (~24 tickers, EOD ajustado 2010–2025).
- Target: triple-barrier vol-scaled (PT/SL = 2σ, horizonte 10d).
- Features: momentum, volatilidade, RSI + ranks cross-section.
- Modelo: LightGBM, split temporal **purgado** (embargo = horizonte).
- **Baseline: ROC AUC ≈ 0.5125** (out-2020 → 2025). Perto do acaso — esperado e
  honesto: confirma ausência de vazamento. É a régua pra todo ganho futuro.

## Limitações conhecidas
- **Survivorship bias**: usa composição ATUAL do índice. Próximo passo é
  composição point-in-time.
- yfinance derruba ~6 tickers por TLS/rate-limit; ingestão tem retry mas
  segue com o que conseguir.

## Rodar
```bash
uv run python -m stockprecog.loop
```

## Roadmap (do sólido pro frontier)
1. ✅ Loop mínimo honesto.
2. Combinatorial Purged CV + Deflated Sharpe Ratio.
3. Fractional differentiation, meta-labeling, features ricas.
4. Frontier: iTransformer / foundation models de série temporal (GPU).
