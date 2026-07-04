## Summary

<!-- What does this PR change, and why? -->

## Related issues

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix (`fix`)
- [ ] New feature (`feat`)
- [ ] Research experiment / result (`exp`)
- [ ] Documentation (`docs`)
- [ ] Tests (`test`)
- [ ] Refactor / perf / chore

## Impact on reported numbers

<!-- Does this change any metric reported in the README or paper (DSR/PSR, Sharpe, Rank IC,
     ablation tables, turnover)? If so, list the old → new values and update the docs. If not,
     write "none". -->

## Checklist

- [ ] `uv sync` succeeds
- [ ] `uv run pytest` passes locally
- [ ] Added/updated tests for the change (especially for leakage-sensitive code: labeling, CPCV,
      embargo/purge, weights, feature timing)
- [ ] No data, secrets, or `BRAPI_TOKEN` committed (`data/` stays gitignored)
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Docs (README / paper / cards) updated to match any changed behavior or numbers
