---
title: "Honest Invalidation of a Cross-Sectional Equity Signal: A López de Prado Pipeline on the Brazilian Stock Market"
author:
  - João Henrique Barbosa
date: 2026-06-15
abstract: |
  We apply a deliberately rigorous Advances-in-Financial-Machine-Learning (AFML)
  pipeline to the cross-sectional prediction of trade success on 111 liquid
  Brazilian (B3) equities over 2010--2026, and report a study of *honest
  invalidation*. Using triple-barrier labels, Combinatorial Purged Cross-Validation
  (CPCV) with purge and embargo, sample-uniqueness weighting, fractional
  differentiation, and a Deflated Sharpe Ratio (DSR) that deflates by a registered
  log of every research trial, we find four results. First, a price-derived signal
  (momentum, volatility, fractionally-differenced price, Amihud illiquidity, and a
  CUSUM/SADF-style regime block) is *statistically* real (gross DSR $\approx 0.41$)
  but *economically* sub-marginal: net of square-root market-impact costs it
  collapses to a Sharpe of $-1.1$ when rebalanced every ten days. Second, the
  binding constraint is turnover $\times$ cost, not signal strength: holding the
  ruler fixed, the same family flips from a viable to a destroyed verdict purely as
  a function of rebalance frequency. Third, a simple linear value/quality
  fundamental tilt, rebalanced quarterly and pre-registered before estimation, is
  the only net-positive configuration we find (net Sharpe $+0.18$,
  PSR $0.92$, DSR $0.72$ deflated over 18 trials; $\approx 0.35$ annualized). We
  stress that DSR $0.72$ sits *below* the conventional $0.95$ skill-confidence
  threshold: the tilt clears its pre-registered net-positive criterion and is the
  best surviving configuration, but the evidence *leans toward* rather than
  *confirms* genuine skill, and the annualized Sharpe is modest.
  Fourth, state-of-the-art machine learning does not beat the linear composite: a
  LightGBM LambdaRank ranker overfits (out-of-sample Rank IC $-0.014$), and the
  TabPFN v2 tabular foundation model merely *ties* a zero-parameter three-factor
  linear composite on the fundamental factors and *loses* on the full feature set.
  The bottleneck is information and transaction cost, not model capacity. We frame
  the contribution as a reproducible anti-self-deception methodology together with
  an honest negative result.
keywords: [backtest overfitting, deflated Sharpe ratio, market microstructure, transaction costs, emerging markets, factor investing, financial machine learning]
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

# 1. Introduction

The efficient-market hypothesis [@fama1970efficient] sets a high bar for any claim
of predictable, tradeable structure in security prices: a candidate signal must
survive not only statistical scrutiny but also the friction of execution. In
practice, the dominant failure mode of quantitative equity research is not the
absence of in-sample structure but its *spurious abundance*. With enough features,
enough model families, and enough rebalancing rules, a backtest can be tuned to
exhibit an arbitrarily attractive Sharpe ratio that is pure selection bias. Harvey,
Liu and Zhu [@harvey2016crosssection] document hundreds of published "factors,"
most of which fail to clear a multiple-testing hurdle. Bailey and López de Prado
[@bailey2014dsr] and Bailey et al. [@bailey2017pbo] formalize the mechanism: when a
researcher selects the best of many trials, the maximum observed Sharpe ratio is
inflated by an amount that grows with the number and variance of the trials, so
that a nominally significant result is, in expectation, a statistical fluke.

The problem is compounded in three ways that are especially acute for an emerging
market such as Brazil. (i) *Transaction cost*: cross-sectional long-short signals
are turnover-intensive, and on mid-cap B3 names the price impact of trading is
convex and large relative to the thin edge of any price-derived predictor
[@amihud2002illiquidity; @toth2011sqrt]. (ii) *Overlapping labels*: holding-period
labels sampled daily induce strong serial dependence, inflating $t$-statistics by
roughly $\sqrt{H}$ for a horizon of $H$ days and corrupting the very ruler used to
adjudicate skill [@lopezdeprado2018afml]. (iii) *Survivorship and corporate-action
integrity*: a universe defined by today's liquid index, or prices not adjusted for
total return, biases base rates upward.

This paper does not present a winning strategy. Instead it presents a *methodology
of honest invalidation* and the negative result it produces. We build a pipeline
following Advances in Financial Machine Learning [@lopezdeprado2018afml] in which
every component is chosen to *remove* a self-deception channel rather than to add
edge: triple-barrier labels matched to the purge horizon, Combinatorial Purged
Cross-Validation with embargo, sample-uniqueness weighting, causal fractional
differentiation, and a Deflated Sharpe Ratio deflated by a registered log of every
trial we ran. Within this ruler we report four findings (Section 4): a
price-derived signal that is statistically real but economically sub-marginal; a
demonstration that turnover $\times$ cost, not signal strength, is the binding
constraint; a single pre-registered fundamental tilt that is the only viable
configuration; and the result that state-of-the-art machine learning does not beat
a zero-parameter linear composite. The contribution is the reproducible ruler plus
the honest verdict it returns.

# 2. Data

**Universe.** We study 111 liquid Brazilian equities listed on B3, selected by an
average daily traded value of at least R\$3\,million and a minimum of 300 daily
bars of history. The sample spans 2010-01-01 to 2026-06-15. The universe
deliberately includes initial public offerings after 2018, which partially
mitigates survivorship bias relative to the common practice of taking only names
that are liquid *today*; a fully point-in-time index-membership reconstruction
remains future work, and we treat residual survivorship as an *optimistic* bias on
all reported numbers.

**Prices.** Daily total-return series are obtained by REST scraping of the public
brapi service, using the `adjustedClose` field, which folds dividends, interest on
own capital (JCP), splits and bonuses into a single back-adjusted series. The
authoritative public source for raw Brazilian prices is the exchange's COTAHIST
files, which are *unadjusted*; the principal difficulty in constructing Brazilian
panels is not price availability but the discontinuity of asset identity across
corporate reorganizations (ticker and trading-name changes), which fragments a
company's dividend history across multiple names. Reconstructed series were
validated against an independent reference (e.g., ABEV3 reconstructed from
COTAHIST plus the exchange dividend feed matches within $\approx 2\%$, the residual
attributable to JCP and convention differences).

**Fundamentals.** Value and quality factors (earnings yield, book-to-price, profit
margin) are taken from quarterly statements and lagged by 90 days to respect the
Brazilian regulatory (CVM) filing deadline, then combined with the *live*
(non-lagged) price so that the resulting ratios are point-in-time correct: the
numerator is known, the denominator is contemporaneous.

# 3. Methodology

Every methodological choice below is included because it closes a specific channel
through which a backtest can lie. We state, for each, *what* it does and *why* it
is necessary.

## 3.1 Triple-barrier labels

Following AFML chapter 3 [@lopezdeprado2018afml], each observation is labeled by
the first of three barriers to be touched within a fixed horizon: an upper
profit-taking barrier and a lower stop-loss barrier, both set at $2\sigma_t$ where
$\sigma_t$ is an exponentially-weighted daily volatility (span 20), and a vertical
barrier at $H = 10$ trading days. The label is the sign of the realized event, and
$t_1$ is the *event-resolution date* (the touch), not the vertical barrier. *Why*:
labeling by the barrier actually hit, and propagating its resolution date $t_1$
into the validation purge, makes the predicted quantity, the realized return, and
the leakage control mutually consistent. Labeling on a fixed horizon while purging
on a different one measures one signal and validates another.

## 3.2 Combinatorial Purged Cross-Validation with embargo

We use CPCV (AFML chapter 12) with $N = 6$ groups and $k = 2$ test groups per
split, giving $\binom{6}{2} = 15$ splits and $5$ reconstructed backtest paths. Two
leakage controls are applied at every train/test boundary. *Purge*: training
observations whose label-resolution interval $[t_0, t_1]$ overlaps the test set are
removed. *Embargo*: a fixed window after each test block is excluded from training.
Critically, the embargo is set to 55 trading days, *not* the 10-day label horizon,
because the embargo must cover both information horizons present in the panel: the
label resolution ($H = 10$) and the memory of the fractionally-differenced feature
($\approx 55$ bars; Section 3.4). Without the second term, test-period prices leak
into training features through the FFD convolution, a subtle indirect leak that an
adversarial audit of the pipeline identified. *Why CPCV at all*: a single
train/test split yields one Sharpe estimate with no distribution; CPCV yields a
distribution of out-of-sample paths from which the dispersion of the performance
estimator can be measured, while purge and embargo keep each path causal.

## 3.3 Deflated and Probabilistic Sharpe Ratio with a trials log

We evaluate economic significance with the Probabilistic Sharpe Ratio (PSR) and
Deflated Sharpe Ratio (DSR) of Bailey and López de Prado [@bailey2014dsr]. Two
corrections are essential and are easy to get wrong.

*Effective sample size.* The PSR/DSR confidence in a Sharpe estimate scales with
$\sqrt{n-1}$. Naively concatenating the test returns of all CPCV splits multiplies
$n$ several-fold (each date appears in multiple test combinations) and, on top of
that, ten-day overlapping returns are serially dependent. We therefore deduplicate
by (ticker, date) and set the effective $n$ to the number of unique dates divided
by the horizon and further scaled by average sample uniqueness, so that the ruler
counts independent bets, not redundant ones.

*Deflation by real research effort.* The DSR deflates the observed Sharpe by the
expected maximum of the trials searched. A common error is to treat the variance of
the CPCV paths as the variance of *research trials* and to set the trial count to
the number of splits. CPCV paths are resamplings of a *single* configuration
(estimator dispersion), not independent trials of a multiple-testing search. We
instead maintain an explicit append-only **log of every research trial** --- every
lever, feature family, and hyperparameter configuration we ran, 21 by the end of the
study --- and deflate each result by the dispersion and count of *those* Sharpe
ratios *as logged at the time that result was computed* (the fundamental tilt of
Section 4.3 is the 18th trial and is therefore deflated by 18; the later machine-learning
runs of Section 4.4 are deflated by 20--21). *Why*: this is
the only defense against the garden of forking paths. A DSR that deflates by an
honest trial count is conservative by construction; one that deflates by a
cross-validation knob is "right for the wrong reason" and would fail to detect skill
if it existed.

## 3.4 Fractional differentiation

Price levels are non-stationary; returns are stationary but memoryless. Following
AFML chapter 5 [@lopezdeprado2018afml] we apply fixed-width-window fractional
differentiation (FFD) with order $d = 0.4$ and a weight-truncation threshold of
$10^{-3}$, which yields a causal window of $\approx 55$ bars. *Why*: $d = 0.4$ is
the smallest differencing order that renders the price series stationary while
preserving the maximum amount of memory, so the resulting feature retains
predictive structure that a first difference (return) discards. In our price-derived
ablation this single feature accounts for the large majority of the realized
discrimination gain.

## 3.5 Sample-uniqueness weighting

Overlapping labels make daily samples redundant: when a ten-day label is sampled
every day, consecutive observations describe nearly the same outcome. Following AFML
chapter 4 [@lopezdeprado2018afml] we compute the average uniqueness of each label
from the concurrency of live $t_1$ intervals and pass it as a sample weight to every
estimator. *Why, and a deliberate restriction*: we weight by uniqueness *only*, not
by uniqueness $\times$ return attribution. Uniqueness down-weights redundant,
crowded regimes and *preserves* the edge; return-attribution weighting, by
contrast, up-weights large-move samples and we found it *destroys* the measured
edge, conflating outcome magnitude with sample importance. Uniqueness also rescales
the effective $n$ used in the PSR/DSR (Section 3.3).

## 3.6 Transaction-cost model: square-root law, one-way

Gross long-short Sharpe ratios are optimistic fiction by construction. We charge a
cost composed of a linear spread/brokerage term plus a market-impact term following
the empirical square-root law of Tóth et al. [@toth2011sqrt],
$$
\text{impact} \;\approx\; \eta\,\sigma\,\sqrt{Q/\mathrm{ADV}},
$$
where $Q$ is order size, ADV is average daily traded value in R\$, and $\sigma$ is
daily volatility. The square-root (concave) form is more realistic than the linear
impact of Almgren-Chriss, which we treat only as historical context for scheduled
execution rather than as an aggregate cost model. Cost is charged per *one-way* leg
proportional to period turnover, *not* per round trip, because turnover already
accounts for each side of a rotation as it occurs over time. We treat the chosen
half-spread (5--10 bps) and $\eta = 1$ as a *conservative floor*: on low-ADV B3
names the true impact is more convex than the square root and the spread widens
under liquidity stress, so the reported costs *underestimate* the tail. *Why this
matters*: as Section 4 shows, the cost model is not a footnote; it is the single
component that reverses the verdict on the price-derived family.

All features are computed point-in-time; the causal chain was audited adversarially
for leakage, and a suite of 26 `pytest` regression tests locks the
label$\leftrightarrow t_1\leftrightarrow$return alignment, the purge/embargo
geometry, the FFD weights, and the PSR/DSR formulas.

# 4. Results

We report four findings. All economic numbers are net of the square-root,
one-way cost model (Section 3.6) unless explicitly labeled "gross," and all
significance is deflated by the registered trials log (Section 3.3). The numbers
are reported as measured; none are inflated or rounded in the favorable direction.

## 4.1 The price-derived signal is statistically real but economically sub-marginal

The price-derived family stacks momentum and volatility features, the
fractionally-differenced price (Section 3.4), an Amihud illiquidity microstructure
block [@amihud2002illiquidity], and a regime block built from CUSUM and an
SADF-style explosiveness statistic [@phillips2015bubbles]. Table 1 traces the
discrimination of this family as components are added. Cross-sectional AUC rises
from a near-coin-flip baseline of $0.5125$ to $0.5232$, with fractional
differentiation contributing roughly three-quarters of the gain --- a *real*
improvement in ranking discrimination.

Table: Price-derived ablation. AUC is cross-sectional ranking discrimination; the
economic verdict is net of costs and deflated. Discrimination improves and is
attributable mostly to fractional differentiation, yet the economic edge never
separates from zero.

| Lever | Component added | AUC | $\Delta$AUC | Economic verdict |
|------|------------------|-----|-------------|------------------|
| L0 | baseline (returns) | 0.5125 | --- | none (single split) |
| L1 | clean total-return panel | 0.5125 | $+0.0000$ | de-risks data, no edge |
| L2 | CPCV + DSR ruler | 0.5146 | $+0.0021^\ast$ | reveals no economic edge |
| L3 | fractional differentiation | 0.5232 | $+0.0086$ | only real AUC jump |
| L4 | meta-labeling | 0.5224 | $-0.0008$ | first negative marginal |

$^\ast$ L2 changed the ruler, not the feature; its $\Delta$AUC is not comparable.

Measured as a strategy, the full price-derived family achieves a *gross* DSR of
$\approx 0.41$. Net of the square-root cost model, rebalanced every ten days, it
collapses to a Sharpe of $-1.1$ --- sub-economic. The result is robust: decile
construction does not rescue it, a longer horizon does not rescue it, and a
spread-only floor leaves it near zero even on the largest, most liquid names. The
discrimination is genuine; the conversion to profit-and-loss is destroyed by
trading friction.

## 4.2 The binding constraint is turnover $\times$ cost, not signal strength

Holding the ruler, the universe, and the signal fixed, the economic verdict is
governed by *turnover*. Table 2 contrasts the same price-derived family at high and
low rebalance intensity. A high-turnover configuration with a barely-positive gross
Sharpe of $+0.06$ is driven to a net Sharpe of $-1.1$; the cost wedge, not the
signal, decides the outcome. This is the central diagnostic of the paper: a fixed
ruler returns *opposite* verdicts for high- versus low-rotation deployments of an
identical predictor.

Table: Turnover contrast for the price-derived family. The signal is unchanged;
only rebalance frequency differs. Cost, scaling with turnover, reverses the verdict.

| Configuration | Rebalance | Gross Sharpe | Net Sharpe | Verdict |
|---------------|-----------|--------------|-----------|---------|
| price, high-turnover | 10 days | $+0.06$ | $-1.1$ | destroyed by cost |
| price, spread-only floor | 10 days | $\approx 0$ | $\approx 0$ | no edge to spend |

## 4.3 A pre-registered fundamental tilt is the only viable configuration

We pre-registered, before estimation, a single hypothesis (H1): a long-short
value/quality tilt, rebalanced *quarterly* (63 trading days), would clear a net
Sharpe above zero after one-way costs, precisely because low turnover keeps the toll
beneath a weak-but-real fundamental edge. The construction is fixed and untuned: the
score is the equal-weighted average of the cross-sectional percentile ranks of
earnings yield, book-to-price and profit margin [@fama1992crosssection;
@fama1993factors; @novy2013quality]; the portfolio is long the top tercile, short
the bottom tercile, equal-weighted and market-neutral; the return is the forward
return of each name to the next rebalance; the acceptance criterion was net
Sharpe $> 0$, with the complementary outcome declared a confirmation of the
kill-criterion.

Table: Pre-registered quarterly fundamental tilt. The first and only economically
viable configuration in the study. DSR is deflated over the full 18-trial log;
the annualized figure scales the per-rebalance net Sharpe by $\sqrt{252/63}$.

| Quantity | Value |
|----------|-------|
| Sharpe, gross | $+0.19$ |
| Sharpe, net (one-way costs) | $+0.18$ |
| PSR ($H_0:\,\mathrm{SR}=0$) | $0.92$ |
| DSR (deflated, 18 trials) | $0.72$ |
| Sharpe, net, annualized | $\approx 0.35$ |

The pre-registered hypothesis passed its stated criterion (net Sharpe $> 0$): net
Sharpe $+0.18$, PSR $0.92$, and a DSR of $0.72$ deflated over all 18 logged trials.
We are careful not to overstate this. PSR $0.92$ and DSR $0.72$ both fall *short* of
the conventional $0.95$ confidence threshold for declaring genuine skill; the result
clears the (weaker) pre-registered net-positive bar and is the only net-positive
configuration in the study after honest costs and honest deflation, but it is best
read as *promising, not confirmed*, and the annualized Sharpe ($\approx 0.35$) is
modest. Its (tentative) viability is *mechanistic*: quarterly rebalancing cuts turnover by
roughly an order of magnitude relative to the ten-day price-derived deployment,
moving the same class of weak edge from the wrong to the right side of the cost
wedge identified in Section 4.2.

## 4.4 State-of-the-art machine learning does not beat the linear composite

If the bottleneck were model capacity, a more expressive learner should extract
more edge from the same features. It does not. Table 3 compares two
state-of-the-art models against the zero-parameter three-factor linear composite of
Section 4.3.

A LightGBM LambdaRank cross-sectional ranker [@ke2017lightgbm; @burges2010ranknet]
over 40 features overfits: its out-of-sample Rank IC is $-0.014$ and its net edge is
$\approx 0$, on an effective sample of only $\approx 58$ quarters. The TabPFN v2
tabular foundation model [@hollmann2025tabpfn], purpose-built for the small-$N$
regime that exactly characterizes this problem, merely *ties* the linear composite
on the fundamental factors ($+0.183$ versus $+0.18$) and *loses* on the full feature
set, where it is pushed to a turnover of $1.10$ per rebalance and churns away its
edge.

Table: Model comparison. The foundation model ties a zero-parameter linear
composite on the viable factors and underperforms on the full set. Capacity is not
the constraint.

| Model | Feature set | OOS Rank IC | Net edge | Note |
|-------|-------------|-------------|----------|------|
| Linear composite (3 factors) | value/quality | --- | $+0.18$ | zero parameters |
| LightGBM LambdaRank | 40 features | $-0.014$ | $\approx 0$ | overfits, eff. $N\approx 58$ |
| TabPFN v2 | value/quality | --- | $+0.183$ | ties the linear composite |
| TabPFN v2 | full features | --- | $<+0.18$ | turnover 1.10, churn |

The verdict is unambiguous: a three-factor linear composite with no free parameters
equals the SOTA tabular foundation model. The constraint is *information and cost*,
not model sophistication.

# 5. Discussion

**Turnover is the gargoyle on the gate.** The four results compose into one claim:
across deployments of the same family, the variable that flips the economic verdict
is turnover $\times$ cost, not the strength of the predictor (Sections 4.1--4.2). A
gross DSR of $0.41$ is not "alpha"; it is a discrimination statistic that the square
-root impact law erases before it becomes P&L. Any study that reports gross Sharpe
ratios for a turnover-intensive cross-sectional signal in an emerging market is, in
effect, reporting the cost it chose not to charge.

**Information dominates model.** That a zero-parameter linear composite ties a
tabular foundation model (Section 4.4) is not a statement that TabPFN is weak --- on
the contrary, it is exactly the right tool for a small-$N$ tabular problem
[@hollmann2025tabpfn]. It is a statement about *where the edge lives*. With $\approx
58$ effective quarters and three weakly-informative factors, the achievable edge is
pinned by the information content of the inputs and the cost of acting on them, and
no amount of capacity manufactures signal that the data do not contain. The
implication for practitioners is to spend the next unit of effort on *orthogonal,
non-price information* and on *lowering turnover*, not on a larger model.

**The fundamental tilt as the only net-positive configuration.** The single
configuration that clears the pre-registered net-positive bar is the simplest one: a slow, linear, value/quality tilt
(Section 4.3). Its survival is not a triumph of cleverness but of *parsimony under
friction* --- a low-turnover deployment of a weak-but-real factor premium that has
survived decades of out-of-sample scrutiny in developed markets [@fama1992crosssection;
@novy2013quality]. Even so, an annualized net Sharpe of $\approx 0.35$ is modest, and
we report it as such.

**On the honesty of the deflated DSR.** The DSR of $0.72$ for the fundamental tilt
is meaningful precisely because it is deflated by all 18 logged trials, not by a
cross-validation knob (Section 3.3). The honest version of this ruler is
*adversarial to its own author*: it raises the bar in proportion to how hard we
searched. The price-derived family failed that bar; the fundamental tilt cleared it.
We caution that our trials were sequential and adaptive, so even a registered-log
deflation likely *understates* the true search effort [@bailey2017pbo], and that
residual survivorship biases all numbers optimistically.

**Limitations.** The universe is not yet fully point-in-time; the cost model is a
conservative floor whose tail is, if anything, worse on illiquid names; the effective
sample for quarterly tests is small; and the embargo and uniqueness corrections,
while necessary, cannot fully repair the limited number of independent bets available
in sixteen years of Brazilian data.

# 6. Conclusion

A rigorous López de Prado pipeline applied to Brazilian equities yields a study of
honest invalidation. Price-derived signals are statistically real but economically
sub-marginal once square-root transaction costs are charged; the binding constraint
is turnover $\times$ cost rather than signal strength; a simple, pre-registered,
quarterly value/quality fundamental tilt is the only economically viable
configuration we found; and a state-of-the-art tabular foundation model does no
better than a zero-parameter linear composite, identifying information and cost ---
not model capacity --- as the bottleneck. The contribution is a reproducible,
anti-self-deception methodology, and an honest negative result that we believe is
more useful to the field than another un-costed, un-deflated backtest. The full
pipeline, trials log, and tests are released to make the verdict reproducible.

# References
