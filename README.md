# Alpha-Strike

A statistical arbitrage backtesting system for S&P 500 pairs trading. Screens
sector-mate pairs for cointegration, backtests a 4-parameter mean-reversion
strategy grid (1,000+ variants) with walk-forward validation, and separates
signal from noise using the Deflated Sharpe Ratio (Bailey & Lopez de Prado)
and permutation testing. The Python package is named `statarb`.

## Why this exists

Backtests lie by default. Run enough parameter combinations and some will
show a great Sharpe ratio purely by chance. This project's central question
isn't "which pairs trading strategy is best?" — it's "how many of the
strategies that look good actually survive a correction for multiple
testing?" The answer, reported at the end of the pipeline, is usually
"very few."

## Pipeline

```
statarb/
├── data.py        # S&P 500 universe, yfinance download, parquet cache
├── signals.py     # z-score signal, Engle-Granger cointegration, hedge ratios
├── backtest.py    # walk-forward splitter, strategy grid, vectorized backtest engine
├── stats.py       # Sharpe/drawdown/hit-rate, Deflated Sharpe Ratio, permutation test
└── reporting.py   # markdown report + matplotlib charts
main.py            # runs the full pipeline end to end
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Python 3.12 is recommended — the scientific stack (numpy/statsmodels/pyarrow)
has broader wheel support there than on very recent Python releases.

## Method

1. **Universe**: current S&P 500 constituents, scraped from Wikipedia (with a
   reduced hardcoded fallback list), 5+ years of daily adjusted close from
   yfinance, cached to parquet so reruns don't re-hit the network.
2. **Pair selection**: group by GICS sector, correlation-prefilter (>0.8) to
   cut ~12,800 intra-sector pairs down to a few hundred candidates, then run
   Engle-Granger cointegration and keep p < 0.05 — computed **only on the
   training window** of each walk-forward fold, never on test data.
3. **Strategy grid**: for every selected pair, cross a grid of z-score
   lookback (20/40/60/90d) × entry threshold (1.5/2.0/2.5σ) × exit threshold
   (0.0/0.5σ) — 24 variants per pair, plus a fixed 3.5σ stop-loss and 30-day
   time-stop risk overlay applied to every variant. 5,000+ variants across
   the full universe once you sum over pairs and walk-forward folds.
4. **Walk-forward backtest**: 2-year rolling train / 6-month test / 6-month
   step. Hedge ratios and cointegration are recomputed on each training
   window only — the test window never leaks into parameter estimation.
   5 bps transaction cost per side (both legs), position sizing vol-targeted
   per pair off training-window spread volatility ("equal risk per pair").
5. **Signal vs. noise**: Sharpe, max drawdown, and hit rate per variant, then
   Deflated Sharpe Ratio (Bailey & Lopez de Prado) to correct for testing
   thousands of variants simultaneously, plus a permutation test that
   reshuffles the underlying price returns (not the finished PnL — shuffling
   a fixed set of numbers can't change its own Sharpe) and reruns the exact
   same signal logic to show what pure noise produces on this universe.

## Results (from a real run, 2020-01 to 2026-07)

- **486 tickers**, 11 sectors, 1,643 trading days; 9 walk-forward folds;
  **5,112 distinct strategy variants** backtested end to end in **~80s**
  (full pipeline including data download and reporting: ~2 minutes warm-cache).
- **1,051 of 5,112 variants (20.6%) look profitable naively** (Sharpe > 1.0).
  **0 survive deflation** (DSR > 0.95). Even the single best naive Sharpe
  (4.38, on AEE/LNT) carries only a 41% probability of reflecting real skill
  once the search over 5,112 variants is accounted for.
- The number of cointegrated pairs found swings from 11 to 100 fold to fold —
  cointegration relationships are not regime-stable on this universe.
- A permutation test on that same top variant (reshuffling AEE's and LNT's
  own daily returns and rerunning the identical z-score/entry/exit logic)
  rejects the noise null in isolation (p < 0.002) — illustrating that a
  per-strategy significance test and a search-corrected one can disagree,
  which is exactly the gap this project is built to surface.
- Full charts and the top-20 table: `reports/report.md` (regenerate with
  `python main.py`; a snapshot from a real run is committed in the repo).

## Data caveats (read before trusting the results)

- **Survivorship bias**: the ticker list is *today's* S&P 500 membership
  applied retroactively. Companies that were removed from the index (delisted,
  acquired, went bankrupt) during the backtest window are absent, which
  inflates results relative to a live, point-in-time index membership.
- **Delistings**: yfinance simply has no data for tickers that no longer
  trade under their old symbol; the loader drops these silently and logs
  which tickers were skipped rather than silently corrupting the panel.
- **Adjusted close**: dividends/splits are adjusted by yfinance's own
  methodology, which can differ slightly from a paid data vendor.
- **No point-in-time GICS sectors**: sector classification is current, not
  as-of-date, so a stock that changed sector during the backtest is grouped
  by where it sits today.


## Tests

```bash
pytest -q
```

28 tests: z-score signal (lookback correctness, no lookahead, zero-variance
handling), hedge ratio / Engle-Granger recovery on synthetic cointegrated
vs. independent series, the walk-forward splitter (fold boundaries, step
size, no train/test overlap), the position state machine (entry/exit/
stop-loss/time-stop/no same-bar re-entry/NaN handling), and Sharpe/
drawdown/DSR sanity checks (including that DSR correctly suppresses false
positives on pure Gaussian noise).

## Status

Complete. Built incrementally module by module — see commit history for how
each piece was developed and verified against real data before the next one
was added.
