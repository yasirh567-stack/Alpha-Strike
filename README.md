# statarb

A statistical arbitrage backtesting system for S&P 500 pairs trading. Screens
sector-mate pairs for cointegration, backtests a 4-parameter mean-reversion
strategy grid (1,000+ variants) with walk-forward validation, and separates
signal from noise using the Deflated Sharpe Ratio (Bailey & Lopez de Prado)
and permutation testing.

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
   hardcoded fallback list), 5+ years of daily adjusted close from yfinance,
   cached to parquet so reruns don't re-hit the network.
2. **Pair selection**: group by GICS sector, run Engle-Granger cointegration
   on every intra-sector pair, keep p < 0.05 — computed **only on the
   training window** of each walk-forward fold, never on test data.
3. **Strategy grid**: for every selected pair, cross a grid of z-score
   lookback (20/40/60/90d) × entry threshold (1.5/2.0/2.5σ) × exit threshold
   (0.0/0.5σ) × stop rule (3.5σ stop-loss, 30-day time stop) — 48 variants
   per pair, 1,000+ across the universe.
4. **Walk-forward backtest**: 2-year rolling train / 6-month test / 6-month
   step. Hedge ratios and cointegration are recomputed on each training
   window only — the test window never leaks into parameter estimation.
   5 bps transaction cost per side, equal-risk position sizing per pair.
5. **Signal vs. noise**: Sharpe, max drawdown, and hit rate per variant, then
   Deflated Sharpe Ratio to correct for testing 1,000+ variants
   simultaneously, plus a permutation test (shuffled returns) to show what
   pure noise looks like on this same universe. The report states explicitly
   how many variants survive deflation vs. how many looked good naively.

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

This is a research/portfolio project, not a trading system — the biases
above are exactly the kind of thing that would need fixing before real
capital touches this.

## Tests

```bash
pytest -q
```

Covers the z-score signal (lookback correctness, no lookahead) and the
walk-forward splitter (fold boundaries, step size, no train/test overlap).

## Status

Work in progress, built incrementally module by module. See commit history.
