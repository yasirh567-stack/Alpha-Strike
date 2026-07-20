"""Core signal primitives: hedge ratio, spread, Engle-Granger cointegration, z-score."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS hedge ratio for log(y) = alpha + beta*log(x) + eps. Returns (alpha, beta)."""
    ly, lx = np.log(y.to_numpy(dtype=float)), np.log(x.to_numpy(dtype=float))
    beta, alpha = np.polyfit(lx, ly, 1)
    return float(alpha), float(beta)


def spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Log-price spread: log(y) - beta*log(x) - alpha."""
    return np.log(y) - beta * np.log(x) - alpha


def engle_granger_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Two-step Engle-Granger cointegration test p-value, on log prices."""
    _, pvalue, _ = coint(np.log(y.to_numpy(dtype=float)), np.log(x.to_numpy(dtype=float)))
    return float(pvalue)


def zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score using only observations up to and including each timestamp.

    `min_periods=lookback` means the first `lookback - 1` points are NaN
    rather than computed from a partial window -- no lookahead, no
    warm-up bias from a shrinking window.
    """
    mean = series.rolling(lookback, min_periods=lookback).mean()
    std = series.rolling(lookback, min_periods=lookback).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def select_cointegrated_pairs(
    prices: pd.DataFrame,
    sectors: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    corr_prefilter: float = 0.8,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Screen intra-sector pairs for Engle-Granger cointegration.

    `prices` should already be sliced to the window the caller wants
    screened (a walk-forward training window in this project) -- this
    function has no notion of train/test, so it cannot leak future data by
    construction; that guarantee lives with the caller passing the right
    slice.

    A same-sector return-correlation prefilter (>`corr_prefilter`) runs
    first: the S&P 500 has ~12,800 intra-sector pairs, and Engle-Granger at
    ~25ms/pair makes exhaustive testing a several-minute job per
    walk-forward fold. Correlation is a cheap, vectorizable proxy for "worth
    testing" and cuts the candidate set by roughly 30x before the slower
    cointegration test runs on what's left.

    Returns DataFrame[ticker_a, ticker_b, sector, correlation, pvalue,
    alpha, beta] for pairs with pvalue < `pvalue_threshold`, sorted by
    pvalue ascending. `alpha`/`beta` are the hedge-ratio regression fit on
    this same window, in log(ticker_a) = alpha + beta*log(ticker_b) form.
    """
    returns = prices.pct_change().dropna(how="all")
    rows = []
    for sector, grp in sectors.groupby("sector"):
        tickers = sorted(t for t in grp["ticker"] if t in prices.columns)
        if len(tickers) < 2:
            continue
        corr = returns[tickers].corr()
        for a, b in combinations(tickers, 2):
            if not (corr.loc[a, b] > corr_prefilter):
                continue
            pair_prices = prices[[a, b]].dropna()
            if len(pair_prices) < min_obs:
                continue
            try:
                pvalue = engle_granger_pvalue(pair_prices[a], pair_prices[b])
                alpha, beta = hedge_ratio(pair_prices[a], pair_prices[b])
            except Exception:
                continue
            rows.append((a, b, sector, corr.loc[a, b], pvalue, alpha, beta))

    cols = ["ticker_a", "ticker_b", "sector", "correlation", "pvalue", "alpha", "beta"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df[df["pvalue"] < pvalue_threshold].sort_values("pvalue").reset_index(drop=True)
