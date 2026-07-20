"""Performance metrics, Deflated Sharpe Ratio, and permutation testing.

Deflated Sharpe Ratio (DSR) follows Bailey & Lopez de Prado (2014), "The
Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
and Non-Normality". It answers: given that N variants were tried, what's
the probability this variant's *true* Sharpe exceeds the Sharpe you'd
expect the best of N pure-luck trials to show?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from statarb.backtest import COST_BPS_PER_SIDE, STOP_LOSS_Z, TIME_STOP_DAYS, generate_positions
from statarb.signals import zscore

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015328


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    if np.isclose(std, 0.0, atol=1e-12):
        return np.nan
    return float(r.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of *trading* days (nonzero return -- a position was open)
    that were profitable. Flat/zero-return days are excluded rather than
    counted as losses, since they carry no signal about the strategy's
    edge."""
    r = returns.dropna()
    active = r[r != 0]
    if len(active) == 0:
        return np.nan
    return float((active > 0).mean())


def summarize_variant(returns: pd.Series) -> dict:
    r = returns.dropna()
    return {
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(r),
        "n_active_days": int(len(r)),
        "n_trading_days": int((r != 0).sum()),
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else np.nan,
        "annualized_return": float(r.mean() * TRADING_DAYS) if len(r) else np.nan,
    }


def summarize_all(variant_returns: pd.DataFrame) -> pd.DataFrame:
    """Sharpe/max drawdown/hit rate/etc for every variant column."""
    rows = {vid: summarize_variant(variant_returns[vid]) for vid in variant_returns.columns}
    return pd.DataFrame(rows).T


def probabilistic_sharpe_ratio(
    sr_observed: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """PSR: P(true Sharpe > sr_benchmark), correcting for skewness/kurtosis
    in the return distribution (Bailey & Lopez de Prado, 2012). `kurtosis`
    is the *non-excess* convention (normal distribution == 3.0)."""
    if n_obs < 2:
        return np.nan
    denom = np.sqrt(max(1 - skew * sr_observed + (kurtosis - 1) / 4 * sr_observed**2, 0.0))
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (sr_observed - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(sp_stats.norm.cdf(z))


def expected_max_sharpe_under_null(sharpe_std: float, n_trials: int) -> float:
    """E[max SR] you'd expect across `n_trials` independent zero-skill
    strategies, given the cross-sectional spread of estimated Sharpes
    (extreme value approximation, Bailey & Lopez de Prado 2014)."""
    if n_trials < 2 or sharpe_std == 0 or np.isnan(sharpe_std):
        return 0.0
    inv = sp_stats.norm.ppf
    term1 = (1 - EULER_MASCHERONI) * inv(1 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * inv(1 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * (term1 + term2))


def deflated_sharpe_ratio(variant_returns: pd.DataFrame) -> pd.Series:
    """DSR for every variant column: probability its true (per-period)
    Sharpe exceeds the Sharpe the best of len(variant_returns.columns)
    pure-noise trials would be expected to produce by chance alone.

    Per-period (non-annualized) Sharpe/skew/kurtosis are used internally,
    consistent with the PSR/DSR formulas as originally specified; results
    (a probability in [0, 1]) don't depend on the annualization convention.
    """
    n_trials = variant_returns.shape[1]
    per_period_sr, skews, kurts, n_obs = {}, {}, {}, {}
    for vid in variant_returns.columns:
        r = variant_returns[vid].dropna()
        n_obs[vid] = len(r)
        std = r.std(ddof=1)
        if len(r) < 4 or np.isclose(std, 0.0, atol=1e-12):
            per_period_sr[vid] = np.nan
            skews[vid] = np.nan
            kurts[vid] = np.nan
            continue
        per_period_sr[vid] = r.mean() / std
        skews[vid] = sp_stats.skew(r)
        kurts[vid] = sp_stats.kurtosis(r, fisher=False)

    sr_series = pd.Series(per_period_sr)
    sr0 = expected_max_sharpe_under_null(sr_series.std(), n_trials)

    dsr = {
        vid: probabilistic_sharpe_ratio(sr_series[vid], sr0, n_obs[vid], skews[vid], kurts[vid])
        for vid in variant_returns.columns
    }
    return pd.Series(dsr, name="deflated_sharpe")


def permutation_null_sharpes(
    price_a: pd.Series,
    price_b: pd.Series,
    alpha: float,
    beta: float,
    lookback: int,
    entry: float,
    exit: float,
    stop: float = STOP_LOSS_Z,
    time_stop: float = TIME_STOP_DAYS,
    cost_bps_per_side: float = COST_BPS_PER_SIDE,
    n_permutations: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of Sharpe ratios for one (pair, parameter) variant.

    Shuffling a *finished* return series cannot change its Sharpe -- mean
    and std are invariant to reordering a fixed set of numbers. Instead this
    randomly reorders each leg's own daily log-returns (destroying any real
    lead/lag or co-movement between the two tickers while preserving each
    leg's own marginal return distribution), rebuilds a synthetic price path
    from the shuffled returns, and reruns the *exact same* z-score/entry/
    exit/stop logic on it. Because that logic is path-dependent, each
    permutation now genuinely produces a different sequence of trades and a
    different Sharpe -- this is what actually varies under the null of "no
    real relationship between these two tickers."
    """
    rng = np.random.default_rng(seed)
    log_a = np.log(price_a.to_numpy())
    log_b = np.log(price_b.to_numpy())
    ret_a = np.diff(log_a)
    ret_b = np.diff(log_b)
    T = len(ret_a)

    z_cols = np.empty((T + 1, n_permutations))
    pair_ret_cols = np.empty((T + 1, n_permutations))

    for k in range(n_permutations):
        perm = rng.permutation(T)
        shuffled_a = np.concatenate([[log_a[0]], log_a[0] + np.cumsum(ret_a[perm])])
        shuffled_b = np.concatenate([[log_b[0]], log_b[0] + np.cumsum(ret_b[perm])])
        spread = pd.Series(shuffled_a - beta * shuffled_b - alpha)
        z_cols[:, k] = zscore(spread, lookback).to_numpy()
        pair_ret_cols[:, k] = np.concatenate([[0.0], np.diff(shuffled_a) - beta * np.diff(shuffled_b)])

    entry_arr = np.full(n_permutations, entry)
    exit_arr = np.full(n_permutations, exit)
    stop_arr = np.full(n_permutations, stop)
    time_stop_arr = np.full(n_permutations, time_stop)

    positions = generate_positions(z_cols, entry_arr, exit_arr, stop_arr, time_stop_arr).astype(float)
    shifted = np.vstack([np.zeros((1, n_permutations)), positions[:-1]])
    gross = shifted * pair_ret_cols
    turnover = np.abs(np.diff(positions, axis=0, prepend=0))
    cost = turnover * (cost_bps_per_side / 10_000.0) * 2
    net = gross - cost

    std = net.std(axis=0, ddof=1)
    safe_std = np.where(std > 0, std, 1)
    sharpes = np.where(std > 0, net.mean(axis=0) / safe_std * np.sqrt(TRADING_DAYS), 0.0)
    return sharpes


def pooled_permutation_null(
    variant_meta: pd.DataFrame,
    prices: pd.DataFrame,
    sample_size: int = 30,
    n_permutations: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Pooled null Sharpe distribution across a random sample of variants.

    Re-estimates each sampled pair's hedge ratio fresh on the full available
    price history (this is for the aggregate noise-distribution chart, not
    the walk-forward backtest itself, so reusing the full history here is
    fine) and runs `permutation_null_sharpes` on each, pooling every
    permutation's Sharpe into one array so the report can show "here's what
    Sharpes pure noise produces on this exact universe."
    """
    from statarb.signals import hedge_ratio  # local import: avoids a cycle at module load time

    rng = np.random.default_rng(seed)
    sample = variant_meta.sample(n=min(sample_size, len(variant_meta)), random_state=seed)
    pooled = []
    for vid, row in sample.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        pair_prices = prices[[a, b]].dropna()
        if len(pair_prices) < row["lookback"] * 2:
            continue
        alpha, beta = hedge_ratio(pair_prices[a], pair_prices[b])
        sharpes = permutation_null_sharpes(
            pair_prices[a], pair_prices[b], alpha, beta,
            lookback=int(row["lookback"]), entry=row["entry"], exit=row["exit"],
            n_permutations=n_permutations, seed=int(rng.integers(1_000_000)),
        )
        pooled.append(sharpes)
    return np.concatenate(pooled) if pooled else np.array([])


def signal_vs_noise_summary(
    variant_stats: pd.DataFrame, dsr: pd.Series, naive_sharpe_threshold: float = 1.0, dsr_threshold: float = 0.95
) -> dict:
    """How many of the variants look good naively vs. how many still look
    good after correcting for having tried all of them."""
    n_total = len(variant_stats)
    naive_good = variant_stats["sharpe"] > naive_sharpe_threshold
    survives = dsr.reindex(variant_stats.index) > dsr_threshold
    return {
        "n_total_variants": n_total,
        "naive_sharpe_threshold": naive_sharpe_threshold,
        "n_naive_good": int(naive_good.sum()),
        "dsr_threshold": dsr_threshold,
        "n_survive_deflation": int(survives.sum()),
        "n_naive_good_that_survive": int((naive_good & survives).sum()),
    }
