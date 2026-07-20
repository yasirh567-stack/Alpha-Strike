import numpy as np
import pandas as pd
import pytest

from statarb.signals import engle_granger_pvalue, hedge_ratio, spread, zscore


def test_zscore_matches_manual_calculation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = zscore(s, lookback=3)
    window = s.iloc[1:4]
    expected = (s.iloc[3] - window.mean()) / window.std(ddof=0)
    assert z.iloc[3] == pytest.approx(expected)


def test_zscore_warmup_is_nan():
    s = pd.Series(np.arange(10, dtype=float))
    z = zscore(s, lookback=5)
    assert z.iloc[:4].isna().all()
    assert z.iloc[4:].notna().all()


def test_zscore_has_no_lookahead():
    """Changing a future value must not change past z-scores."""
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=50).cumsum())
    z_before = zscore(base, lookback=10)

    mutated = base.copy()
    mutated.iloc[40:] += 1000.0
    z_after = zscore(mutated, lookback=10)

    pd.testing.assert_series_equal(z_before.iloc[:40], z_after.iloc[:40])


def test_zscore_zero_variance_window_is_nan_not_inf():
    s = pd.Series([5.0] * 10)
    z = zscore(s, lookback=4)
    assert z.iloc[4:].isna().all()


def test_zscore_mean_reverting_series_oscillates_around_zero():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(scale=1.0, size=2000))
    z = zscore(s, lookback=30).dropna()
    assert abs(z.mean()) < 0.2
    assert z.std() == pytest.approx(1.0, abs=0.3)


def test_hedge_ratio_recovers_known_relationship():
    rng = np.random.default_rng(2)
    x = pd.Series(np.exp(rng.normal(scale=0.01, size=500).cumsum() + 4.0))
    true_beta, true_alpha = 1.7, 0.3
    noise = rng.normal(scale=0.001, size=500)
    y = np.exp(true_alpha + true_beta * np.log(x) + noise)
    alpha, beta = hedge_ratio(y, x)
    assert beta == pytest.approx(true_beta, abs=0.05)
    assert alpha == pytest.approx(true_alpha, abs=0.05)


def test_spread_is_stationary_for_cointegrated_series():
    rng = np.random.default_rng(3)
    x = pd.Series(np.exp(rng.normal(scale=0.01, size=1000).cumsum() + 4.0))
    beta = 1.2
    noise = rng.normal(scale=0.02, size=1000)
    y = np.exp(beta * np.log(x) + noise)
    alpha, fitted_beta = hedge_ratio(y, x)
    resid = spread(y, x, alpha, fitted_beta)
    assert resid.std() < 0.1


def test_engle_granger_detects_cointegrated_pair():
    rng = np.random.default_rng(4)
    x = pd.Series(np.exp(rng.normal(scale=0.01, size=1000).cumsum() + 4.0))
    noise = rng.normal(scale=0.01, size=1000)
    y = np.exp(1.5 * np.log(x) + noise)
    pvalue = engle_granger_pvalue(y, x)
    assert pvalue < 0.05


def test_engle_granger_rejects_independent_random_walks():
    rng = np.random.default_rng(5)
    x = pd.Series(np.exp(rng.normal(scale=0.01, size=1000).cumsum() + 4.0))
    y = pd.Series(np.exp(rng.normal(scale=0.01, size=1000).cumsum() + 4.0))
    pvalue = engle_granger_pvalue(y, x)
    assert pvalue > 0.05
