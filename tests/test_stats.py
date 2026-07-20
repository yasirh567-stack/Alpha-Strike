import numpy as np
import pandas as pd
import pytest

from statarb.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)


def test_sharpe_ratio_known_value():
    r = pd.Series([0.01, -0.01, 0.01, -0.01, 0.02])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_ratio_zero_vol_is_nan():
    r = pd.Series([0.01] * 10)
    assert np.isnan(sharpe_ratio(r))


def test_max_drawdown_known_path():
    # equity: 1 -> 1.1 -> 0.99 -> 1.155; trough 0.99 vs peak 1.1 = -10%
    r = pd.Series([0.10, -0.10, 0.1666666667])
    dd = max_drawdown(r)
    assert dd == pytest.approx(-0.10, abs=1e-6)


def test_max_drawdown_is_zero_for_monotonic_gains():
    r = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert max_drawdown(r) == pytest.approx(0.0)


def test_hit_rate_ignores_flat_days():
    r = pd.Series([0.01, 0.0, -0.01, 0.0, 0.02, 0.0])
    assert hit_rate(r) == pytest.approx(2 / 3)


def test_expected_max_sharpe_increases_with_more_trials():
    sr_low = expected_max_sharpe_under_null(sharpe_std=0.5, n_trials=10)
    sr_high = expected_max_sharpe_under_null(sharpe_std=0.5, n_trials=5000)
    assert sr_high > sr_low > 0


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe_under_null(sharpe_std=0.5, n_trials=1) == 0.0


def test_deflated_sharpe_mostly_low_for_pure_noise():
    """If every variant is genuinely zero-skill Gaussian noise, few (if any)
    should show a high probability of exceeding the noise-adjusted benchmark
    -- that's the entire point of the deflation correction."""
    rng = np.random.default_rng(42)
    n_variants, n_obs = 500, 250
    noise = pd.DataFrame(rng.normal(scale=0.01, size=(n_obs, n_variants)))
    noise.columns = [f"v{i}" for i in range(n_variants)]
    dsr = deflated_sharpe_ratio(noise)
    assert (dsr > 0.95).mean() < 0.05
