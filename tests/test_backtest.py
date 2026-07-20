import numpy as np
import pandas as pd
import pytest

from statarb.backtest import WalkForwardSplitter, generate_positions


@pytest.fixture
def daily_index():
    return pd.date_range("2015-01-01", "2026-01-01", freq="B")


def test_splitter_fold_boundaries(daily_index):
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=6).split(daily_index)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end == fold.test_start
        assert fold.train_start == fold.test_start - pd.DateOffset(years=2)
        assert fold.test_end == fold.test_start + pd.DateOffset(months=6)


def test_splitter_step_size_between_folds(daily_index):
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=6).split(daily_index)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_start == prev.test_start + pd.DateOffset(months=6)


def test_splitter_no_train_test_overlap(daily_index):
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=6).split(daily_index)
    for fold in folds:
        assert fold.train_end <= fold.test_start
        train_range = pd.Interval(fold.train_start, fold.train_end, closed="left")
        test_range = pd.Interval(fold.test_start, fold.test_end, closed="left")
        assert train_range.overlaps(test_range) is False or train_range.right == test_range.left


def test_splitter_folds_stay_within_available_history(daily_index):
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=6).split(daily_index)
    for fold in folds:
        assert fold.train_start >= daily_index.min()
        assert fold.test_end <= daily_index.max()


def test_splitter_smaller_step_produces_overlapping_test_windows(daily_index):
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=3).split(daily_index)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_start == prev.test_start + pd.DateOffset(months=3)
        assert nxt.test_start < prev.test_end  # overlap is expected when step < test length


def test_splitter_empty_when_history_shorter_than_train_plus_test():
    short_index = pd.date_range("2024-01-01", "2024-06-01", freq="B")
    folds = WalkForwardSplitter(train_years=2, test_months=6, step_months=6).split(short_index)
    assert folds == []


def test_generate_positions_entry_and_exit():
    z = np.array([[0.0], [1.6], [1.8], [0.4], [0.0]])
    entry, exit_, stop, time_stop = np.array([1.5]), np.array([0.5]), np.array([3.5]), np.array([30])
    pos = generate_positions(z, entry, exit_, stop, time_stop)
    assert list(pos[:, 0]) == [0, -1, -1, 0, 0]


def test_generate_positions_stop_loss_triggers():
    # bar 2 breaches the stop (|z|>=3.5) and exits; bar 3 is a fresh bar where
    # z=4.2 still clears the entry threshold, so re-entry there is expected --
    # only same-bar re-entry right after an exit is blocked.
    z = np.array([[2.0], [3.0], [4.0], [4.2]])
    entry, exit_, stop, time_stop = np.array([1.5]), np.array([0.5]), np.array([3.5]), np.array([30])
    pos = generate_positions(z, entry, exit_, stop, time_stop)
    assert list(pos[:, 0]) == [-1, -1, 0, -1]


def test_generate_positions_time_stop_triggers():
    # 3-bar time stop exits at bar 3; bar 4 is fresh and z=2.0 still clears
    # entry, so it re-enters immediately (same re-entry rule as the stop-loss case).
    z = np.full((5, 1), 2.0)  # stays entered-but-never-reverting
    entry, exit_, stop, time_stop = np.array([1.5]), np.array([0.5]), np.array([3.5]), np.array([3])
    pos = generate_positions(z, entry, exit_, stop, time_stop)
    assert list(pos[:, 0]) == [-1, -1, -1, 0, -1]


def test_generate_positions_no_reentry_same_bar_as_exit():
    z = np.array([[2.0], [4.0]])  # bar 1: stop-loss triggers exit; z still extreme, must not re-enter same bar
    entry, exit_, stop, time_stop = np.array([1.5]), np.array([0.5]), np.array([3.5]), np.array([30])
    pos = generate_positions(z, entry, exit_, stop, time_stop)
    assert list(pos[:, 0]) == [-1, 0]


def test_generate_positions_nan_forces_flat():
    z = np.array([[np.nan], [2.0], [np.nan]])
    entry, exit_, stop, time_stop = np.array([1.5]), np.array([0.5]), np.array([3.5]), np.array([30])
    pos = generate_positions(z, entry, exit_, stop, time_stop)
    assert list(pos[:, 0]) == [0, -1, 0]
