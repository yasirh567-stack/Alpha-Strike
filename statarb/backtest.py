"""Walk-forward splitter and vectorized strategy-grid backtest engine.

A "variant" is one (pair, lookback, entry, exit) combination, backtested as
an independent single-pair strategy -- not blended into one portfolio. The
point of this project is to test 1,000+ such hypotheses and see how many
survive correction for multiple testing (see stats.py), so each variant's
daily return series is kept separate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.signals import select_cointegrated_pairs, zscore

LOOKBACKS = (20, 40, 60, 90)
ENTRY_THRESHOLDS = (1.5, 2.0, 2.5)
EXIT_THRESHOLDS = (0.0, 0.5)
STOP_LOSS_Z = 3.5
TIME_STOP_DAYS = 30
COST_BPS_PER_SIDE = 5.0
TARGET_DAILY_VOL = 0.01
MAX_POSITION_SCALE = 5.0
MIN_POSITION_SCALE = 0.2


def strategy_grid() -> pd.DataFrame:
    """The full (lookback x entry x exit) grid; stop-loss and time-stop are
    fixed risk overlays applied to every variant, not swept parameters."""
    rows = [
        {"lookback": lb, "entry": e, "exit": x, "stop": STOP_LOSS_Z, "time_stop": TIME_STOP_DAYS}
        for lb in LOOKBACKS for e in ENTRY_THRESHOLDS for x in EXIT_THRESHOLDS
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """Rolling-origin walk-forward: `train_years` of history, `test_months`
    forward-tested, stepping `step_months` between folds. `train_end` always
    equals `test_start` -- the two windows never overlap."""

    def __init__(self, train_years: int = 2, test_months: int = 6, step_months: int = 6):
        self.train_years = train_years
        self.test_months = test_months
        self.step_months = step_months

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        index = pd.DatetimeIndex(index).sort_values()
        train_len = pd.DateOffset(years=self.train_years)
        test_len = pd.DateOffset(months=self.test_months)
        step = pd.DateOffset(months=self.step_months)

        folds = []
        test_start = index.min() + train_len
        while True:
            test_end = test_start + test_len
            if test_end > index.max():
                break
            folds.append(Fold(test_start - train_len, test_start, test_start, test_end))
            test_start = test_start + step
        return folds


def generate_positions(
    z: np.ndarray, entry: np.ndarray, exit: np.ndarray, stop: np.ndarray, time_stop: np.ndarray
) -> np.ndarray:
    """Vectorized entry/exit/stop-loss/time-stop state machine.

    z: (T, V) per-variant z-score. entry/exit/stop/time_stop: (V,) thresholds,
    one per variant/column. Returns positions (T, V) in {-1, 0, 1}: +1 = long
    the spread (long ticker_a, short beta*ticker_b), -1 = short the spread.

    The state machine is inherently path-dependent (whether you're in a trade
    today depends on whether you were in one yesterday), so the loop over T
    can't be removed -- but every per-step operation is a numpy op across all
    V columns at once, so the loop only runs T times total regardless of how
    many thousands of variants V is, not T*V times.

    A position closed on a given bar cannot re-open in the same bar (avoids
    a stop-loss immediately re-triggering a fresh entry in the same
    direction on the same extreme reading).
    """
    T, V = z.shape
    current = np.zeros(V, dtype=np.int8)
    bars = np.zeros(V, dtype=np.int32)
    out = np.zeros((T, V), dtype=np.int8)
    valid = ~np.isnan(z)

    for t in range(T):
        zt = z[t]
        v = valid[t]
        prev = current

        open_long = prev == 1
        open_short = prev == -1
        exit_band = np.abs(zt) <= exit
        exit_time = bars >= time_stop
        exit_stop = (open_long & (zt <= -stop)) | (open_short & (zt >= stop))
        do_exit = (open_long | open_short) & v & (exit_band | exit_time | exit_stop)
        current = np.where(do_exit, 0, prev)

        flat_before = (prev == 0) & v
        do_long = flat_before & (zt <= -entry)
        do_short = flat_before & (zt >= entry)
        current = np.where(do_long, 1, current)
        current = np.where(do_short, -1, current)
        current = np.where(v, current, 0)

        bars = np.where(current == 0, 0, np.where(prev == current, bars + 1, 1))
        out[t] = current

    return out


@dataclass
class BacktestResult:
    variant_returns: pd.DataFrame  # date x variant_id, net daily return, NaN when not in that fold's book
    variant_positions: pd.DataFrame  # date x variant_id, position in {-1,0,1}
    variant_meta: pd.DataFrame  # variant_id -> ticker_a, ticker_b, sector, lookback, entry, exit
    fold_log: pd.DataFrame  # per-fold summary: dates, n_pairs screened/selected, n_variants


def run_walkforward_backtest(
    prices: pd.DataFrame,
    constituents: pd.DataFrame,
    splitter: WalkForwardSplitter | None = None,
    grid: pd.DataFrame | None = None,
    cost_bps_per_side: float = COST_BPS_PER_SIDE,
    pvalue_threshold: float = 0.05,
) -> BacktestResult:
    """Run the full walk-forward, strategy-grid backtest.

    For each fold: screen cointegrated pairs on the training window only
    (statarb.signals.select_cointegrated_pairs), fix each pair's hedge ratio
    from that same training window, then backtest the full parameter grid
    for every selected pair over the (out-of-sample) test window. Position
    sizing is vol-targeted per pair using training-window spread volatility
    ("equal risk per pair": a jumpy pair gets a smaller position than a calm
    one so no single pair dominates variance). Transaction costs are charged
    on both legs whenever a variant's position size changes.
    """
    splitter = splitter or WalkForwardSplitter()
    grid = grid if grid is not None else strategy_grid()
    n_variants_per_pair = len(grid)
    entry_g = grid["entry"].to_numpy()
    exit_g = grid["exit"].to_numpy()
    stop_g = grid["stop"].to_numpy()
    time_stop_g = grid["time_stop"].to_numpy()

    folds = splitter.split(prices.index)
    log_prices = np.log(prices)

    returns_by_variant: dict[str, list[pd.Series]] = {}
    positions_by_variant: dict[str, list[pd.Series]] = {}
    meta_by_variant: dict[str, dict] = {}
    fold_rows = []

    for fold in folds:
        train = prices.loc[fold.train_start : fold.train_end]
        pairs = select_cointegrated_pairs(train, constituents, pvalue_threshold=pvalue_threshold)

        test_mask = (prices.index >= fold.test_start) & (prices.index < fold.test_end)
        test_dates = prices.index[test_mask]
        fold_rows.append({
            "train_start": fold.train_start, "train_end": fold.train_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "n_pairs_selected": len(pairs), "n_test_days": len(test_dates),
        })
        if pairs.empty or len(test_dates) < 2:
            continue

        full = log_prices.loc[fold.train_start : fold.test_end]
        n_pairs = len(pairs)
        V = n_pairs * n_variants_per_pair

        z_cols = np.empty((len(test_dates), V), dtype=float)
        ret_matrix = np.empty((len(test_dates), V), dtype=float)
        entry_arr = np.tile(entry_g, n_pairs)
        exit_arr = np.tile(exit_g, n_pairs)
        stop_arr = np.tile(stop_g, n_pairs)
        time_stop_arr = np.tile(time_stop_g, n_pairs)
        variant_ids: list[str] = []

        for i, row in enumerate(pairs.itertuples()):
            a, b, alpha, beta = row.ticker_a, row.ticker_b, row.alpha, row.beta
            full_spread = full[a] - beta * full[b] - alpha
            train_vol = full_spread.loc[fold.train_start : fold.train_end].diff().std()
            size_mult = np.clip(TARGET_DAILY_VOL / (train_vol + 1e-12), MIN_POSITION_SCALE, MAX_POSITION_SCALE)

            pair_ret = (full[a].diff() - beta * full[b].diff()).reindex(test_dates).fillna(0.0).to_numpy()
            zs = {lb: zscore(full_spread, lb).reindex(test_dates).to_numpy() for lb in LOOKBACKS}

            base = i * n_variants_per_pair
            for j, grow in enumerate(grid.itertuples()):
                col = base + j
                z_cols[:, col] = zs[grow.lookback]
                ret_matrix[:, col] = pair_ret * size_mult
                vid = f"{a}|{b}|lb{grow.lookback}|en{grow.entry}|ex{grow.exit}"
                variant_ids.append(vid)
                meta_by_variant.setdefault(vid, {
                    "ticker_a": a, "ticker_b": b, "sector": row.sector,
                    "lookback": grow.lookback, "entry": grow.entry, "exit": grow.exit,
                })

        positions = generate_positions(z_cols, entry_arr, exit_arr, stop_arr, time_stop_arr).astype(float)
        shifted = np.vstack([np.zeros((1, V)), positions[:-1]])
        gross_pnl = shifted * ret_matrix
        turnover = np.abs(np.diff(positions, axis=0, prepend=0))
        cost = turnover * (cost_bps_per_side / 10_000.0) * 2
        net_pnl = gross_pnl - cost

        for col, vid in enumerate(variant_ids):
            returns_by_variant.setdefault(vid, []).append(pd.Series(net_pnl[:, col], index=test_dates))
            positions_by_variant.setdefault(vid, []).append(pd.Series(positions[:, col], index=test_dates))

    variant_returns = pd.DataFrame({vid: pd.concat(s) for vid, s in returns_by_variant.items()}).sort_index()
    variant_positions = pd.DataFrame({vid: pd.concat(s) for vid, s in positions_by_variant.items()}).sort_index()
    variant_meta = pd.DataFrame(
        [{"variant_id": vid, **meta} for vid, meta in meta_by_variant.items()]
    ).set_index("variant_id")
    fold_log = pd.DataFrame(fold_rows)

    return BacktestResult(variant_returns, variant_positions, variant_meta, fold_log)


def build_trade_log(positions: pd.Series, returns: pd.Series, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """Reconstruct discrete trades (entry/exit/pnl) from a single variant's
    position and return series -- used for reporting on a handful of
    representative strategies, not for all 1,000+ variants at once."""
    pos = positions.dropna()
    rets = returns.reindex(pos.index).fillna(0.0)
    trades = []
    entry_date, direction, trade_pnl = None, 0, 0.0
    for date, p, r in zip(pos.index, pos.to_numpy(), rets.to_numpy()):
        if entry_date is None and p != 0:
            entry_date, direction, trade_pnl = date, int(np.sign(p)), r
        elif entry_date is not None and p != 0:
            trade_pnl += r
        elif entry_date is not None and p == 0:
            trades.append({
                "ticker_a": ticker_a, "ticker_b": ticker_b,
                "direction": "long_spread" if direction == 1 else "short_spread",
                "entry_date": entry_date, "exit_date": date,
                "holding_days": (date - entry_date).days, "pnl": trade_pnl,
            })
            entry_date, direction, trade_pnl = None, 0, 0.0
    return pd.DataFrame(trades)
