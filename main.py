"""End-to-end statarb pipeline: data -> walk-forward backtest -> stats -> report.

Run with `python main.py`. See README.md for the method and data caveats.
"""
from __future__ import annotations

import logging
import time

from statarb.backtest import WalkForwardSplitter, run_walkforward_backtest, strategy_grid
from statarb.data import load_universe
from statarb.reporting import (
    REPORTS_DIR,
    build_markdown_report,
    plot_equity_curves,
    plot_sharpe_histograms,
    top_strategies_table,
)
from statarb.stats import (
    deflated_sharpe_ratio,
    pooled_permutation_null,
    signal_vs_noise_summary,
    summarize_all,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("statarb.main")

START_DATE = "2020-01-01"
NAIVE_SHARPE_THRESHOLD = 1.0
DSR_THRESHOLD = 0.95


def main() -> None:
    t0 = time.time()

    logger.info("=== Stage 1: data ===")
    constituents, prices = load_universe(start=START_DATE)
    logger.info("Universe: %d tickers, %d sectors, %d trading days (%s to %s)",
                prices.shape[1], constituents["sector"].nunique(), prices.shape[0],
                prices.index.min().date(), prices.index.max().date())

    logger.info("=== Stage 2: walk-forward backtest (strategy grid) ===")
    splitter = WalkForwardSplitter(train_years=2, test_months=6, step_months=6)
    result = run_walkforward_backtest(prices, constituents, splitter=splitter, grid=strategy_grid())
    logger.info("%d folds, %d distinct strategy variants backtested",
                len(result.fold_log), result.variant_returns.shape[1])

    logger.info("=== Stage 3: stats (Sharpe, Deflated Sharpe Ratio, permutation test) ===")
    variant_stats = summarize_all(result.variant_returns)
    dsr = deflated_sharpe_ratio(result.variant_returns)
    null_sharpes = pooled_permutation_null(result.variant_meta, prices, sample_size=30, n_permutations=200)
    svn = signal_vs_noise_summary(variant_stats, dsr, NAIVE_SHARPE_THRESHOLD, DSR_THRESHOLD)
    logger.info(
        "Naive good (Sharpe>%.1f): %d/%d. Survive deflation (DSR>%.2f): %d/%d",
        NAIVE_SHARPE_THRESHOLD, svn["n_naive_good"], svn["n_total_variants"],
        DSR_THRESHOLD, svn["n_survive_deflation"], svn["n_total_variants"],
    )

    logger.info("=== Stage 4: reporting ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    eq_path = plot_equity_curves(result.variant_returns, dsr)
    hist_path = plot_sharpe_histograms(variant_stats["sharpe"], dsr, null_sharpes, dsr_threshold=DSR_THRESHOLD)
    top_table = top_strategies_table(variant_stats, result.variant_meta, dsr, top_n=20)
    universe_summary = {
        "n_tickers": prices.shape[1], "n_sectors": constituents["sector"].nunique(),
        "start_date": prices.index.min().date(), "end_date": prices.index.max().date(),
        "n_days": prices.shape[0],
    }
    report_path = build_markdown_report(
        universe_summary=universe_summary, fold_log=result.fold_log, variant_stats=variant_stats, dsr=dsr,
        signal_vs_noise=svn, top_table=top_table, equity_curves_path=eq_path, sharpe_histograms_path=hist_path,
    )
    logger.info("Report written to %s", report_path)
    logger.info("Full pipeline finished in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
