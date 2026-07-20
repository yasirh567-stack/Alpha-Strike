"""End-to-end statarb pipeline. Built incrementally -- stages are added as
the corresponding module lands; run with `python main.py`."""
from __future__ import annotations

import logging

from statarb.data import load_universe
from statarb.signals import select_cointegrated_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("statarb.main")

START_DATE = "2020-01-01"
INITIAL_TRAIN_END = "2022-01-01"


def main() -> None:
    logger.info("=== Stage 1: data ===")
    constituents, prices = load_universe(start=START_DATE)
    logger.info("Universe: %d tickers, %d sectors, %d trading days (%s to %s)",
                prices.shape[1], constituents["sector"].nunique(), prices.shape[0],
                prices.index.min().date(), prices.index.max().date())

    logger.info("=== Stage 2: signals / pair screening (illustrative, single window) ===")
    train = prices.loc[START_DATE:INITIAL_TRAIN_END]
    pairs = select_cointegrated_pairs(train, constituents)
    logger.info("Found %d cointegrated pairs (p<0.05) on %s to %s training window",
                len(pairs), train.index.min().date(), train.index.max().date())
    logger.info("Top 5 by p-value:\n%s", pairs.head(5).to_string())


if __name__ == "__main__":
    main()
