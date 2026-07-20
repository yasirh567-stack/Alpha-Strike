"""End-to-end statarb pipeline. Built incrementally -- stages are added as
the corresponding module lands; run with `python main.py`."""
from __future__ import annotations

import logging

from statarb.data import load_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("statarb.main")

START_DATE = "2020-01-01"


def main() -> None:
    logger.info("=== Stage 1: data ===")
    constituents, prices = load_universe(start=START_DATE)
    logger.info("Universe: %d tickers, %d sectors, %d trading days (%s to %s)",
                prices.shape[1], constituents["sector"].nunique(), prices.shape[0],
                prices.index.min().date(), prices.index.max().date())


if __name__ == "__main__":
    main()
