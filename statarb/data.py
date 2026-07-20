"""S&P 500 universe + price data: Wikipedia scrape, yfinance download, parquet cache."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
PRICE_CACHE_DIR = CACHE_DIR / "prices"

# Reduced emergency fallback used only if the live Wikipedia scrape fails.
# NOT a substitute for the full ~500-name index -- large, long-listed names
# chosen because they are extremely unlikely to have changed ticker/sector
# since. Spans all 11 GICS sectors so sector-pair selection still has
# something to work with.
FALLBACK_SP500 = [
    ("AAPL", "Information Technology"), ("MSFT", "Information Technology"),
    ("NVDA", "Information Technology"), ("AVGO", "Information Technology"),
    ("ORCL", "Information Technology"), ("CSCO", "Information Technology"),
    ("IBM", "Information Technology"), ("TXN", "Information Technology"),
    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("MCD", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"), ("LOW", "Consumer Discretionary"),
    ("GOOGL", "Communication Services"), ("META", "Communication Services"),
    ("NFLX", "Communication Services"), ("DIS", "Communication Services"),
    ("CMCSA", "Communication Services"), ("T", "Communication Services"),
    ("JPM", "Financials"), ("BAC", "Financials"), ("WFC", "Financials"),
    ("GS", "Financials"), ("MS", "Financials"), ("C", "Financials"),
    ("AXP", "Financials"), ("BLK", "Financials"),
    ("JNJ", "Health Care"), ("UNH", "Health Care"), ("PFE", "Health Care"),
    ("MRK", "Health Care"), ("ABBV", "Health Care"), ("LLY", "Health Care"),
    ("TMO", "Health Care"), ("ABT", "Health Care"),
    ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"), ("SLB", "Energy"),
    ("PG", "Consumer Staples"), ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),
    ("WMT", "Consumer Staples"), ("COST", "Consumer Staples"), ("CL", "Consumer Staples"),
    ("HON", "Industrials"), ("UPS", "Industrials"), ("CAT", "Industrials"),
    ("BA", "Industrials"), ("GE", "Industrials"), ("LMT", "Industrials"),
    ("LIN", "Materials"), ("APD", "Materials"), ("SHW", "Materials"), ("NEM", "Materials"),
    ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"), ("D", "Utilities"),
    ("PLD", "Real Estate"), ("AMT", "Real Estate"), ("EQIX", "Real Estate"), ("SPG", "Real Estate"),
]


def _yf_symbol(ticker: str) -> str:
    """Wikipedia uses '.' for share classes (BRK.B); yfinance wants '-' (BRK-B)."""
    return ticker.strip().replace(".", "-")


def get_sp500_constituents(min_expected: int = 400) -> pd.DataFrame:
    """Return DataFrame[ticker, sector] for current S&P 500 membership.

    Scrapes Wikipedia's maintained constituent table; falls back to a reduced
    hardcoded blue-chip list (see FALLBACK_SP500) if the scrape fails or the
    page format changed enough that the result looks wrong.
    """
    try:
        # Wikipedia 403s on the default urllib user-agent pandas uses internally.
        resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        table = tables[0]
        df = table[["Symbol", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "GICS Sector": "sector"}
        )
        df["ticker"] = df["ticker"].map(_yf_symbol)
        df = df.dropna().drop_duplicates("ticker").reset_index(drop=True)
        if len(df) < min_expected:
            raise ValueError(f"Wikipedia table only had {len(df)} rows, expected ~500")
        logger.info("Loaded %d S&P 500 constituents from Wikipedia", len(df))
        return df
    except Exception as exc:
        logger.warning("Wikipedia scrape failed (%s); using reduced fallback list", exc)
        return pd.DataFrame(FALLBACK_SP500, columns=["ticker", "sector"])


def _fetch_from_cache(ticker: str, start: str, end: str, cache_dir: Path) -> pd.Series | None:
    fp = cache_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None
    s = pd.read_parquet(fp)["close"]
    # Allow slack for `start`/`end` landing on weekends/holidays with no trading day.
    starts_early_enough = s.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
    ends_late_enough = s.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)
    if starts_early_enough and ends_late_enough:
        return s
    return None


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: Path = PRICE_CACHE_DIR,
    max_missing_frac: float = 0.05,
) -> pd.DataFrame:
    """Download daily adjusted close for `tickers`, caching per-ticker to parquet.

    Returns a wide DataFrame (date index, ticker columns). Tickers with no
    data at all (delisted, IPO'd after `start`, bad symbol) or with more than
    `max_missing_frac` missing observations are dropped and logged rather than
    silently corrupting the panel. Small gaps (<=5 sessions, e.g. mismatched
    holiday calendars) are forward-filled.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    cached: dict[str, pd.Series] = {}
    to_fetch: list[str] = []
    for t in tickers:
        s = _fetch_from_cache(t, start, end, cache_dir)
        if s is not None:
            cached[t] = s
        else:
            to_fetch.append(t)

    no_data: list[str] = []
    if to_fetch:
        logger.info("Downloading %d/%d tickers from yfinance", len(to_fetch), len(tickers))
        raw = yf.download(
            to_fetch, start=start, end=end, auto_adjust=True,
            group_by="ticker", progress=False, threads=True,
        )
        for t in to_fetch:
            try:
                col = raw["Close"] if len(to_fetch) == 1 else raw[t]["Close"]
                col = col.dropna()
            except (KeyError, TypeError):
                col = pd.Series(dtype=float)
            if col.empty:
                no_data.append(t)
                continue
            col = col.rename("close")
            col.to_frame().to_parquet(cache_dir / f"{t}.parquet")
            cached[t] = col

    if no_data:
        logger.warning("No usable data for %d/%d tickers (delisted/renamed/no history): %s",
                        len(no_data), len(tickers), no_data)

    panel = pd.DataFrame(cached).sort_index()
    frac_missing = panel.isna().mean()
    keep = frac_missing[frac_missing <= max_missing_frac].index
    dropped = sorted(set(panel.columns) - set(keep))
    if dropped:
        logger.warning("Dropping %d/%d tickers with >%.0f%% missing data in-window: %s",
                        len(dropped), panel.shape[1], max_missing_frac * 100, dropped)
    panel = panel[sorted(keep)].ffill(limit=5)
    logger.info("Final price panel: %d tickers x %d trading days (%s to %s)",
                panel.shape[1], panel.shape[0], panel.index.min().date(), panel.index.max().date())
    return panel


def load_universe(start: str, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entrypoint: constituents + price panel, aligned to each other."""
    constituents = get_sp500_constituents()
    prices = download_prices(constituents["ticker"].tolist(), start, end)
    constituents = constituents[constituents["ticker"].isin(prices.columns)].reset_index(drop=True)
    return constituents, prices
