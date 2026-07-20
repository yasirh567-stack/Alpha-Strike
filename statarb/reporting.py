"""Matplotlib charts + markdown report assembly."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Validated categorical palette + chart chrome, see the project's dataviz
# skill reference (references/palette.md) -- fixed hue order, not re-derived.
LIGHT_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "figure.facecolor": LIGHT_SURFACE,
    "axes.facecolor": LIGHT_SURFACE,
    "savefig.facecolor": LIGHT_SURFACE,
})


def plot_equity_curves(
    variant_returns: pd.DataFrame, dsr: pd.Series, top_n: int = 8, out_path: Path | None = None
) -> Path:
    """Equity curves for the top-`top_n` variants by Deflated Sharpe Ratio.

    Inactive days (variant not in that fold's tradable book) are treated as
    flat (fillna 0), not skipped -- skipping would draw a straight line
    across a gap as if capital compounded through it, which it didn't.
    """
    top_ids = dsr.dropna().sort_values(ascending=False).head(top_n).index
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, vid in enumerate(top_ids):
        r = variant_returns[vid].fillna(0.0)
        equity = (1 + r).cumprod() * 100
        a, b = vid.split("|")[:2]
        ax.plot(equity.index, equity.to_numpy(), color=CATEGORICAL[i % len(CATEGORICAL)],
                 linewidth=1.6, label=f"{a}/{b} (DSR={dsr[vid]:.2f})")
    ax.axhline(100, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_ylabel("Equity (start = 100)")
    ax.set_title(f"Equity curves: top {top_n} variants by Deflated Sharpe Ratio", fontsize=13)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = out_path or REPORTS_DIR / "equity_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sharpe_histograms(
    naive_sharpes: pd.Series,
    dsr: pd.Series,
    null_sharpes: np.ndarray,
    dsr_threshold: float = 0.95,
    out_path: Path | None = None,
) -> Path:
    """Two panels: (1) observed naive Sharpes vs. the permutation-test null
    distribution, same units, directly comparable; (2) the Deflated Sharpe
    Ratio distribution (a probability, different units from panel 1 by
    construction -- DSR corrects for the *number of trials*, which a raw
    Sharpe histogram can't show on its own)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    naive = naive_sharpes.dropna()
    lo = min(naive.min(), null_sharpes.min())
    hi = max(naive.max(), null_sharpes.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(null_sharpes, bins=bins, color=CATEGORICAL[7], alpha=0.6, density=True,
            label=f"Permutation null, pure noise (n={len(null_sharpes)})")
    ax.hist(naive, bins=bins, color=CATEGORICAL[0], alpha=0.55, density=True,
            label=f"Observed naive Sharpe (n={len(naive)})")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Density")
    ax.set_title("Naive Sharpe vs. pure-noise null distribution", fontsize=12)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    dsr_valid = dsr.dropna()
    ax.hist(dsr_valid, bins=40, color=CATEGORICAL[0], alpha=0.85)
    ax.axvline(dsr_threshold, color=CRITICAL, linewidth=1.5, linestyle="--")
    n_survive = int((dsr_valid > dsr_threshold).sum())
    ax.text(dsr_threshold, ax.get_ylim()[1] * 0.92, f"  {n_survive}/{len(dsr_valid)} survive",
            color=CRITICAL, fontsize=9, va="top")
    ax.set_xlabel("Deflated Sharpe Ratio (P[true Sharpe > noise benchmark])")
    ax.set_ylabel("Count")
    ax.set_title("Deflated Sharpe Ratio across all variants", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = out_path or REPORTS_DIR / "sharpe_histograms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def top_strategies_table(
    variant_stats: pd.DataFrame, variant_meta: pd.DataFrame, dsr: pd.Series, top_n: int = 20
) -> pd.DataFrame:
    base = variant_stats.drop(columns=["deflated_sharpe"], errors="ignore")
    merged = base.join(variant_meta).join(dsr.rename("deflated_sharpe"))
    merged = merged.sort_values("sharpe", ascending=False).head(top_n)
    # variant_id (the DataFrame index) is dropped here rather than shown --
    # it's built as "ticker_a|ticker_b|lb..|en..|ex..", and those literal '|'
    # characters would otherwise corrupt the markdown table's column structure.
    cols = ["ticker_a", "ticker_b", "sector", "lookback", "entry", "exit", "sharpe",
            "deflated_sharpe", "max_drawdown", "hit_rate", "n_trading_days", "total_return"]
    out = merged[cols].reset_index(drop=True)
    out.index = range(1, len(out) + 1)
    return out


def _df_to_markdown_table(df: pd.DataFrame, float_cols: dict[str, str]) -> str:
    fmt = df.copy()
    for col, spec in float_cols.items():
        fmt[col] = fmt[col].map(lambda v: format(v, spec) if pd.notna(v) else "n/a")
    header = "| " + " | ".join(["#"] + list(fmt.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(fmt.columns) + 1)) + " |"
    rows = [
        "| " + " | ".join([str(i)] + [str(v) for v in row]) + " |"
        for i, row in zip(fmt.index, fmt.to_numpy())
    ]
    return "\n".join([header, sep, *rows])


def build_markdown_report(
    *,
    universe_summary: dict,
    fold_log: pd.DataFrame,
    variant_stats: pd.DataFrame,
    dsr: pd.Series,
    signal_vs_noise: dict,
    top_table: pd.DataFrame,
    equity_curves_path: Path,
    sharpe_histograms_path: Path,
    out_path: Path | None = None,
) -> Path:
    out_path = out_path or REPORTS_DIR / "report.md"
    n_total = signal_vs_noise["n_total_variants"]
    n_naive = signal_vs_noise["n_naive_good"]
    n_survive = signal_vs_noise["n_survive_deflation"]

    fold_table = fold_log.copy()
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        fold_table[c] = pd.to_datetime(fold_table[c]).dt.date
    fold_md = "\n".join(
        f"| {r.train_start} to {r.train_end} | {r.test_start} to {r.test_end} | {r.n_pairs_selected} | {r.n_test_days} |"
        for r in fold_table.itertuples()
    )

    top_md = _df_to_markdown_table(
        top_table,
        float_cols={
            "sharpe": ".2f", "deflated_sharpe": ".3f", "max_drawdown": ".1%",
            "hit_rate": ".1%", "total_return": ".1%",
        },
    )

    text = f"""# statarb backtest report

Generated by `main.py`. This report exists to answer one question: after
testing {n_total:,} strategy variants, how many of them represent real edge
versus how many just got lucky?

## Universe

- {universe_summary['n_tickers']} tickers, {universe_summary['n_sectors']} GICS sectors
- {universe_summary['start_date']} to {universe_summary['end_date']} ({universe_summary['n_days']} trading days)
- See the README for survivorship-bias and data-quality caveats that apply to every number below.

## Walk-forward folds

2-year rolling train window, 6-month test window, 6-month step. Pairs are
screened for Engle-Granger cointegration (p<0.05) and hedge ratios fixed
*only* on each fold's training window; the test window is never used for
either.

| Train window | Test window | Pairs selected | Test days |
|---|---|---|---|
{fold_md}

Note how much the pair count varies fold to fold ({fold_log['n_pairs_selected'].min()}-{fold_log['n_pairs_selected'].max()}):
cointegration relationships are not regime-stable, and a strategy that
looks great in one window can lose its entire tradable universe in the next.

## Signal vs. noise: the core result

**{n_naive:,} of {n_total:,} variants ({n_naive/n_total:.1%}) look profitable naively**
(annualized Sharpe > {signal_vs_noise['naive_sharpe_threshold']}).
**{n_survive:,} survive** after correcting for the fact that {n_total:,} variants
were tried (Deflated Sharpe Ratio > {signal_vs_noise['dsr_threshold']}).

![Sharpe histograms]({sharpe_histograms_path.name})

Left panel: the observed naive Sharpe distribution against a permutation-test
null built by reshuffling each pair's own daily returns (destroying any real
lead/lag relationship while preserving each leg's volatility) and rerunning
the identical signal logic on the shuffled path -- this is what pure noise
looks like on this exact universe with these exact rules. Right panel: the
Deflated Sharpe Ratio for every variant, which folds in *how many* variants
were tried; almost the entire mass sits near zero.

A subtlety worth stating plainly: a pair-specific permutation test on the
single best variant can still reject the null in isolation (that one pair's
co-movement isn't obviously random). The DSR is more skeptical because it
also accounts for the fact that this "best" variant was cherry-picked out of
{n_total:,} attempts. Both tests are valid; they answer different questions,
and the gap between their answers *is* the multiple-testing problem this
project is built to demonstrate.

## Equity curves

![Equity curves]({equity_curves_path.name})

Top variants by Deflated Sharpe Ratio -- shown for illustration of what the
"best" results look like, not as a claim that they clear the survival bar
above (most don't).

## Top 20 variants by naive Sharpe

{top_md}

`deflated_sharpe` is the number that matters: compare it to the naive
`sharpe` column and note how little of the naive ranking survives.
"""
    out_path.write_text(text)
    return out_path
