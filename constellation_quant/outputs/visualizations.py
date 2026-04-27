"""Plot generators for the ablation report.

All functions return either a matplotlib Figure or a filesystem path to a
saved PNG; callers decide whether to embed as base64 in HTML or reference by
path. Designed to work headlessly (no display, Agg backend).

Charts produced:
    - ablation_bar_chart(df, metric) — one bar per variant, metric on y-axis
    - equity_curves_overlay(results) — variants on the same axes
    - drawdown_overlay(results)
    - rolling_ic_timeseries(daily_ic)
    - monthly_return_heatmap(daily_returns)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib
matplotlib.use("Agg")            # headless-safe (no $DISPLAY required)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402


# ── Style ──────────────────────────────────────────────────────────────────


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize":   (10, 5),
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "font.size":        11,
    })


# ── Individual charts ──────────────────────────────────────────────────────


def ablation_bar_chart(
    df: pd.DataFrame,
    metric: str = "sharpe",
    sort: bool = True,
    title: Optional[str] = None,
    save_to: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of one metric across variants.

    Args:
        df: Row per variant with columns {variant, <metric>, ...}.
        metric: Column to plot.
        sort: Sort variants by metric descending.
        title: Optional title override.
        save_to: If given, save as PNG and return the path's parent Figure.
    """
    _apply_style()
    view = df.copy()
    if sort:
        view = view.sort_values(metric, ascending=False)

    fig, ax = plt.subplots()
    ax.bar(view["variant"], view[metric], edgecolor="black", alpha=0.85)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Variant comparison: {metric}")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def equity_curves_overlay(
    results: Mapping[str, pd.Series],
    title: str = "Equity curves",
    save_to: Optional[Path] = None,
) -> plt.Figure:
    """Overlay multiple equity curves on one axis.

    `results` is `{variant_name: equity_series}` where each series is indexed
    by date with the portfolio's running equity.
    """
    _apply_style()
    fig, ax = plt.subplots()
    for name, series in results.items():
        ax.plot(series.index, series.values, label=name, linewidth=1.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.4)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def drawdown_overlay(
    drawdowns: Mapping[str, pd.Series],
    title: str = "Drawdowns",
    save_to: Optional[Path] = None,
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots()
    for name, series in drawdowns.items():
        ax.fill_between(series.index, series.values, 0.0, alpha=0.25, label=name)
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def rolling_ic_timeseries(
    daily_ic: pd.Series,
    window: int = 20,
    title: str = "Rolling IC",
    save_to: Optional[Path] = None,
) -> plt.Figure:
    _apply_style()
    rolling = daily_ic.rolling(window, min_periods=window // 2).mean()
    fig, ax = plt.subplots()
    ax.plot(daily_ic.index, daily_ic.values, color="grey", alpha=0.35, label="daily IC")
    ax.plot(rolling.index, rolling.values, color="tab:blue", linewidth=1.5,
             label=f"{window}d rolling")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("IC")
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def monthly_return_heatmap(
    daily_returns: pd.Series,
    title: str = "Monthly returns",
    save_to: Optional[Path] = None,
) -> plt.Figure:
    _apply_style()
    monthly = (1.0 + daily_returns).resample("ME").prod() - 1.0
    grid = monthly.groupby([monthly.index.year, monthly.index.month]).mean().unstack(fill_value=np.nan)
    grid.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: grid.shape[1]]

    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(grid))))
    im = ax.imshow(grid.values, aspect="auto", cmap="RdYlGn",
                    vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=0)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="return")
    ax.set_title(title)
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig
