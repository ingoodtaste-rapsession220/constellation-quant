"""SPY 2022-2024 buy-and-hold baseline — what 'doing nothing' returned.

Sets the floor that any active strategy must beat. Computed with the same
risk-free assumption as our backtester (rf = 0) so the Sharpe comparison
is apples to apples.

    python scripts/spy_baseline.py

Optionally:
    --start 2022-01-01 --end 2024-12-31
    --csv-out logs/phase_metrics.csv  --phase-tag spy_baseline
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start",      type=str, default="2022-01-01")
    p.add_argument("--end",        type=str, default="2024-12-31")
    p.add_argument("--ticker",     type=str, default="SPY")
    p.add_argument("--csv-out",    type=Path, default=None)
    p.add_argument("--phase-tag",  type=str, default="spy_baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. pip install yfinance", file=sys.stderr)
        return 2

    hist = yf.Ticker(args.ticker).history(
        start=args.start, end=args.end, auto_adjust=False, actions=False,
    )
    if hist.empty:
        print(f"ERROR: no data for {args.ticker} in {args.start} → {args.end}",
              file=sys.stderr)
        return 1

    px = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
    log_ret = np.log(px / px.shift(1)).dropna()
    n = log_ret.size

    daily_mean = float(log_ret.mean())
    daily_std  = float(log_ret.std(ddof=1))
    sharpe     = daily_mean / daily_std * float(np.sqrt(252)) if daily_std > 0 else float("nan")
    annual_log = daily_mean * 252
    annual_ret = float(np.exp(annual_log) - 1)
    cum        = float(np.exp(log_ret.sum()) - 1)
    equity     = px / float(px.iloc[0])
    max_dd     = float((equity / equity.cummax() - 1).min())
    hit_rate   = float((log_ret > 0).mean())

    print("=" * 64)
    print(f"{args.ticker} BUY-AND-HOLD BASELINE  ({args.start} → {args.end})")
    print("=" * 64)
    print(f"trading days:    {n}")
    print(f"annual Sharpe:   {sharpe:+.3f}")
    print(f"annual return:   {annual_ret:+.2%}")
    print(f"cumulative ret:  {cum:+.2%}")
    print(f"max drawdown:    {max_dd:+.2%}")
    print(f"hit rate (>0):   {hit_rate:.3f}")
    print(f"daily vol:       {daily_std:.4f}")

    if args.csv_out is not None:
        import csv
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "timestamp", "phase_tag", "checkpoint", "period_start", "period_end",
            "n_dates", "mean_ic", "median_ic", "std_ic", "t_stat",
            "frac_pos", "frac_neg", "avg_range_over_std", "verdict",
        ]
        # Reuse the diagnose_test_ic schema so a single CSV holds both
        # IC diagnostics and the SPY baseline. IC fields are NaN for SPY.
        row = [
            datetime.now().isoformat(timespec="seconds"),
            args.phase_tag,
            f"{args.ticker}_buy_and_hold",
            args.start, args.end,
            n,
            "nan", "nan", "nan", "nan",
            f"{hit_rate:.3f}", f"{1 - hit_rate:.3f}",
            "nan",
            f"verdict: BASELINE  Sharpe={sharpe:+.2f}  annret={annual_ret:+.1%}  maxdd={max_dd:+.1%}",
        ]
        existed = args.csv_out.exists()
        with args.csv_out.open("a", newline="") as f:
            writer = csv.writer(f)
            if not existed:
                writer.writerow(header)
            writer.writerow(row)
        print(f"\nAppended baseline row to {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
