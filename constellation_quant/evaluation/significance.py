"""Statistical significance tests for ablation comparisons.

Three tests land here — each returns a dataclass with the statistic,
p-value, and effect size where meaningful. Use these to decide whether an
ablation variant's IC / Sharpe delta is real or noise.

* `paired_t_test_ic(daily_ic_a, daily_ic_b)` — canonical comparison of two
  models' daily ICs on the same dates. Null: mean_A == mean_B.

* `diebold_mariano(errors_a, errors_b, h=1)` — Diebold-Mariano test of equal
  predictive accuracy. Less parametric than a t-test; robust to
  autocorrelated errors via Newey-West variance.

* `bootstrap_sharpe_diff(returns_a, returns_b, n_bootstrap=1000)` —
  non-parametric CI for the Sharpe difference between two strategies. Uses
  paired block-bootstrap resampling to respect the time dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


TRADING_DAYS_PER_YEAR = 252


@dataclass
class SignificanceResult:
    """Generic significance test result."""
    name:       str
    statistic:  float
    pvalue:     float
    effect:     float                    # raw effect size (e.g. mean diff)
    n:          int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


@dataclass
class BootstrapResult:
    statistic:   float
    ci_low:      float
    ci_high:     float
    alpha:       float

    def significant(self) -> bool:
        """CI excludes zero → the observed difference is unlikely to be noise."""
        return self.ci_low > 0 or self.ci_high < 0


# ── Paired t-test on daily IC ──────────────────────────────────────────────


def paired_t_test_ic(
    daily_ic_a: np.ndarray,
    daily_ic_b: np.ndarray,
) -> SignificanceResult:
    """Paired t-test on day-by-day IC series. NaN-safe."""
    a = np.asarray(daily_ic_a, dtype=np.float64)
    b = np.asarray(daily_ic_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = a.size
    if n < 2:
        return SignificanceResult("paired_t_ic", float("nan"), float("nan"),
                                     float("nan"), n)

    diff = a - b
    mean = diff.mean()
    # Sample std with ddof=1 (unbiased).
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return SignificanceResult("paired_t_ic", float("inf"), 0.0, float(mean), n)

    t_stat = mean / (sd / np.sqrt(n))
    pvalue = _two_sided_t_pvalue(t_stat, df=n - 1)
    return SignificanceResult("paired_t_ic", float(t_stat),
                                 float(pvalue), float(mean), n)


# ── Diebold-Mariano ────────────────────────────────────────────────────────


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> SignificanceResult:
    """Diebold-Mariano test (1995) with Newey-West variance.

    Args:
        errors_a, errors_b: Forecast errors from model A and B on the same
            dates. Typically `prediction - actual`.
        h: Forecast horizon. Used for the Newey-West lag truncation (h-1).
        power: Exponent in the loss function L(e) = |e|^power. Default 2
            (squared loss).
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = a.size
    if n < max(2, h):
        return SignificanceResult("diebold_mariano", float("nan"),
                                     float("nan"), float("nan"), n)

    loss_a = np.abs(a) ** power
    loss_b = np.abs(b) ** power
    d = loss_a - loss_b
    mean_d = d.mean()

    # Newey-West variance with `h-1` lags.
    lags = max(h - 1, 0)
    variance = _newey_west_variance(d, lags)
    if variance <= 0:
        return SignificanceResult("diebold_mariano", float("nan"),
                                     float("nan"), float(mean_d), n)

    dm_stat = mean_d / np.sqrt(variance / n)
    pvalue = _two_sided_normal_pvalue(dm_stat)
    return SignificanceResult("diebold_mariano", float(dm_stat),
                                 float(pvalue), float(mean_d), n)


# ── Bootstrap Sharpe difference ────────────────────────────────────────────


def bootstrap_sharpe_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> BootstrapResult:
    """Paired block-bootstrap CI for the difference in annualised Sharpe.

    Both series must be aligned on the same dates. Uses a moving-block
    resample (block_size consecutive days) to preserve short-term serial
    correlation.
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = a.size

    observed = _annualised_sharpe(a) - _annualised_sharpe(b)
    if n <= block_size:
        return BootstrapResult(float(observed), float("nan"), float("nan"), alpha)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    starts = rng.integers(0, n - block_size + 1, size=(n_bootstrap, n_blocks))
    for b_i in range(n_bootstrap):
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts[b_i]])[:n]
        diffs[b_i] = _annualised_sharpe(a[idx]) - _annualised_sharpe(b[idx])
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return BootstrapResult(float(observed), lo, hi, alpha)


# ── Low-level helpers ──────────────────────────────────────────────────────


def _annualised_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return float("nan")
    sd = returns.std(ddof=0)
    if sd < 1e-12:
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def _newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance estimator with Bartlett kernel."""
    n = x.size
    xc = x - x.mean()
    gamma0 = float((xc ** 2).mean())
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gamma_k = float((xc[k:] * xc[:-k]).mean())
        var += 2.0 * w * gamma_k
    return max(var, 0.0)


def _two_sided_t_pvalue(t: float, df: int) -> float:
    """Two-sided p-value for Student-t. Uses scipy when available, else a normal approx."""
    try:
        from scipy.stats import t as t_dist
        return 2.0 * float(t_dist.sf(abs(t), df=df))
    except ImportError:
        return _two_sided_normal_pvalue(t)


def _two_sided_normal_pvalue(z: float) -> float:
    """Two-sided p-value for a standard normal (fallback when scipy is absent)."""
    try:
        from scipy.stats import norm
        return 2.0 * float(norm.sf(abs(z)))
    except ImportError:
        # Abramowitz & Stegun 7.1.26 CDF approximation.
        import math
        x = abs(z) / math.sqrt(2.0)
        t = 1.0 / (1.0 + 0.3275911 * x)
        a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
        erf = 1.0 - ((((a[4]*t + a[3])*t + a[2])*t + a[1])*t + a[0]) * t * math.exp(-x*x)
        cdf = 0.5 * (1.0 + erf)
        return 2.0 * (1.0 - cdf)
