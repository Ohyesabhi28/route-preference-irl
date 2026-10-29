"""
metrics.py – Statistics helpers for per-trip evaluation.

Every metric is first computed per held-out trip (1-D array), then summarised
as mean, std and a percentile-bootstrap confidence interval.  Methods are
compared with paired tests on the same trips.
"""

import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean, ignoring NaNs. Returns (lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def summarize(x, **kw) -> dict:
    """mean / std / sem / 95 % CI / n of a per-trip metric array."""
    x = np.asarray(x, dtype=float)
    v = x[~np.isnan(x)]
    lo, hi = bootstrap_ci(v, **kw)
    return {
        "mean": float(v.mean()) if len(v) else float("nan"),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
        "ci_lo": lo, "ci_hi": hi, "n": int(len(v)),
    }


def paired_wilcoxon(a, b) -> dict:
    """Paired Wilcoxon signed-rank test of a vs b on the same trips."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) < 5 or np.allclose(d, 0):
        return {"mean_diff": float(d.mean()) if len(d) else float("nan"),
                "p_value": float("nan"), "n": int(len(d))}
    res = stats.wilcoxon(d)
    return {"mean_diff": float(d.mean()), "p_value": float(res.pvalue),
            "n": int(len(d))}


def summarize_per_trip(per_trip: dict, reference: str = None) -> dict:
    """
    per_trip: {method: {metric: array}}  ->
              {method: {metric: summary-dict (+ 'vs_ref' paired test)}}
    """
    out = {}
    for method, mets in per_trip.items():
        out[method] = {}
        for m, arr in mets.items():
            s = summarize(arr)
            if reference and method != reference and m in per_trip.get(reference, {}):
                s["vs_" + reference] = paired_wilcoxon(arr, per_trip[reference][m])
            out[method][m] = s
    return out
