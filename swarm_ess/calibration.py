"""Split-conformal calibration of an ESS estimator against the reference.

A score s = log(e_hat / e_ref) is computed on simulated calibration worlds.
The calibrated estimator e_hat * exp(-q) does not exceed the reference
with probability at least 1-beta for a new exchangeable world.
"""
from __future__ import annotations
import math
import numpy as np


def conformal_quantile(scores, beta=0.05, weights=None):
    """ceil((m+1)(1-beta))-th smallest score (+inf if it does not exist).

    Integer weights repeat a score (used by the world bootstrap).
    """
    s = np.asarray(scores, float)
    w = np.ones_like(s, dtype=int) if weights is None else np.asarray(weights, int)
    m = int(w.sum())
    rank = math.ceil((m + 1) * (1 - beta))
    if rank > m:
        return math.inf
    order = np.argsort(s, kind='stable')
    cum = np.cumsum(w[order])
    return float(s[order][np.searchsorted(cum, rank)])


def apply(ess, q, count):
    """Shrink log-ESS by q and clip to [1, count]."""
    if math.isinf(q):
        return np.ones_like(np.asarray(ess, float))
    return np.exp(np.clip(np.log(ess) - q, 0.0, np.log(count)))


def standardise(scores, centre, scale):
    """Prefix-wise standardisation used before the within-world maximum."""
    return (scores - centre) / scale


def fit_standardisation(scores):
    """Median and MAD-type scale per prefix from an independent fold."""
    centre = np.median(scores, axis=0)
    scale = np.median(np.abs(scores - centre), axis=0) * 1.4826
    return centre, np.maximum(scale, 1e-3)
