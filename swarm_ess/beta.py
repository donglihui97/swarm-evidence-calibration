"""Working Beta reports for a candidate effective sample size."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta as _beta


def beta_params(success, count, ess):
    s = ess * success / count
    return 1.0 + s, 1.0 + ess - s


def report(success, count, ess, target, level=0.95):
    """Return P(f>1/2), central interval, coverage of target, Brier, side."""
    a, b = beta_params(success, count, ess)
    p_pos = float(_beta.sf(0.5, a, b))
    lo, hi = _beta.ppf([(1 - level) / 2, (1 + level) / 2], a, b)
    truth = int(target >= 0.5)
    side = int(a >= b)
    conf = p_pos if side else 1 - p_pos
    return dict(p_pos=p_pos, conf=conf, side=side, correct=int(side == truth),
                lo=float(lo), hi=float(hi), width=float(hi - lo),
                cover=int(lo <= target <= hi), brier=(p_pos - truth) ** 2)
