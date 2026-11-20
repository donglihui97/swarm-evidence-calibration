"""Bounded-mean tests over independent launch groups.

Group means Z_g are independent with common mean f under the null. Each rule
returns directional log-evidence; a side is certified when it reaches
log(2H/alpha) for H declared looks (union bound over sides and looks).
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

LAMBDAS = np.linspace(0.1, 1.9, 19)


def threshold(alpha=0.05, looks=8):
    return float(np.log(2 * looks / alpha))


def mixture_log_e(z):
    """Uniform mixture of 19 fixed bets (betting e-value)."""
    z = np.asarray(z, float)
    out = []
    for sign in (1.0, -1.0):
        logs = np.log1p(sign * LAMBDAS[:, None] * (z[None, :] - 0.5)).sum(1)
        out.append(float(logsumexp(logs) - np.log(len(LAMBDAS))))
    return tuple(out)


def fixed_log_e(z, lam=1.9):
    z = np.asarray(z, float)
    return tuple(float(np.log1p(s * lam * (z - 0.5)).sum()) for s in (1.0, -1.0))


def hoeffding_log_inv_p(z):
    z = np.asarray(z, float)
    d = float(z.mean() - 0.5)
    return 2 * len(z) * max(0.0, d) ** 2, 2 * len(z) * max(0.0, -d) ** 2


RULES = dict(mixture=mixture_log_e, fixed=fixed_log_e, hoeffding=hoeffding_log_inv_p)


def certified_side(scores, thr):
    pos, neg = scores[0] >= thr, scores[1] >= thr
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    return None


def min_groups_constant_vector(d, delta):
    """Smallest G with (1+2d)^G >= 1/delta: necessary for any level-delta test
    to reject the constant vector 1/2+d deterministically."""
    return int(np.ceil(np.log(1 / delta) / np.log1p(2 * d)))


def mixture_min_groups(d, delta, gmax=10000):
    """Smallest G at which the mixture can reach 1/delta at deviation d."""
    thr = np.log(1 / delta)
    for g in range(1, gmax):
        if logsumexp(g * np.log1p(LAMBDAS * d)) - np.log(len(LAMBDAS)) >= thr:
            return g
    return None
