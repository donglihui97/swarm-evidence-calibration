import sys, pathlib
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarm_ess import simulator, estimators as est, calibration, stopping


def brute(y, x, size, M, ell, centered):
    yy = y.ravel().astype(float)
    pts = (np.floor(x.reshape(-1, 2)) % size) + 0.5
    rid = np.repeat(np.arange(y.shape[0]), y.shape[1])
    p = yy.mean(); v = p * (1 - p); r = yy - p
    w = est.kernel_weights(size, M, [ell])[0]
    ax = np.arange(-M, M + 1)
    d = pts[:, None, :] - pts[None, :, :]
    K = np.zeros(d.shape[:2])
    for a in range(len(ax)):
        for b in range(len(ax)):
            if w[a, b]:
                K += w[a, b] * np.cos(2 * np.pi * (ax[a] * d[..., 0] + ax[b] * d[..., 1]) / size)
    Km = K - K.mean(1)[:, None] - K.mean(0)[None, :] + K.mean() if centered else K
    off = rid[:, None] != rid[None, :]
    return (np.sum((np.outer(r, r) / v * Km)[off]), np.sum((Km ** 2)[off]), np.sum(K[off]))


@pytest.mark.parametrize('centered', [True, False])
def test_spectral_sums_are_exact(centered):
    rng = np.random.default_rng(7)
    size, M = 32, 3
    y = rng.integers(0, 2, (5, 6)).astype(np.int8)
    x = rng.uniform(0, size, (5, 6, 2))
    lengths = est.lengths_grid(size)
    old = est.MAX_MODE
    res = est.spectral_ess(y, x, size, np.ones(5) * 6, max_mode=M, centered=centered,
                           return_details=True)
    for j in (0, 7, 15):
        A, B, _ = brute(y, x, size, M, lengths[j], centered)
        assert np.isclose(res['A'][j], A, rtol=1e-9, atol=1e-9)
        assert np.isclose(res['B'][j], B, rtol=1e-9, atol=1e-9)


def test_cross_sum_matches_bruteforce():
    rng = np.random.default_rng(8)
    size, M = 32, 3
    y = rng.integers(0, 2, (4, 7)).astype(np.int8)
    x = rng.uniform(0, size, (4, 7, 2))
    res = est.spectral_ess(y, x, size, np.ones(4) * 7, max_mode=M, return_details=True)
    _, _, C = brute(y, x, size, M, res['length'], True)
    assert np.isclose(res['cross'], res['amplitude'] * C, rtol=1e-9, atol=1e-9)



def test_reference_noise_reduces_dependence():
    w = simulator.simulate(3, 16.0, .48, 1)
    e0, _ = est.reference_ess(w.field, w.x)
    e1, _ = est.reference_ess(w.field, w.x, flip_prob=.1)
    assert e1 > e0


def test_conformal_quantile_rank_and_weights():
    s = np.arange(1, 21, dtype=float)
    assert calibration.conformal_quantile(s, .05) == 20.0
    assert np.isinf(calibration.conformal_quantile(s[:10], .05))
    assert calibration.conformal_quantile([1., 2.], .05, weights=[10, 10]) == 2.0


def test_minimum_groups():
    assert stopping.min_groups_constant_vector(.2, 1 / 320) == 18
    assert stopping.mixture_min_groups(.2, 1 / 320) >= 18
