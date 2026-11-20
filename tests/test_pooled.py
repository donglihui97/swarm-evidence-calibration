import numpy as np
from swarm_ess import estimators as est


def test_pooled_local_iid_close_to_n():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (40, 64))
    e = est.pooled_local_ess(y)
    assert np.all(e > 0.8 * 64) and np.all(e <= 64)


def test_pooled_local_constant_patch_is_one_reading():
    y = np.zeros((10, 32)); y[:5] = 1
    e = est.pooled_local_ess(y)
    assert np.allclose(e, 1.0)


def test_pooled_local_mask_equals_truncation():
    rng = np.random.default_rng(3)
    y = (np.cumsum(rng.normal(size=(12, 40)), 1) > 0).astype(float)
    mask = np.zeros_like(y); mask[:, :25] = 1
    assert np.allclose(est.pooled_local_ess(y, mask=mask), est.pooled_local_ess(y[:, :25]))
