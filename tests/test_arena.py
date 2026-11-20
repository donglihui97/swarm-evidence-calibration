import numpy as np
from swarm_ess import arena, estimators as est


def test_arena_reference_matches_monte_carlo():
    cfg = arena.ArenaConfig(launch='nest', samples=16)
    run = arena.simulate_arena(11, 'cluster2', 0.3, cfg)
    tiles = np.minimum((run.x_true / cfg.tile).astype(int), 19)
    _, var = arena.arena_reference('cluster2', 0.3, tiles, 20)
    rng = np.random.default_rng(5)
    err = []
    for _ in range(3000):
        fl = arena.make_floor('cluster2', 0.3, 20, rng)
        err.append(fl[tiles[..., 0], tiles[..., 1]].mean() - fl.mean())
    assert abs(var / np.var(err) - 1) < 0.1


def test_random_floor_has_exact_fill():
    fl = arena.make_floor('random', 0.48, 20, np.random.default_rng(3))
    assert fl.sum() == 192


def test_mask_equals_truncation():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, (6, 10)).astype(float)
    x = rng.uniform(0, 128, (6, 10, 2))
    loc = np.full(6, 4.0)
    short = est.spectral_ess(y[:, :7], x[:, :7], 128, loc)
    mask = np.zeros((6, 10)); mask[:, :7] = 1
    masked = est.spectral_ess(y, x, 128, loc, mask=mask)
    assert abs(short['ess'] - masked['ess']) < 1e-8 * short['ess']


def test_walls_and_no_overlap():
    cfg = arena.ArenaConfig(n_robots=50, samples=8)
    run = arena.simulate_arena(4, 'random', 0.3, cfg)
    assert run.x_true.min() >= cfg.radius - 1e-12
    assert run.x_true.max() <= cfg.side - cfg.radius + 1e-12
