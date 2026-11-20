"""Bounded-arena swarm with disc robots, collisions, local radio and odometry.

This second simulator departs from the torus model of ``simulator.py`` in the
ways a physical collective-perception experiment does: a walled square arena
tiled with black and white floor tiles, differential-drive robots of finite
size that avoid walls and each other and are pushed apart on contact, a
run-and-turn random walk, launches from a nest, single-hop broadcasts with
range limits and packet loss, and positions taken from exact tracking, noisy
radio localisation or dead reckoning. Operating estimators receive the
reported positions; only ``arena_reference`` reads the floor.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
import numpy as np
from scipy import integrate
from scipy.stats import norm


@dataclass(frozen=True)
class ArenaConfig:
    side: float = 2.0            # arena side (m)
    tile: float = 0.1            # floor tile side (m)
    n_robots: int = 20
    radius: float = 0.035        # robot radius (m), e-puck sized
    speed: float = 0.10          # forward speed (m/s)
    omega: float = np.pi / 2     # turning rate in place (rad/s)
    dt: float = 0.1              # integration step (s)
    period: float = 1.0          # ground-sensor sampling period (s)
    samples: int = 64
    mean_run: float = 10.0       # mean straight-run duration (s)
    sense: float = 0.06          # proximity range beyond the body (m)
    comm_range: float = 0.5      # radio range (m)
    loss: float = 0.2            # packet-loss probability
    launch: str = 'uniform'      # 'uniform' or 'nest'
    nest: float = 0.5            # nest side in the lower-left corner (m)
    position: str = 'exact'      # 'exact', 'radio' or 'odometry'
    radio_sd: float = 0.05       # radio-localisation noise per axis (m)
    odo_scale_sd: float = 0.02   # per-robot wheel-scale bias (relative)
    odo_drift_sd: float = 0.005  # per-robot heading drift (rad/s)
    odo_step_sd: float = 0.01    # heading noise per step (rad)
    flip: float = 0.0            # ground-sensor flip probability


# ------------------------------------------------------------------ floors
def _latent_corr(period, ell):
    """Exact correlation of Gaussian-smoothed white noise on a periodic lattice."""
    f = np.fft.fftfreq(period)
    fx, fy = np.meshgrid(f, f, indexing='ij')
    s = ell / np.sqrt(2.0)
    filt = np.exp(-0.5 * s ** 2 * (2 * np.pi) ** 2 * (fx * fx + fy * fy))
    c = np.fft.ifft2(filt ** 2).real
    return filt, c / c[0, 0]


LATENT_PERIOD = 64


def make_floor(kind, fill, n_tiles, rng):
    """Binary tile colours (1 = black). Returns an n_tiles x n_tiles array.

    'random': exactly round(fill*n^2) black tiles in random places (the
    standard collective-perception arena). 'clusterL': a Gaussian latent field
    with correlation exp(-d^2/(2L^2)) in tile units, thresholded at the
    population quantile so that each tile is black with probability fill.
    """
    if kind == 'random':
        k = int(round(fill * n_tiles * n_tiles))
        flat = np.zeros(n_tiles * n_tiles, np.uint8)
        flat[rng.permutation(n_tiles * n_tiles)[:k]] = 1
        return flat.reshape(n_tiles, n_tiles)
    ell = float(kind[len('cluster'):])
    filt, _ = _latent_corr(LATENT_PERIOD, ell)
    noise = rng.normal(size=(LATENT_PERIOD, LATENT_PERIOD))
    z = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    z /= np.sqrt(np.sum(filt ** 2) / LATENT_PERIOD ** 2)
    return (z[:n_tiles, :n_tiles] > norm.ppf(1 - fill)).astype(np.uint8)


def _orthant(t, r):
    """P(X>t, Y>t) for a standard bivariate normal with correlation r."""
    if r <= 0:
        r = max(r, -0.999999)
    g = lambda rho: np.exp(-t * t / (1 + rho)) / np.sqrt(1 - rho * rho)
    val, _ = integrate.quad(g, 0.0, min(r, 0.999999))
    return norm.sf(t) ** 2 + val / (2 * np.pi)


@lru_cache(maxsize=None)
def tile_covariance(kind, fill, n_tiles):
    """Covariance of two tile colours as a function of the tile offset.

    Returns a (2n-1) x (2n-1) array indexed by offset + n - 1.
    """
    m = 2 * n_tiles - 1
    if kind == 'random':
        big = n_tiles * n_tiles
        f = round(fill * big) / big
        c = np.full((m, m), -f * (1 - f) / (big - 1))
        c[n_tiles - 1, n_tiles - 1] = f * (1 - f)
        return c
    ell = float(kind[len('cluster'):])
    _, r = _latent_corr(LATENT_PERIOD, ell)
    t = norm.ppf(1 - fill)
    d = np.arange(-(n_tiles - 1), n_tiles)
    cache = {}
    c = np.empty((m, m))
    for a, dx in enumerate(d):
        for b, dy in enumerate(d):
            rho = round(float(r[dx % LATENT_PERIOD, dy % LATENT_PERIOD]), 12)
            if rho not in cache:
                cache[rho] = _orthant(t, rho) - fill * fill
            c[a, b] = cache[rho]
    return c


def model_fill(kind, fill, n_tiles):
    if kind == 'random':
        big = n_tiles * n_tiles
        return round(fill * big) / big
    return fill


def arena_reference(kind, fill, tiles_hit, n_tiles, flip=0.0):
    """Evaluation-only reference ESS for the arena fill ratio.

    tiles_hit: integer tile indices (row, col) of the pooled readings, from
    true positions. The target is the realised arena rate
    p_A = flip + (1-2 flip) f_A with f_A the black-tile fraction. Conditional
    on the paths, Var(ybar - p_A) = (1-2 flip)^2 w' S w + flip(1-flip)/n with
    w = h/n - 1/n_tiles^2 (h the reading count per tile) and S the tile
    covariance. The reference is p(1-p)/Var, clipped to [1, n].
    """
    tiles_hit = np.asarray(tiles_hit).reshape(-1, 2)
    n = len(tiles_hit)
    h = np.bincount(tiles_hit[:, 0] * n_tiles + tiles_hit[:, 1],
                    minlength=n_tiles * n_tiles).reshape(n_tiles, n_tiles)
    w = h / n - 1.0 / (n_tiles * n_tiles)
    m = 2 * n_tiles - 1
    W = np.fft.fft2(w, s=(2 * m, 2 * m))
    auto = np.fft.ifft2(np.abs(W) ** 2).real          # sum_a w_a w_{a+d}
    idx = np.arange(-(n_tiles - 1), n_tiles) % (2 * m)
    auto = auto[np.ix_(idx, idx)]
    var_z = float(np.sum(auto * tile_covariance(kind, fill, n_tiles)))
    f = model_fill(kind, fill, n_tiles)
    p = flip + (1 - 2 * flip) * f
    var = (1 - 2 * flip) ** 2 * max(var_z, 0.0) + flip * (1 - flip) / n
    return float(np.clip(p * (1 - p) / max(var, 1e-300), 1.0, n)), var


# --------------------------------------------------------------- robots
@dataclass
class ArenaRun:
    floor: np.ndarray            # n_tiles x n_tiles
    y: np.ndarray                # N x T readings (after flips)
    x_true: np.ndarray           # N x T x 2 true positions (m)
    x_rep: np.ndarray            # N x T x 2 reported positions (m)
    recv: np.ndarray             # N x N x T: robot i received robot j's reading t
    target: float                # p_A
    f_arena: float
    contacts: int                # robot-robot contacts resolved


def _launch(cfg, rng):
    # the nest side grows with sqrt(N) so that its density matches 20 robots in cfg.nest
    nest = min(cfg.side, cfg.nest * np.sqrt(cfg.n_robots / 20.0))
    lo, hi = cfg.radius, (nest if cfg.launch == 'nest' else cfg.side) - cfg.radius
    pos = np.empty((cfg.n_robots, 2))
    k = 0
    tries = 0
    while k < cfg.n_robots:
        tries += 1
        if tries > 100000:
            raise RuntimeError('launch area too small for the swarm')
        cand = rng.uniform(lo, hi, 2)
        if k == 0 or np.min(np.hypot(*(pos[:k] - cand).T)) > 2.2 * cfg.radius:
            pos[k] = cand
            k += 1
    return pos


def simulate_arena(seed, kind, fill, cfg: ArenaConfig):
    rng = np.random.default_rng(seed)
    n_tiles = int(round(cfg.side / cfg.tile))
    floor = make_floor(kind, fill, n_tiles, rng)
    N, r = cfg.n_robots, cfg.radius
    pos = _launch(cfg, rng)
    head = rng.uniform(-np.pi, np.pi, N)
    turning = np.zeros(N, bool)
    turn_left = np.zeros(N)
    run_left = rng.exponential(cfg.mean_run, N)
    # odometry state
    est = pos.copy()
    est_head = head.copy()
    scale_bias = 1 + rng.normal(0, cfg.odo_scale_sd, N)
    drift = rng.normal(0, cfg.odo_drift_sd, N)
    steps_per_sample = int(round(cfg.period / cfg.dt))
    T = cfg.samples
    y = np.empty((N, T), np.int8)
    xt = np.empty((N, T, 2))
    xr = np.empty((N, T, 2))
    recv = np.zeros((N, N, T), bool)
    contacts = 0
    for s in range(T):
        for _ in range(steps_per_sample):
            # obstacle detection in a +-60 degree frontal cone
            u = np.column_stack([np.cos(head), np.sin(head)])
            ahead = pos + (r + cfg.sense) * u
            wall = np.any((ahead < r) | (ahead > cfg.side - r), axis=1)
            d = pos[None, :, :] - pos[:, None, :]            # i -> j
            dist = np.hypot(d[..., 0], d[..., 1])
            np.fill_diagonal(dist, np.inf)
            front = np.einsum('ijk,ik->ij', d, u) > 0.5 * dist
            near = np.any((dist < 2 * r + cfg.sense) & front, axis=1)
            start = (~turning) & (wall | near | (run_left <= 0))
            if start.any():
                k = np.flatnonzero(start)
                obst = (wall | near)[k]
                ang = np.where(obst, rng.uniform(np.pi / 2, np.pi, k.size) * rng.choice([-1, 1], k.size),
                               rng.uniform(-np.pi, np.pi, k.size))
                turning[k] = True
                turn_left[k] = ang
            # turning in place
            dth = np.clip(turn_left, -cfg.omega * cfg.dt, cfg.omega * cfg.dt) * turning
            head = (head + dth + np.pi) % (2 * np.pi) - np.pi
            turn_left -= dth
            done = turning & (np.abs(turn_left) < 1e-9)
            if done.any():
                turning[done] = False
                run_left[done] = rng.exponential(cfg.mean_run, int(done.sum()))
            # forward motion
            fwd = (~turning).astype(float) * cfg.speed * cfg.dt
            run_left -= np.where(turning, 0.0, cfg.dt)
            old = pos.copy()
            pos = pos + fwd[:, None] * np.column_stack([np.cos(head), np.sin(head)])
            # contact resolution: push overlapping discs apart, keep inside walls
            for _ in range(3):
                d = pos[None, :, :] - pos[:, None, :]
                dist = np.hypot(d[..., 0], d[..., 1])
                np.fill_diagonal(dist, np.inf)
                ov = np.clip(2 * r - dist, 0, None)
                if not ov.any():
                    break
                contacts += int(np.count_nonzero(ov) // 2)
                push = -0.5 * (ov / np.maximum(dist, 1e-9))[..., None] * d
                pos = pos + push.sum(1)
                pos = np.clip(pos, r, cfg.side - r)
            pos = np.clip(pos, r, cfg.side - r)
            # dead reckoning from commanded wheel motion (contacts unobserved)
            est_head = est_head + dth + drift * cfg.dt + rng.normal(0, cfg.odo_step_sd, N) * (fwd > 0)
            est = est + (fwd * scale_bias)[:, None] * np.column_stack([np.cos(est_head), np.sin(est_head)])
            est = np.clip(est, 0.0, cfg.side)
        cell = np.minimum((pos / cfg.tile).astype(int), n_tiles - 1)
        y[:, s] = floor[cell[:, 0], cell[:, 1]]
        xt[:, s] = pos
        if cfg.position == 'exact':
            xr[:, s] = pos
        elif cfg.position == 'radio':
            xr[:, s] = np.clip(pos + rng.normal(0, cfg.radio_sd, (N, 2)), 0.0, cfg.side)
        elif cfg.position == 'odometry':
            xr[:, s] = est
        else:
            raise ValueError(cfg.position)
        d = pos[None, :, :] - pos[:, None, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        ok = (dist <= cfg.comm_range) & (rng.random((N, N)) >= cfg.loss)
        np.fill_diagonal(ok, True)
        recv[:, :, s] = ok
    if cfg.flip > 0:
        flips = np.random.default_rng([seed, 1]).random(y.shape) < cfg.flip
        y = np.where(flips, 1 - y, y).astype(np.int8)
    f_arena = float(floor.mean())
    return ArenaRun(floor, y, xt, xr, recv, cfg.flip + (1 - 2 * cfg.flip) * f_arena, f_arena, contacts)
