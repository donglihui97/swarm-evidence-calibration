"""Binary fields, deployment designs and correlated random walks on a torus.

Sensor flips use a separate random stream, so noisy and noiseless runs
share fields and trajectories.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def generate_field(size, xi, fill, rng):
    """Threshold a Gaussian-smoothed white field at its empirical quantile."""
    noise = rng.normal(size=(size, size))
    if xi == 0:
        smooth = noise
    else:
        f = np.fft.fftfreq(size)
        fx, fy = np.meshgrid(f, f, indexing='ij')
        filt = np.exp(-0.5 * xi ** 2 * (2 * np.pi) ** 2 * (fx * fx + fy * fy))
        smooth = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    cutoff = np.quantile(smooth, 1.0 - fill)
    return (smooth >= cutoff).astype(np.uint8)


def heading_update(angle, target, persistence, noise):
    delta = np.arctan2(np.sin(target - angle), np.cos(target - angle))
    return (angle + (1.0 - persistence) * delta + noise + np.pi) % (2 * np.pi) - np.pi


@dataclass
class World:
    field: np.ndarray        # L x L binary field
    y: np.ndarray            # N x T sensed values (after optional flips)
    x: np.ndarray            # N x T x 2 continuous positions
    group: np.ndarray        # N launch-group labels
    flip_prob: float


def simulate(seed, xi, fill, groups, sigma=2.0, follow=False, n_robots=50, size=128,
             samples=64, persistence=0.85, step=1.0, heading_noise=0.08, flip_prob=0.0):
    """Simulate one world.

    groups: number of independent launch groups (equal size). groups=n_robots
    gives independent uniform origins. Within a group, origins are the group
    centre plus wrapped Gaussian offsets of standard deviation sigma.
    follow: each follower moves 0.35/(1+sigma/size) of the periodic shortest
    displacement towards the first robot of its group after every step.
    """
    if n_robots % groups:
        raise ValueError('equal-sized launch groups are required')
    rng = np.random.default_rng(seed)
    field = generate_field(size, xi, fill, rng)
    ids = np.repeat(np.arange(groups), n_robots // groups)
    centres = rng.uniform(0, size, (groups, 2))
    pos = (centres[ids] + rng.normal(0, sigma, (n_robots, 2))) % size
    angles = rng.uniform(-np.pi, np.pi, n_robots)
    y = np.empty((n_robots, samples), np.int8)
    x = np.empty((n_robots, samples, 2))
    pull = 0.35 / (1 + sigma / size)
    for t in range(samples):
        angles = heading_update(angles, rng.uniform(-np.pi, np.pi, n_robots), persistence,
                                rng.normal(0, heading_noise, n_robots))
        pos += step * np.column_stack([np.cos(angles), np.sin(angles)])
        if follow:
            for g in range(groups):
                idx = np.flatnonzero(ids == g)
                d = pos[idx[0]] - pos[idx[1:]]
                d -= size * np.round(d / size)
                pos[idx[1:]] += pull * d
        pos %= size
        cell = np.floor(pos).astype(int)
        y[:, t] = field[cell[:, 0], cell[:, 1]]
        x[:, t] = pos
    if flip_prob > 0:
        noise_rng = np.random.default_rng([seed, 1])
        flips = noise_rng.random(y.shape) < flip_prob
        y = np.where(flips, 1 - y, y).astype(np.int8)
    return World(field, y, x, ids, float(flip_prob))
