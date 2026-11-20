"""Timing of the centred spectral fit versus direct all-pairs evaluation."""
from common import *
import time
from swarm_ess import simulator, estimators as est

def allpairs(y, x, size, local, M=7):
    """Direct evaluation of A, B and the cross sum over all different-robot pairs."""
    yy = y.ravel().astype(float); pts = (np.floor(x.reshape(-1, 2)) % size) + .5
    rid = np.repeat(np.arange(y.shape[0]), y.shape[1]); n = len(yy)
    p = yy.mean(); v = p * (1 - p); r = yy - p
    lengths = est.lengths_grid(size); W = est.kernel_weights(size, M, lengths)
    ax = np.arange(-M, M + 1)
    # pairwise features via cos/sin of all modes: K(x-x') = sum_k q_k cos(k.(x-x'))
    ph = 2 * np.pi * (pts[:, 0, None] * ax[None, :]) / size
    ph2 = 2 * np.pi * (pts[:, 1, None] * ax[None, :]) / size
    E = (np.exp(1j * ph)[:, :, None] * np.exp(1j * ph2)[:, None, :]).reshape(n, -1)
    off = rid[:, None] != rid[None, :]
    best = None
    for j, w in enumerate(W):
        K = (E * w.ravel()[None, :]) @ E.conj().T
        K = K.real
        m = K.mean(1); Kc = K - m[:, None] - m[None, :] + m.mean()
        A = float(np.sum((r[:, None] * r[None, :] / v * Kc)[off])); B = float(np.sum((Kc ** 2)[off]))
        a = min(1, max(0, A / max(B, 1e-12))); loss = a * a * B - 2 * a * A
        if best is None or loss < best[0]:
            best = (loss, a, float(np.sum(K[off])))
    return best[1] * best[2]

rows = []
for (N, T) in [(10, 16), (20, 32), (50, 64), (100, 64)]:
    w = simulator.simulate(777 + N, 8.0, .48, 1, n_robots=N, samples=T)
    loc = est.pooled_local_ess(w.y)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); res = est.spectral_ess(w.y, w.x, 128, loc); ts.append(time.perf_counter() - t0)
    row = dict(N=N, T=T, n=N * T, spectral_ms=1000 * np.median(ts))
    if N * T <= 3200:
        t0 = time.perf_counter(); c = allpairs(w.y, w.x, 128, loc); row['allpairs_ms'] = 1000 * (time.perf_counter() - t0)
        row['rel_diff'] = abs(c - res['cross']) / max(abs(c), 1e-9)
    rows.append(row); print(row, flush=True)
    NUMBERS[f'bench_{N}x{T}'] = row
save_numbers()
