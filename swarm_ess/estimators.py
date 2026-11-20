"""Effective-sample-size estimators and the evaluation-only reference.

Operating estimators receive only binary observations, sensed positions,
the arena side and robot identities. Only ``reference_ess`` reads the field.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import convolve2d

N_LENGTHS = 16
MAX_MODE = 7


# ---------------------------------------------------------------- local ESS
def local_ess(stream):
    """Initial-positive-sequence ESS of one robot's binary stream.

    Positive autocorrelations are summed with taper (1-k/n) up to the first
    non-positive lag or floor(sqrt(n)) lags. A constant stream returns 1.
    """
    y = np.asarray(stream, float)
    n = y.size
    c = y - y.mean()
    den = float(c @ c)
    if den <= 1e-12:
        return 1.0
    if n < 3:
        return float(n)
    total = 1.0
    for k in range(1, max(1, int(np.sqrt(n))) + 1):
        rho = float(c[:-k] @ c[k:]) / den
        if rho <= 0:
            break
        total += 2.0 * (1 - k / n) * rho
    return float(n / max(1.0, total))


def pooled_local_ess(y, mask=None):
    """Swarm-pooled local ESS of every robot's stream.

    One autocorrelation function by time lag is estimated from all robots'
    readings, centred at the pooled mean, so that a robot that stays in one
    patch keeps the level difference to the swarm as within-robot dependence.
    The sum is truncated by Geyer's initial monotone positive sequence on
    pair sums rho(2m)+rho(2m+1). Robot i then gets
    D_i = n_i + 2 sum_k P_i(k) rho(k), with P_i(k) its number of reading pairs
    at lag k (valid entries of ``mask`` only), and e_i = n_i^2 / D_i.

    A robot whose own readings all show one colour has no evidence of
    decorrelation and receives e_i = 1, as in the per-robot correction. With
    sensor flips such streams are rare; the swarm-level single-colour check
    of ``spectral_ess`` covers that case.
    """
    y = np.asarray(y, float)
    m = np.ones_like(y) if mask is None else np.asarray(mask, float)
    N, T = y.shape
    n = m.sum(1)
    tot = m.sum()
    c = (y - (y * m).sum() / tot) * m
    var0 = float((c * c).sum() / tot)
    if var0 <= 1e-12:
        return np.ones(N)
    if T < 2:
        return np.where(_one_colour_rows(y, m, n), np.minimum(n, 1.0), n)
    rho = np.zeros(T)
    rho[0] = 1.0
    pairs = np.zeros((N, T))
    for k in range(1, T):
        pk = (m[:, :-k] * m[:, k:]).sum(1)
        pairs[:, k] = pk
        pt = pk.sum()
        if pt > 0:
            rho[k] = float((c[:, :-k] * c[:, k:]).sum()) / pt / var0
    keep = np.zeros(T)
    keep[0] = 1.0
    prev = np.inf
    K = 0
    for j in range(T // 2):
        a, b = 2 * j, 2 * j + 1
        g = rho[a] + rho[b]
        if g <= 0:
            break
        scale = min(1.0, prev / g)
        if a > 0:
            keep[a] = rho[a] * scale
        keep[b] = rho[b] * scale
        prev = g * scale
        K = b
    D = n + 2.0 * (pairs[:, 1:K + 1] * keep[1:K + 1]).sum(1)
    D = np.clip(D, n, n * n)
    e = np.where(n > 0, n * n / np.maximum(D, 1e-12), 0.0)
    return np.where(_one_colour_rows(y, m, n), np.minimum(e, 1.0), e)


def _one_colour_rows(y, m, n):
    """Robots whose valid readings all show the same colour."""
    lo = np.where(m > 0, y, np.inf).min(1)
    hi = np.where(m > 0, y, -np.inf).max(1)
    return (n > 0) & (hi - lo <= 0)


def grouped_local_ess(counts, local):
    """Propagate robot variances v/e_i without cross-robot covariance."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    d = float(np.sum(counts ** 2 / np.minimum(local, counts)))
    return float(n * n / max(n, d)), d


# ---------------------------------------------------------- spectral basis
def lengths_grid(size, n_lengths=N_LENGTHS):
    return np.geomspace(max(1.0, size / 64.0), size / 2.0, n_lengths)


def kernel_weights(size, max_mode, lengths):
    """Normalised Gaussian weights q_k(l) on nonzero modes |k_1|,|k_2|<=M."""
    ax = np.arange(-max_mode, max_mode + 1)
    r2 = ax[:, None] ** 2 + ax[None, :] ** 2
    out = []
    for ell in lengths:
        w = np.exp(-0.5 * (2 * np.pi * ell / size) ** 2 * r2)
        w[max_mode, max_mode] = 0.0
        out.append(w / w.sum())
    return np.asarray(out)


def mode_sums(y, x, size, max_mode, mask=None):
    """U_i(k)=sum_t phi_k(x_it), V_i(k)=sum_t y_it phi_k(x_it), |k_j|<=2M.

    An optional 0/1 mask (N x T) drops padded entries of ragged streams.
    """
    k = np.arange(-2 * max_mode, 2 * max_mode + 1)
    cells = np.floor(x) % size + 0.5          # sensed cell centres
    ex = np.exp(2j * np.pi / size * cells[..., 0, None] * k)   # N x T x K2
    ey = np.exp(2j * np.pi / size * cells[..., 1, None] * k)
    if mask is not None:
        ex = ex * mask[..., None]
    U = np.einsum('ntk,ntl->nkl', ex, ey)
    V = np.einsum('ntk,ntl->nkl', ex * y[..., None], ey)
    return U, V


def _cross(Z):
    """sum_{i!=j} Z_i conj(Z_j) for stacked robot features."""
    return np.abs(Z.sum(0)) ** 2 - np.sum(np.abs(Z) ** 2, 0)


def spectral_ess(y, x, size, local, max_mode=MAX_MODE, centered=True, return_details=False,
                 flip_prob=0.0, mask=None):
    """Exact finite-spectrum cross-robot covariance correction.

    Fits v*a*K_l to all ordered different-robot residual products by least
    squares, with K_l a zero-mean periodic Gaussian-spectrum kernel. With
    ``centered=True`` the model kernel is centred in the same way as the
    residuals (K_c(x,x') = K - m(x) - m(x') + mbar), which removes the bias
    that pooled centring induces when the footprint is small compared with
    the kernel length. ``centered=False`` gives the previous estimator.
    All sums are evaluated exactly from mode sums; no pair is sampled.

    If the pooled readings show no variation beyond sensor flips of known
    rate ``flip_prob`` (all readings consistent with one true colour), the
    swarm has observed one patch and the estimator returns 1.

    ``mask`` (N x T, 0/1) marks the valid entries of ragged streams, for
    example the readings a robot received from each neighbour.
    """
    y = np.asarray(y, float)
    if mask is None:
        counts = np.full(y.shape[0], y.shape[1], float)
        n = counts.sum()
        p = y.mean()
    else:
        mask = np.asarray(mask, float)
        y = y * mask
        counts = mask.sum(1)
        n = counts.sum()
        p = y.sum() / n
    v = p * (1 - p)
    grouped, d_local = grouped_local_ess(counts, local)
    lengths = lengths_grid(size)
    one_colour = v <= 1e-12
    if flip_prob > 0:
        tol = 3.0 * np.sqrt(flip_prob * (1 - flip_prob) / n)
        one_colour = one_colour or min(abs(p - flip_prob), abs(p - 1 + flip_prob)) <= tol
    if one_colour:
        res = dict(ess=1.0, amplitude=0.0, length=float(lengths[-1]), cross=0.0,
                   status='constant_patch', grouped=grouped)
        return res
    M = max_mode
    U, V = mode_sums(y, x, size, M, mask)
    c = 2 * M
    sl = slice(c - M, c + M + 1)
    Uk = U[:, sl, sl]
    G = V[:, sl, sl] - p * Uk                    # residual features
    QU = _cross(Uk).real                         # target cross sum features
    weights = kernel_weights(size, M, lengths)
    if centered:
        K = 2 * M + 1
        ubar = Uk.sum(0) / n
        R = y.sum(1) - p * counts
        QH = _cross(G - R[:, None, None] * ubar[None]).real
        kk = np.arange(K) - M
        Ukh = U[:, c + kk[:, None, None, None] + kk[None, None, :, None],
                c + kk[None, :, None, None] + kk[None, None, None, :]]
        ub_k = ubar[None, :, :, None, None]
        ub_h = ubar[None, None, None, :, :]
        W = (Ukh - ub_h * Uk[:, :, :, None, None] - ub_k * Uk[:, None, None, :, :]
             + counts[:, None, None, None, None] * ub_k * ub_h)
        QW = _cross(W).real
        A = np.einsum('jab,ab->j', weights, QH) / v
        B = np.einsum('jab,abcd,jcd->j', weights, QW, weights)
    else:
        QG = _cross(G).real
        Qfull = _cross(U).real
        A = np.einsum('jab,ab->j', weights, QG) / v
        B = np.array([np.sum(convolve2d(w, w) * Qfull) for w in weights])
    amp = np.clip(A / np.maximum(B, 1e-12), 0.0, 1.0)
    obj = amp ** 2 * B - 2 * amp * A
    j = int(np.argmin(obj))
    cross = float(amp[j] * np.sum(weights[j] * QU))
    ess = float(np.clip(n * n / max(n, d_local + max(0.0, cross)), 1.0, n))
    res = dict(ess=ess, amplitude=float(amp[j]), length=float(lengths[j]), cross=cross,
               status='fitted' if amp[j] > 0 else 'zero_amplitude', grouped=grouped)
    if return_details:
        res.update(A=A, B=B, objective=obj)
    return res


# ------------------------------------------------------------ reference
def field_correlation(field):
    f = np.asarray(field, float)
    c = f - f.mean()
    var = float(np.mean(c * c))
    return np.fft.ifft2(np.abs(np.fft.fft2(c)) ** 2).real / (f.size * var)


def displacement_counts(x, size):
    cells = (np.floor(x.reshape(-1, 2)).astype(np.int64)) % size
    h = np.bincount(cells[:, 0] * size + cells[:, 1], minlength=size * size).reshape(size, size)
    return np.rint(np.fft.ifft2(np.abs(np.fft.fft2(h)) ** 2).real), len(cells)


def reference_ess(field, x, flip_prob=0.0, corr=None):
    """Spatiotemporal reference e_Z = n^2 / max(n, D), evaluation only.

    D sums the field correlation over all ordered observation pairs
    (common uniform translation of the realised paths). With symmetric
    flips, distinct observations keep a fraction rho of the covariance.
    """
    size = field.shape[0]
    corr = field_correlation(field) if corr is None else corr
    counts, n = displacement_counts(x, size)
    D = float(np.sum(counts * corr))
    if flip_prob > 0:
        f = float(field.mean())
        p = flip_prob + (1 - 2 * flip_prob) * f
        rho = (1 - 2 * flip_prob) ** 2 * f * (1 - f) / (p * (1 - p))
        D = n + rho * (D - n)
    return float(np.clip(n * n / max(n, D), 1.0, n)), D


def target_rate(field, flip_prob=0.0):
    f = float(field.mean())
    return flip_prob + (1 - 2 * flip_prob) * f
