import numpy as np
from scipy.special import logsumexp


def normalise_log_weights(logw):
    lse = logsumexp(logw)
    return logw - lse, lse


def ess(logw_normalised):
    return float(1.0 / np.sum(np.exp(2 * logw_normalised)))


def _search(cdf, u):
    idx = np.searchsorted(cdf, u, side="right")
    return np.minimum(idx, cdf.size - 1)


def multinomial(w, rng, u=None):
    n = w.size
    u = rng.random(n) if u is None else u
    return _search(np.cumsum(w), np.sort(u))


def stratified(w, rng, u=None):
    n = w.size
    u = rng.random(n) if u is None else u
    return _search(np.cumsum(w), (np.arange(n) + u) / n)


def systematic(w, rng, u=None):
    n = w.size
    u0 = rng.random() if u is None else float(np.atleast_1d(u)[0])
    return _search(np.cumsum(w), (np.arange(n) + u0) / n)


def residual(w, rng, u=None):
    n = w.size
    counts = np.floor(n * w).astype(int)
    idx = np.repeat(np.arange(n), counts)
    rest = n - counts.sum()
    if rest:
        r = n * w - counts
        r /= r.sum()
        idx = np.concatenate([idx, _search(np.cumsum(r), np.sort(rng.random(rest)))])
    return idx


SCHEMES = {
    "multinomial": multinomial,
    "stratified": stratified,
    "systematic": systematic,
    "residual": residual,
}


def get_scheme(name):
    return SCHEMES[name]
