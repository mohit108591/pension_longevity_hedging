import numpy as np
from scipy.optimize import brentq

from ..utils.timeseries import RandomWalkDrift
from .base import StochasticMortalityModel, poisson_deviance


def _normalise(a, b, k):
    kbar = k.mean()
    a = a + b * kbar
    k = k - kbar
    s = b.sum()
    return a, b / s, k * s


def _refit_kappa(a, b, deaths, exposures, k0):
    k = k0.copy()
    for t in range(k.size):
        target = deaths[:, t].sum()
        f = lambda kt, t=t, target=target: (
            np.sum(exposures[:, t] * np.exp(a + b * kt)) - target
        )
        lo, hi = k0[t] - 50.0, k0[t] + 50.0
        try:
            k[t] = brentq(f, lo, hi, xtol=1e-10)
        except ValueError:
            pass
    return k


class LeeCarter(StochasticMortalityModel):
    name = "LC"

    def __init__(self, method="poisson", reestimate=True, max_iter=1000, tol=1e-9):
        super().__init__()
        self.method = method
        self.reestimate = reestimate
        self.max_iter = max_iter
        self.tol = tol
        self.offset = None

    def _svd_start(self, logm):
        a = logm.mean(1)
        u, s, vt = np.linalg.svd(logm - a[:, None], full_matrices=False)
        b, k = u[:, 0], s[0] * vt[0]
        if b.sum() < 0:
            b, k = -b, -k
        return _normalise(a, b, k)

    def fit(self, data, offset=None):
        self.data = data
        self.offset = (
            np.zeros(data.deaths.shape) if offset is None else np.asarray(offset, float)
        )
        D, E = data.deaths, data.exposures
        logm = data.log_rates - self.offset
        a, b, k = self._svd_start(logm)
        if self.method == "svd":
            if self.reestimate and offset is None:
                k = _refit_kappa(a, b, D, E, k)
                a, b, k = _normalise(a, b, k)
        else:
            a, b, k, self.n_iter_ = self._poisson_newton(
                D, E * np.exp(self.offset), a, b, k
            )
        self.a, self.b, self.k = a, b, k
        self.ts = RandomWalkDrift.fit(k)
        self.fitted_ = True
        return self

    def _poisson_newton(self, D, E, a, b, k):
        dev_old = np.inf
        for it in range(self.max_iter):
            mu = E * np.exp(a[:, None] + np.outer(b, k))
            a = a + (D - mu).sum(1) / mu.sum(1)
            mu = E * np.exp(a[:, None] + np.outer(b, k))
            k = k + ((D - mu) * b[:, None]).sum(0) / (mu * b[:, None] ** 2).sum(0)
            a, b, k = _normalise(a, b, k)
            mu = E * np.exp(a[:, None] + np.outer(b, k))
            b = b + ((D - mu) * k[None, :]).sum(1) / (mu * k[None, :] ** 2).sum(1)
            a, b, k = _normalise(a, b, k)
            dev = poisson_deviance(D, E, a[:, None] + np.outer(b, k))
            if abs(dev_old - dev) < self.tol * max(1.0, abs(dev)):
                break
            dev_old = dev
        return a, b, k, it + 1

    def fitted_log_rates(self):
        return self.offset + self.a[:, None] + np.outer(self.b, self.k)

    @property
    def n_params(self):
        nx, nt = self.data.deaths.shape
        return 2 * nx + nt - 2

    def kappa_paths(self, horizon, n_sims, rng, parameter_uncertainty=True):
        return self.ts.simulate(
            [self.k[-1]], horizon, n_sims, rng, parameter_uncertainty
        )[..., 0]

    def log_rates_from_kappa(self, kappa):
        return self.a[None, :, None] + self.b[None, :, None] * kappa[:, None, :]

    def simulate_log_rates(self, horizon, n_sims, rng, parameter_uncertainty=True):
        return self.log_rates_from_kappa(
            self.kappa_paths(horizon, n_sims, rng, parameter_uncertainty)
        )

    def central_log_rates(self, horizon):
        k = self.ts.central([self.k[-1]], horizon)[:, 0]
        return self.a[:, None] + np.outer(self.b, k)
