import numpy as np

from ..utils.timeseries import AR1, RandomWalkDrift
from .base import StochasticMortalityModel, cohort_index, poisson_deviance
from .lee_carter import LeeCarter


class RenshawHaberman(StochasticMortalityModel):
    name = "RH-H1"

    def __init__(self, max_iter=2000, tol=1e-8, min_cells=5):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.min_cells = min_cells

    def fit(self, data):
        self.data = data
        nx, _ = data.deaths.shape
        self.cohorts, self.cinv, self.cvalid = cohort_index(
            data.ages, data.years, self.min_cells
        )
        w = self.mask()
        D, E = data.deaths * w, data.exposures * w
        lc = LeeCarter("poisson").fit(data)
        a, b, k = lc.a.copy(), lc.b.copy(), lc.k.copy()
        g = np.zeros(self.cohorts.size)
        b0 = 1.0 / nx
        idx = self.cinv.ravel()
        dev_old = np.inf
        eta = lambda: a[:, None] + np.outer(b, k) + b0 * g[self.cinv]
        for it in range(self.max_iter):
            mu = E * np.exp(eta())
            a += (D - mu).sum(1) / np.maximum(mu.sum(1), 1e-12)
            mu = E * np.exp(eta())
            sc = np.bincount(idx, ((D - mu) * b0).ravel(), g.size)
            inf = np.bincount(idx, (mu * b0**2).ravel(), g.size)
            g = np.where(self.cvalid, g + sc / np.maximum(inf, 1e-12), 0.0)
            gbar = g[self.cvalid].mean()
            g[self.cvalid] -= gbar
            a += b0 * gbar
            mu = E * np.exp(eta())
            k += ((D - mu) * b[:, None]).sum(0) / np.maximum(
                (mu * b[:, None] ** 2).sum(0), 1e-12
            )
            kbar = k.mean()
            a += b * kbar
            k -= kbar
            mu = E * np.exp(eta())
            b += ((D - mu) * k[None, :]).sum(1) / np.maximum(
                (mu * k[None, :] ** 2).sum(1), 1e-12
            )
            s = b.sum()
            b, k = b / s, k * s
            dev = poisson_deviance(D, E, eta(), w)
            if abs(dev_old - dev) < self.tol * max(1.0, dev):
                break
            dev_old = dev
        self.n_iter_ = it + 1
        self.a, self.b, self.k, self.g, self.b0 = a, b, k, g, b0
        self.ts = RandomWalkDrift.fit(k)
        self.gamma_ts = AR1.fit(np.diff(g[self.cvalid]))
        self.fitted_ = True
        return self

    def fitted_log_rates(self):
        return self.a[:, None] + np.outer(self.b, self.k) + self.b0 * self.g[self.cinv]

    def mask(self):
        return self.cvalid[self.cinv]

    @property
    def n_params(self):
        nx, nt = self.data.deaths.shape
        return 2 * nx + nt + int(self.cvalid.sum()) - 3

    def simulate_log_rates(self, horizon, n_sims, rng, parameter_uncertainty=True):
        ages, years = self.data.ages, self.data.years
        kp = self.ts.simulate(
            [self.k[-1]], horizon, n_sims, rng, parameter_uncertainty
        )[..., 0]
        gv = self.g[self.cvalid]
        cv = self.cohorts[self.cvalid]
        last = cv.max()
        fy = years[-1] + np.arange(1, horizon + 1)
        coh = fy[None, :] - ages[:, None]
        n_new = int(max(0, coh.max() - last))
        table = np.zeros((n_sims, coh.max() - cv.min() + 1))
        table[:, cv - cv.min()] = gv
        if n_new:
            dg = self.gamma_ts.simulate(gv[-1] - gv[-2], n_new, n_sims, rng)
            table[:, last - cv.min() + 1 :] = gv[-1] + np.cumsum(dg, axis=1)
        gam = table[:, coh - cv.min()]
        return (
            self.a[None, :, None]
            + self.b[None, :, None] * kp[:, None, :]
            + self.b0 * gam
        )
