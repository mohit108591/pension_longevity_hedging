import numpy as np
from scipy.special import expit

from ..utils.timeseries import AR1, RandomWalkDrift
from .base import StochasticMortalityModel, cohort_index


class CairnsBlakeDowd(StochasticMortalityModel):
    def __init__(
        self, quadratic=False, cohort=False, max_iter=200, tol=1e-8, min_cells=5
    ):
        super().__init__()
        self.quadratic = quadratic
        self.cohort = cohort
        self.max_iter = max_iter
        self.tol = tol
        self.min_cells = min_cells
        self.name = "M7" if (quadratic and cohort) else ("CBD-Q" if quadratic else "M5")

    def _design(self, ages):
        u = ages - ages.mean()
        cols = [np.ones_like(u), u]
        self.sigma2 = float(np.mean(u**2))
        if self.quadratic:
            cols.append(u**2 - self.sigma2)
        return np.column_stack(cols)

    def _eta(self):
        return self.X @ self.k + self.gamma[self.cinv]

    def fit(self, data):
        self.data = data
        ages = data.ages.astype(float)
        self.X = self._design(ages)
        D, E0 = data.deaths, data.initial_exposures
        self.cohorts, self.cinv, self.cvalid = cohort_index(
            data.ages, data.years, self.min_cells
        )
        self.gamma = np.zeros(self.cohorts.size)
        w = self.mask()
        q = np.clip(D / E0, 1e-6, 1 - 1e-6)
        D, E0 = D * w, E0 * w
        z = np.log(q / (1 - q))
        self.k = np.linalg.lstsq(self.X, z, rcond=None)[0]
        ll_old = -np.inf
        for it in range(self.max_iter):
            self._update_kappa(D, E0)
            if self.cohort:
                self._update_gamma(D, E0)
                self._apply_constraints()
            ll = self._binomial_ll(D, E0)
            if abs(ll - ll_old) < self.tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        self.n_iter_ = it + 1
        self.ts = RandomWalkDrift.fit(self.k.T)
        if self.cohort:
            g = self.gamma[self.cvalid]
            self.gamma_ts = AR1.fit(np.diff(g))
        self.fitted_ = True
        return self

    def _binomial_ll(self, D, E0):
        q = np.clip(expit(self._eta()), 1e-12, 1 - 1e-12)
        return float(np.sum(D * np.log(q) + (E0 - D) * np.log(1 - q)))

    def _update_kappa(self, D, E0):
        off = self.gamma[self.cinv]
        for t in range(D.shape[1]):
            beta = self.k[:, t]
            for _ in range(4):
                eta = self.X @ beta + off[:, t]
                q = expit(eta)
                w = E0[:, t] * q * (1 - q)
                score = self.X.T @ (D[:, t] - E0[:, t] * q)
                info = self.X.T @ (w[:, None] * self.X)
                beta = beta + np.linalg.solve(info, score)
            self.k[:, t] = beta

    def _update_gamma(self, D, E0):
        q = expit(self._eta())
        idx = self.cinv.ravel()
        score = np.bincount(idx, (D - E0 * q).ravel(), self.cohorts.size)
        info = np.bincount(idx, (E0 * q * (1 - q)).ravel(), self.cohorts.size)
        step = np.where(self.cvalid, score / np.maximum(info, 1e-12), 0.0)
        self.gamma = np.where(self.cvalid, self.gamma + step, 0.0)

    def _apply_constraints(self):
        c = self.cohorts[self.cvalid].astype(float)
        g = self.gamma[self.cvalid]
        xbar = self.data.ages.mean()
        cc = c - c.mean()
        degree = 2 if self.quadratic else 1
        P = np.column_stack([cc**p for p in range(degree + 1)])
        phi = np.linalg.lstsq(P, g, rcond=None)[0]
        self.gamma[self.cvalid] = g - P @ phi
        s = self.data.years - xbar - c.mean()
        if degree == 2:
            p0, p1, p2 = phi
            self.k[0] += p0 + p1 * s + p2 * s**2 + p2 * self.sigma2
            self.k[1] += -p1 - 2 * p2 * s
            self.k[2] += p2
        else:
            p0, p1 = phi
            self.k[0] += p0 + p1 * s
            self.k[1] += -p1

    def fitted_qx(self):
        return expit(self._eta())

    def fitted_log_rates(self):
        return np.log(-np.log1p(-self.fitted_qx()))

    def mask(self):
        return (
            self.cvalid[self.cinv]
            if self.cohort
            else np.ones(self.data.deaths.shape, bool)
        )

    @property
    def n_params(self):
        nt = self.data.years.size
        p = self.k.shape[0] * nt
        if self.cohort:
            p += int(self.cvalid.sum()) - (3 if self.quadratic else 2)
        return p

    def _future_gamma(self, horizon, n_sims, rng):
        ages, years = self.data.ages, self.data.years
        last_valid = self.cohorts[self.cvalid].max()
        c_needed = np.arange(
            years[-1] + 1 - ages.max(), years[-1] + horizon - ages.min() + 1
        )
        full = np.zeros((n_sims, c_needed.size))
        lookup = dict(zip(self.cohorts[self.cvalid], self.gamma[self.cvalid]))
        n_new = int(max(0, c_needed.max() - last_valid))
        if n_new:
            g = self.gamma[self.cvalid]
            dg = self.gamma_ts.simulate(g[-1] - g[-2], n_new, n_sims, rng)
            new = g[-1] + np.cumsum(dg, axis=1)
        for j, c in enumerate(c_needed):
            if c in lookup:
                full[:, j] = lookup[c]
            elif c > last_valid:
                full[:, j] = new[:, c - last_valid - 1]
        return c_needed, full

    def simulate_log_rates(self, horizon, n_sims, rng, parameter_uncertainty=True):
        kp = self.ts.simulate(
            self.k[:, -1], horizon, n_sims, rng, parameter_uncertainty
        )
        eta = np.einsum("xp,nhp->nxh", self.X, kp)
        if self.cohort:
            c_needed, g = self._future_gamma(horizon, n_sims, rng)
            fy = self.data.years[-1] + np.arange(1, horizon + 1)
            coh = fy[None, :] - self.data.ages[:, None]
            eta = eta + g[:, coh - c_needed[0]]
        q = expit(eta)
        return np.log(-np.log1p(-q))
