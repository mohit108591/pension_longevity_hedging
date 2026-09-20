from dataclasses import dataclass

import numpy as np


@dataclass
class RandomWalkDrift:
    drift: np.ndarray
    cov: np.ndarray
    n_obs: int

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[:, None]
        dx = np.diff(x, axis=0)
        mu = dx.mean(0)
        cov = np.atleast_2d(np.cov(dx, rowvar=False, ddof=1))
        return cls(mu, cov, dx.shape[0])

    def simulate(self, last, horizon, n_sims, rng, parameter_uncertainty=True):
        d = self.drift.size
        chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(d))
        mu = np.broadcast_to(self.drift, (n_sims, d)).copy()
        if parameter_uncertainty:
            mu += rng.standard_normal((n_sims, d)) @ chol.T / np.sqrt(self.n_obs)
        eps = rng.standard_normal((n_sims, horizon, d)) @ chol.T
        steps = mu[:, None, :] + eps
        return np.asarray(last, float)[None, None, :] + np.cumsum(steps, axis=1)

    def central(self, last, horizon):
        return (
            np.asarray(last, float)[None, :]
            + np.arange(1, horizon + 1)[:, None] * self.drift[None, :]
        )


@dataclass
class AR1:
    const: float
    phi: float
    sigma: float

    @classmethod
    def fit(cls, x, demean=True):
        x = np.asarray(x, float)
        y, z = x[1:], x[:-1]
        if demean:
            X = np.column_stack([np.ones_like(z), z])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            c, phi = coef
        else:
            c, phi = 0.0, float(z @ y / (z @ z))
        phi = float(np.clip(phi, -0.995, 0.995))
        resid = y - c - phi * z
        return cls(float(c), phi, float(resid.std(ddof=2 if demean else 1)))

    @property
    def mean(self):
        return self.const / (1 - self.phi)

    def simulate(self, last, horizon, n_sims, rng):
        out = np.empty((n_sims, horizon))
        prev = np.full(n_sims, float(last))
        for h in range(horizon):
            prev = (
                self.const + self.phi * prev + self.sigma * rng.standard_normal(n_sims)
            )
            out[:, h] = prev
        return out

    def central(self, last, horizon):
        out = np.empty(horizon)
        prev = float(last)
        for h in range(horizon):
            prev = self.const + self.phi * prev
            out[h] = prev
        return out


@dataclass
class VAR1:
    mean: np.ndarray
    phi: np.ndarray
    cov: np.ndarray

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, float)
        y, z = x[1:], x[:-1]
        X = np.column_stack([np.ones(len(z)), z])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c, phi = coef[0], coef[1:].T
        resid = y - X @ coef
        cov = resid.T @ resid / (len(y) - X.shape[1])
        mean = np.linalg.solve(np.eye(len(c)) - phi, c)
        return cls(mean, phi, cov)

    def simulate(self, last, horizon, n_sims, rng):
        d = self.mean.size
        chol = np.linalg.cholesky(self.cov + 1e-14 * np.eye(d))
        out = np.empty((n_sims, horizon, d))
        prev = np.broadcast_to(np.asarray(last, float), (n_sims, d)).copy()
        for h in range(horizon):
            prev = (
                self.mean
                + (prev - self.mean) @ self.phi.T
                + rng.standard_normal((n_sims, d)) @ chol.T
            )
            out[:, h] = prev
        return out
