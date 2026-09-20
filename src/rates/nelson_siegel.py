import numpy as np
from scipy.optimize import minimize

from ..filtering.kalman import LinearGaussianSSM, kalman_filter, rts_smoother
from ..utils.timeseries import VAR1
from .curves import DiscountCurve


def ns_loadings(tau, lam):
    x = lam * np.asarray(tau, float)
    x = np.where(x == 0, 1e-10, x)
    s = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), s, s - np.exp(-x)])


def _tril_from(v, n=3):
    L = np.zeros((n, n))
    L[np.tril_indices(n)] = v
    L[np.diag_indices(n)] = np.exp(np.diag(L))
    return L


class DynamicNelsonSiegel:
    def __init__(self, dt=1 / 12, lam=0.7308, estimate_lambda=True):
        self.dt = dt
        self.lam = lam
        self.estimate_lambda = estimate_lambda

    def two_step(self, panel):
        L = ns_loadings(panel.maturities, self.lam)
        betas = np.linalg.lstsq(L, panel.yields.T, rcond=None)[0].T
        var = VAR1.fit(betas)
        resid = panel.yields - betas @ L.T
        return betas, var, resid.std(0)

    def _unpack(self, u):
        i = 0
        lam = np.exp(u[i]) if self.estimate_lambda else self.lam
        i += int(self.estimate_lambda)
        mu = u[i : i + 3]
        i += 3
        phi = np.tanh(u[i : i + 3])
        i += 3
        L = _tril_from(u[i : i + 6])
        i += 6
        h = np.exp(u[i])
        return lam, mu, np.diag(phi), L, h

    def _ssm(self, u, p):
        lam, mu, Phi, L, h = self._unpack(u)
        Z = ns_loadings(self.maturities, lam)
        Q = L @ L.T
        P0 = np.eye(3) * 1e-3
        return LinearGaussianSSM(
            Z, np.zeros(p), np.full(p, h**2), Phi, (np.eye(3) - Phi) @ mu, Q, mu, P0
        )

    def fit(self, panel, method="mle", maxiter=400):
        self.maturities = np.asarray(panel.maturities, float)
        betas, var, resid_sd = self.two_step(panel)
        phi0 = np.clip(np.diag(var.phi), -0.995, 0.995)
        Lc = np.linalg.cholesky(var.cov + 1e-12 * np.eye(3))
        lv = Lc[np.tril_indices(3)].copy()
        diag_pos = [0, 2, 5]
        lv[diag_pos] = np.log(np.diag(Lc))
        u0 = np.concatenate(
            [
                [np.log(self.lam)] if self.estimate_lambda else [],
                var.mean,
                np.arctanh(phi0),
                lv,
                [np.log(resid_sd.mean())],
            ]
        )
        y = panel.yields
        f = lambda u: -kalman_filter(y, self._ssm(u, y.shape[1])).loglik
        if method == "mle":
            opt = minimize(
                f, u0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10}
            )
            u, self.converged = opt.x, bool(opt.success)
        else:
            u, self.converged = u0, True
        self.u = u
        self.lam, self.mu, self.Phi, self.L, self.h = self._unpack(u)
        ssm = self._ssm(u, y.shape[1])
        res = kalman_filter(y, ssm)
        self.loglik = res.loglik
        self.factors, _ = rts_smoother(res, ssm)
        self.two_step_factors = betas
        return self

    def curve_from_factors(self, beta, grid=None):
        grid = np.arange(0.5, 80.5, 0.5) if grid is None else grid
        z = ns_loadings(grid, self.lam) @ beta
        return DiscountCurve(grid, z)

    def current_curve(self, grid=None):
        return self.curve_from_factors(self.factors[-1], grid)

    def simulate_factors(self, horizon_years, n_sims, rng):
        steps = round(horizon_years / self.dt)
        b = np.broadcast_to(self.factors[-1], (n_sims, 3)).copy()
        for _ in range(steps):
            b = (
                self.mu
                + (b - self.mu) @ self.Phi.T
                + rng.standard_normal((n_sims, 3)) @ self.L.T
            )
        return b

    def zero_rates(self, factors, maturities):
        return factors @ ns_loadings(maturities, self.lam).T
