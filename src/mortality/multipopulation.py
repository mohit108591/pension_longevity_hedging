import numpy as np

from ..utils.timeseries import AR1, RandomWalkDrift
from .base import poisson_loglik
from .lee_carter import LeeCarter


class LiLee:
    name = "Li-Lee ACF"

    def fit(self, populations):
        self.pops = list(populations)
        logs = [p.log_rates for p in self.pops]
        self.a = [l.mean(1) for l in logs]
        w = np.array([p.exposures.sum() for p in self.pops], float)
        w /= w.sum()
        centred = sum(wi * (l - a[:, None]) for wi, l, a in zip(w, logs, self.a))
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        B, K = u[:, 0], s[0] * vt[0]
        sgn = B.sum()
        self.B, self.K = B / sgn, K * sgn
        self.b, self.k, self.k_ts = [], [], []
        for l, a in zip(logs, self.a):
            r = l - a[:, None] - np.outer(self.B, self.K)
            u, s, vt = np.linalg.svd(r, full_matrices=False)
            b, k = u[:, 0], s[0] * vt[0]
            sb = b.sum() if abs(b.sum()) > 1e-8 else np.sign(b[np.argmax(abs(b))])
            b, k = b / sb, k * sb
            self.b.append(b)
            self.k.append(k)
            self.k_ts.append(AR1.fit(k))
        self.K_ts = RandomWalkDrift.fit(self.K)
        return self

    def fitted_log_rates(self, i):
        return (
            self.a[i][:, None]
            + np.outer(self.B, self.K)
            + np.outer(self.b[i], self.k[i])
        )

    def loglik(self):
        return sum(
            poisson_loglik(p.deaths, p.exposures, self.fitted_log_rates(i))
            for i, p in enumerate(self.pops)
        )

    def explained_variance(self, i):
        l = self.pops[i].log_rates
        tot = np.sum((l - l.mean(1, keepdims=True)) ** 2)
        common = np.sum((l - self.a[i][:, None] - np.outer(self.B, self.K)) ** 2)
        full = np.sum((l - self.fitted_log_rates(i)) ** 2)
        return {"R2_common": 1 - common / tot, "R2_acf": 1 - full / tot}

    def simulate_log_rates(self, horizon, n_sims, rng):
        K = self.K_ts.simulate([self.K[-1]], horizon, n_sims, rng)[..., 0]
        out = []
        for i in range(len(self.pops)):
            k = self.k_ts[i].simulate(self.k[i][-1], horizon, n_sims, rng)
            out.append(
                self.a[i][None, :, None]
                + self.B[None, :, None] * K[:, None, :]
                + self.b[i][None, :, None] * k[:, None, :]
            )
        return out


class RelativeSpreadModel:
    name = "Relative LC spread"

    def fit(self, scheme, reference_log_rates):
        self.scheme = scheme
        self.offset = np.asarray(reference_log_rates, float)
        self.lc = LeeCarter("poisson").fit(scheme, offset=self.offset)
        self.alpha, self.beta, self.kappa = self.lc.a, self.lc.b, self.lc.k
        self.ts = AR1.fit(self.kappa, demean=False)
        return self

    def spread_log_rates(self, kappa_paths):
        return (
            self.alpha[None, :, None]
            + self.beta[None, :, None] * kappa_paths[:, None, :]
        )

    def simulate_spread(self, horizon, n_sims, rng):
        k = self.ts.simulate(self.kappa[-1], horizon, n_sims, rng)
        return k, self.spread_log_rates(k)

    def central_spread(self, horizon, last_kappa=None):
        last = self.kappa[-1] if last_kappa is None else last_kappa
        last = np.atleast_1d(last)
        steps = self.ts.phi ** np.arange(1, horizon + 1)
        k = last[:, None] * steps[None, :]
        return self.spread_log_rates(k)

    def loglik(self):
        return self.lc.loglik()
