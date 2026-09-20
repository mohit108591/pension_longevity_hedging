from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, ndtr

from .resampling import ess, get_scheme, systematic


@dataclass
class ParticleNoise:
    z: np.ndarray
    u: np.ndarray

    @classmethod
    def draw(cls, n_steps, n_particles, n_noise, rng):
        return cls(
            rng.standard_normal((n_steps, n_particles, n_noise)),
            rng.standard_normal(n_steps),
        )

    def perturb(self, rho, rng):
        s = np.sqrt(1 - rho**2)
        return ParticleNoise(
            rho * self.z + s * rng.standard_normal(self.z.shape),
            rho * self.u + s * rng.standard_normal(self.u.shape),
        )


@dataclass
class FilterOutput:
    loglik: float
    particles: np.ndarray
    logw: np.ndarray
    ancestors: np.ndarray
    ess: np.ndarray
    resampled: np.ndarray

    def filtering_mean(self):
        w = np.exp(self.logw)
        return np.einsum("tn,tnd->td", w, self.particles)

    def filtering_quantiles(self, dim, qs=(0.05, 0.5, 0.95)):
        out = np.empty((self.particles.shape[0], len(qs)))
        for t in range(self.particles.shape[0]):
            x = self.particles[t, :, dim]
            order = np.argsort(x)
            cdf = np.cumsum(np.exp(self.logw[t, order]))
            out[t] = x[order][np.minimum(np.searchsorted(cdf, qs), x.size - 1)]
        return out

    def sample_terminal(self, rng, size=1):
        idx = rng.choice(self.logw.shape[1], size=size, p=np.exp(self.logw[-1]))
        return self.particles[-1, idx]

    def trace_lineage(self, idx):
        n = self.particles.shape[0]
        path = np.empty((n, self.particles.shape[2]))
        for t in range(n - 1, -1, -1):
            path[t] = self.particles[t, idx]
            if t > 0:
                idx = self.ancestors[t, idx]
        return path

    def sample_trajectory(self, rng):
        idx = rng.choice(self.logw.shape[1], p=np.exp(self.logw[-1]))
        return self.trace_lineage(idx)


def particle_filter(
    model,
    theta,
    n_particles,
    rng,
    noise=None,
    resampling="systematic",
    ess_threshold=0.5,
    auxiliary=False,
    sort_for_correlation=False,
):
    n_steps = model.n_steps
    if noise is None:
        noise = ParticleNoise.draw(n_steps, n_particles, model.n_noise, rng)
    resample = get_scheme(resampling)
    N = n_particles
    parts = np.empty((n_steps, N, model.dim))
    logw_store = np.empty((n_steps, N))
    anc_store = np.zeros((n_steps, N), dtype=int)
    ess_store = np.empty(n_steps)
    res_flag = np.zeros(n_steps, bool)
    x, incr = model.propose(theta, 0, None, noise.z[0])
    lse = logsumexp(incr)
    loglik = lse - np.log(N)
    logW = incr - lse
    parts[0], logw_store[0] = x, logW
    ess_store[0] = ess(logW)
    for t in range(1, n_steps):
        u_t = ndtr(noise.u[t])
        if auxiliary:
            v = model.log_first_stage(theta, t, x)
            lw1 = logW + v
            l1 = logsumexp(lw1)
            anc = _resample(
                resample, x, lw1 - l1, rng, u_t, sort_for_correlation, model
            )
            res_flag[t] = True
            x_new, incr = model.propose(theta, t, x[anc], noise.z[t])
            lw = incr - v[anc]
            l2 = logsumexp(lw)
            loglik += l1 + l2 - np.log(N)
            logW = lw - l2
        else:
            if ess_store[t - 1] < ess_threshold * N:
                anc = _resample(
                    resample, x, logW, rng, u_t, sort_for_correlation, model
                )
                prev = np.full(N, -np.log(N))
                res_flag[t] = True
            else:
                anc = np.arange(N)
                prev = logW
            x_new, incr = model.propose(theta, t, x[anc], noise.z[t])
            lw = prev + incr
            lse = logsumexp(lw)
            loglik += lse
            logW = lw - lse
        x = x_new
        parts[t], logw_store[t], anc_store[t] = x, logW, anc
        ess_store[t] = ess(logW)
        if not np.isfinite(loglik):
            break
    return FilterOutput(
        float(loglik), parts, logw_store, anc_store, ess_store, res_flag
    )


def _resample(resample, x, logW, rng, u, sort_for_correlation, model):
    w = np.exp(logW)
    w /= w.sum()
    uu = np.array([u]) if resample is systematic else None
    if sort_for_correlation:
        order = np.argsort(model.sort_key(x), kind="stable")
        return order[resample(w[order], rng, uu)]
    return resample(w, rng, uu)


def estimate_loglik_variance(model, theta, n_particles, reps, rng, **kw):
    lls = np.array(
        [
            particle_filter(model, theta, n_particles, rng, **kw).loglik
            for _ in range(reps)
        ]
    )
    return float(lls.mean()), float(lls.std(ddof=1))


def tune_particle_count(
    model, theta, rng, target_sd=1.2, start=64, max_particles=8192, reps=20, **kw
):
    n = start
    history = []
    while True:
        _, sd = estimate_loglik_variance(model, theta, n, reps, rng, **kw)
        history.append((n, sd))
        if sd <= target_sd or n >= max_particles:
            return n, history
        n *= 2
