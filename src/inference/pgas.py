import time

import numpy as np
from scipy.special import logsumexp

from ..filtering.particle import particle_filter
from ..filtering.resampling import get_scheme
from .pmmh import ChainResult


def conditional_smc(
    model,
    theta,
    ref,
    n_particles,
    rng,
    ancestor_sampling=True,
    resampling="multinomial",
):
    T, N = model.n_steps, n_particles
    resample = get_scheme(resampling)
    X = np.empty((T, N, model.dim))
    A = np.zeros((T, N), dtype=int)
    x, logw = model.propose(theta, 0, None, rng.standard_normal((N, model.n_noise)))
    x[-1] = ref[0]
    logw[-1] = model.weight(theta, 0, None, ref[0])[0]
    X[0] = x
    for t in range(1, T):
        W = np.exp(logw - logsumexp(logw))
        anc = resample(W, rng)
        x_new, lw_new = model.propose(
            theta, t, x[anc[:-1]], rng.standard_normal((N - 1, model.n_noise))
        )
        if ancestor_sampling:
            la = logw - logsumexp(logw) + model.log_transition(theta, t, x, ref[t])
            a_ref = rng.choice(N, p=np.exp(la - logsumexp(la)))
        else:
            a_ref = N - 1
        anc[-1] = a_ref
        lw_ref = model.weight(theta, t, x[a_ref][None, :], ref[t])[0]
        x = np.vstack([x_new, ref[t][None, :]])
        logw = np.append(lw_new, lw_ref)
        X[t], A[t] = x, anc
    k = rng.choice(N, p=np.exp(logw - logsumexp(logw)))
    path = np.empty((T, model.dim))
    for t in range(T - 1, -1, -1):
        path[t] = X[t, k]
        if t > 0:
            k = A[t, k]
    return path


class ParticleGibbs:
    def __init__(
        self,
        model,
        space,
        n_particles=64,
        n_iter=2000,
        rng=None,
        ancestor_sampling=True,
        verbose=0,
    ):
        self.model = model
        self.space = space
        self.N = n_particles
        self.n_iter = n_iter
        self.rng = rng or np.random.default_rng()
        self.ancestor_sampling = ancestor_sampling
        self.verbose = verbose
        self.label = "PGAS" if ancestor_sampling else "PG"

    def run(self, init=None):
        t0 = time.perf_counter()
        theta = dict(init or self.space.initial())
        out = particle_filter(self.model, theta, max(self.N, 256), self.rng)
        traj = out.sample_trajectory(self.rng)
        d, T = self.space.dim, self.model.n_steps
        samples = np.empty((self.n_iter, d))
        trajs = np.empty((self.n_iter, T, self.model.dim))
        moved = np.zeros((self.n_iter, T), bool)
        for i in range(self.n_iter):
            new = conditional_smc(
                self.model, theta, traj, self.N, self.rng, self.ancestor_sampling
            )
            moved[i] = np.abs(new[:, 0] - traj[:, 0]) > 1e-12
            traj = new
            theta = self.model.gibbs_update(theta, traj, self.space, self.rng)
            samples[i] = self.space.to_array(theta)
            trajs[i] = traj
            if self.verbose and (i + 1) % self.verbose == 0:
                print(
                    f"[{self.label}] iter {i + 1}/{self.n_iter} update-rate={moved[: i + 1].mean():.3f}"
                )
        return ChainResult(
            self.space.names,
            samples,
            np.full(self.n_iter, np.nan),
            moved.any(1),
            trajs[:, -1, :].copy(),
            trajs,
            time.perf_counter() - t0,
            self.label,
            {"update_rate_by_time": moved.mean(0)},
        )
