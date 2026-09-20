import time
from dataclasses import dataclass, field

import numpy as np

from ..filtering.particle import ParticleNoise, particle_filter


@dataclass
class ChainResult:
    names: list
    samples: np.ndarray
    loglik: np.ndarray
    accepted: np.ndarray
    terminal_states: np.ndarray
    trajectories: np.ndarray
    elapsed: float
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def acceptance_rate(self):
        return float(self.accepted.mean())

    def as_dicts(self):
        return [dict(zip(self.names, row)) for row in self.samples]

    def posterior_mean(self):
        return dict(zip(self.names, self.samples.mean(0)))

    def thin(self, burn_in, every=1):
        sl = slice(burn_in, None, every)
        return ChainResult(
            self.names,
            self.samples[sl],
            self.loglik[sl],
            self.accepted[sl],
            self.terminal_states[sl],
            self.trajectories[sl],
            self.elapsed,
            self.label,
            self.extra,
        )


class AdaptiveProposal:
    def __init__(
        self, dim, init_scale=0.05, start=200, eps=1e-6, beta=0.05, sd_factor=2.562
    ):
        self.dim = dim
        self.sd = sd_factor**2 / dim
        self.init_cov = (init_scale**2) * np.eye(dim)
        self.start = start
        self.eps = eps
        self.beta = beta
        self.n = 0
        self.mean = np.zeros(dim)
        self.cov = np.zeros((dim, dim))

    def update(self, u):
        self.n += 1
        if self.n == 1:
            self.mean = u.copy()
            return
        d = u - self.mean
        self.mean += d / self.n
        self.cov += (np.outer(d, u - self.mean) - self.cov) / self.n

    def draw(self, u, rng):
        if self.n < self.start or rng.random() < self.beta:
            L = np.linalg.cholesky(self.init_cov / self.dim)
        else:
            L = np.linalg.cholesky(self.sd * self.cov + self.eps * np.eye(self.dim))
        return u + L @ rng.standard_normal(self.dim)


class PMMH:
    def __init__(
        self,
        model,
        space,
        n_particles=256,
        n_iter=3000,
        rng=None,
        correlated=False,
        rho=0.99,
        resampling="systematic",
        auxiliary=False,
        adapt=True,
        init_scale=0.05,
        adapt_start=200,
        store_trajectories=True,
        verbose=0,
        label=None,
    ):
        self.model = model
        self.space = space
        self.N = n_particles
        self.n_iter = n_iter
        self.rng = rng or np.random.default_rng()
        self.correlated = correlated
        self.rho = rho
        self.pf_kw = {
            "resampling": resampling,
            "auxiliary": auxiliary,
            "ess_threshold": 1.0 if correlated else 0.5,
            "sort_for_correlation": correlated,
        }
        self.proposal = AdaptiveProposal(
            space.dim, init_scale, adapt_start if adapt else n_iter + 1
        )
        self.store_trajectories = store_trajectories
        self.verbose = verbose
        self.label = label or ("CPM-PMMH" if correlated else "PMMH")

    def _target(self, u, noise):
        theta = self.space.constrain(u)
        lp = self.space.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, None, theta
        out = particle_filter(
            self.model, theta, self.N, self.rng, noise=noise, **self.pf_kw
        )
        return lp + self.space.log_jacobian(u) + out.loglik, out.loglik, out, theta

    def _draw_noise(self):
        return ParticleNoise.draw(
            self.model.n_steps, self.N, self.model.n_noise, self.rng
        )

    def run(self, init=None):
        t0 = time.perf_counter()
        d = self.space.dim
        u = self.space.unconstrain(init or self.space.initial())
        noise = self._draw_noise() if self.correlated else None
        post, ll, out, theta = self._target(u, noise)
        tries = 0
        while not np.isfinite(post) and tries < 50:
            u = u + 0.1 * self.rng.standard_normal(d)
            post, ll, out, theta = self._target(u, noise)
            tries += 1
        samples = np.empty((self.n_iter, d))
        lls = np.empty(self.n_iter)
        acc = np.zeros(self.n_iter, bool)
        term = np.empty((self.n_iter, self.model.dim))
        trajs = np.empty(
            (
                self.n_iter if self.store_trajectories else 0,
                self.model.n_steps,
                self.model.dim,
            )
        )
        cur_term = out.sample_terminal(self.rng)[0]
        cur_traj = out.sample_trajectory(self.rng) if self.store_trajectories else None
        for i in range(self.n_iter):
            u_new = self.proposal.draw(u, self.rng)
            noise_new = noise.perturb(self.rho, self.rng) if self.correlated else None
            post_new, ll_new, out_new, theta_new = self._target(u_new, noise_new)
            if np.log(self.rng.random()) < post_new - post:
                u, post, ll, theta = u_new, post_new, ll_new, theta_new
                noise = noise_new
                cur_term = out_new.sample_terminal(self.rng)[0]
                if self.store_trajectories:
                    cur_traj = out_new.sample_trajectory(self.rng)
                acc[i] = True
            self.proposal.update(u)
            samples[i] = self.space.to_array(theta)
            lls[i] = ll
            term[i] = cur_term
            if self.store_trajectories:
                trajs[i] = cur_traj
            if self.verbose and (i + 1) % self.verbose == 0:
                print(
                    f"[{self.label}] iter {i + 1}/{self.n_iter} acc={acc[: i + 1].mean():.3f} loglik={ll:.2f}"
                )
        return ChainResult(
            self.space.names,
            samples,
            lls,
            acc,
            term,
            trajs,
            time.perf_counter() - t0,
            self.label,
        )


class KalmanMH(PMMH):
    def __init__(self, model, space, n_iter=3000, rng=None, **kw):
        super().__init__(
            model,
            space,
            n_particles=1,
            n_iter=n_iter,
            rng=rng,
            store_trajectories=False,
            **kw,
        )
        self.label = "exact-MH (Kalman)"

    def _target(self, u, noise):
        theta = self.space.constrain(u)
        lp = self.space.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf, -np.inf, None, theta
        ll = self.model.loglik(theta)
        return (
            lp + self.space.log_jacobian(u) + ll,
            ll,
            _KalmanStub(self.model, theta),
            theta,
        )


class _KalmanStub:
    def __init__(self, model, theta):
        self.model, self.theta = model, theta

    def sample_terminal(self, rng):
        return self.model.draw_states(self.theta, rng, 1)[:, -1]

    def sample_trajectory(self, rng):
        return self.model.draw_states(self.theta, rng, 1)[0]
