import numpy as np
from scipy import stats
from scipy.special import gammaln, ndtr, ndtri

from ..inference.parameters import (
    InvGammaOnScale,
    Parameter,
    ParameterSpace,
    ShiftedBeta,
)

LOG2PI = np.log(2 * np.pi)


def _lognorm(x, m, v):
    return -0.5 * (LOG2PI + np.log(v) + (x - m) ** 2 / v)


class PoissonLeeCarterSV:
    dim = 3
    n_noise = 3

    def __init__(
        self,
        data,
        alpha,
        beta,
        kappa_hat,
        jumps=True,
        k0_sd=2.0,
        newton_steps=3,
        proposal_inflation=1.15,
    ):
        self.data = data
        self.alpha = np.asarray(alpha, float)
        self.beta = np.asarray(beta, float)
        self.D = data.deaths.T.astype(float)
        self.E = data.exposures.T.astype(float)
        self.n_steps = self.D.shape[0]
        self.kappa_hat = np.asarray(kappa_hat, float)
        self.jumps = jumps
        self.k0_sd = k0_sd
        self.newton_steps = newton_steps
        self.infl = proposal_inflation**2
        self.log_e_alpha = np.log(self.E) + self.alpha[None, :]
        self.const = (
            np.sum(self.D * np.log(self.E) - gammaln(self.D + 1), axis=1)
            + self.D @ self.alpha
        )
        self.Db = self.D @ self.beta
        self.fisher = self.D @ self.beta**2

    def space(self):
        params = [
            Parameter("mu", stats.norm(-0.5, 1.5), "identity", -0.6),
            Parameter("omega", stats.norm(-0.5, 1.5), "identity", -0.5),
            Parameter("phi", ShiftedBeta(20.0, 1.5), "tanh", 0.85),
            Parameter("sigma_h", InvGammaOnScale(3.0, 0.3), "log", 0.3),
        ]
        if self.jumps:
            params += [
                Parameter("jump_p", stats.beta(1.5, 30.0), "logit", 0.03),
                Parameter("jump_mu", stats.norm(2.0, 1.0), "identity", 2.0),
                Parameter("jump_sd", stats.lognorm(0.5, scale=1.0), "log", 1.0),
            ]
        return ParameterSpace(params)

    def _jump(self, theta, name, default=0.0):
        return theta.get(name, default) if self.jumps else default

    def obs_loglik(self, t, k):
        eta = self.log_e_alpha[t][None, :] + np.outer(k, self.beta)
        return self.const[t] + k * self.Db[t] - np.exp(eta).sum(1)

    def _laplace(self, t, m, v):
        prec_lik = self.fisher[t]
        k = (m / v + self.kappa_hat[t] * prec_lik) / (1.0 / v + prec_lik)
        for _ in range(self.newton_steps):
            mu_x = np.exp(self.log_e_alpha[t][None, :] + np.outer(k, self.beta))
            grad = self.Db[t] - mu_x @ self.beta - (k - m) / v
            hess = -(mu_x @ self.beta**2) - 1.0 / v
            k = k - grad / hess
        mu_x = np.exp(self.log_e_alpha[t][None, :] + np.outer(k, self.beta))
        hess = -(mu_x @ self.beta**2) - 1.0 / v
        return k, -self.infl / hess

    def _prior_moments(self, theta, x_prev, h, j):
        if x_prev is None:
            return np.full(h.shape, self.kappa_hat[0]), np.full(h.shape, self.k0_sd**2)
        m = x_prev[:, 0] + theta["mu"] + j * self._jump(theta, "jump_mu")
        v = np.exp(h) + j * self._jump(theta, "jump_sd", 0.0) ** 2
        return m, v

    def _draw_h_j(self, theta, x_prev, z):
        omega, phi, s = theta["omega"], theta["phi"], theta["sigma_h"]
        if x_prev is None:
            h = omega + s / np.sqrt(1 - phi**2) * z[:, 1]
            j = np.zeros(z.shape[0])
        else:
            h = omega + phi * (x_prev[:, 1] - omega) + s * z[:, 1]
            p = self._jump(theta, "jump_p")
            j = (ndtr(z[:, 2]) < p).astype(float)
        return h, j

    def propose(self, theta, t, x_prev, z):
        h, j = self._draw_h_j(theta, x_prev, z)
        m, v = self._prior_moments(theta, x_prev, h, j)
        mode, s2 = self._laplace(t, m, v)
        k = mode + np.sqrt(s2) * z[:, 0]
        logw = _lognorm(k, m, v) + self.obs_loglik(t, k) - _lognorm(k, mode, s2)
        return np.column_stack([k, h, j]), logw

    def weight(self, theta, t, x_prev, x):
        x = np.atleast_2d(x)
        k, h, j = x[:, 0], x[:, 1], x[:, 2]
        m, v = self._prior_moments(theta, x_prev, h, j)
        mode, s2 = self._laplace(t, m, v)
        return _lognorm(k, m, v) + self.obs_loglik(t, k) - _lognorm(k, mode, s2)

    def log_transition(self, theta, t, x_prev, x):
        x = np.atleast_2d(x)
        k, h, j = x[:, 0], x[:, 1], x[:, 2]
        omega, phi, s = theta["omega"], theta["phi"], theta["sigma_h"]
        lh = _lognorm(h, omega + phi * (x_prev[:, 1] - omega), s**2)
        m, v = self._prior_moments(theta, x_prev, h, j)
        lk = _lognorm(k, m, v)
        if self.jumps:
            p = theta["jump_p"]
            lj = np.where(j > 0.5, np.log(p), np.log1p(-p))
        else:
            lj = 0.0
        return lh + lk + lj

    def log_first_stage(self, theta, t, x_prev):
        omega, phi, s = theta["omega"], theta["phi"], theta["sigma_h"]
        hbar = omega + phi * (x_prev[:, 1] - omega)
        p = self._jump(theta, "jump_p")
        m = x_prev[:, 0] + theta["mu"] + p * self._jump(theta, "jump_mu")
        v = np.exp(hbar + 0.5 * s**2) + p * (
            self._jump(theta, "jump_sd") ** 2
            + (1 - p) * self._jump(theta, "jump_mu") ** 2
        )
        mode, s2 = self._laplace(t, m, v)
        s2 = s2 / self.infl
        return (
            self.obs_loglik(t, mode)
            + _lognorm(mode, m, v)
            + 0.5 * (LOG2PI + np.log(s2))
        )

    def sort_key(self, x):
        return x[:, 0]

    def complete_loglik(self, theta, traj):
        k, h = traj[:, 0], traj[:, 1]
        prev = traj[:-1]
        ll = _lognorm(k[0], self.kappa_hat[0], self.k0_sd**2)
        ll += _lognorm(
            h[0], theta["omega"], theta["sigma_h"] ** 2 / (1 - theta["phi"] ** 2)
        )
        ll += np.sum(self.log_transition(theta, 1, prev, traj[1:]))
        return float(ll)

    def gibbs_update(self, theta, traj, space, rng):
        theta = dict(theta)
        k, h, j = traj[:, 0], traj[:, 1], traj[:, 2]
        dk = np.diff(k)
        jj = j[1:]
        mu_j = self._jump(theta, "jump_mu")
        sd_j = self._jump(theta, "jump_sd")
        v = np.exp(h[1:]) + jj * sd_j**2
        pr = space.params[space.names.index("mu")].prior
        prec = 1 / pr.var() + np.sum(1 / v)
        mean = (pr.mean() / pr.var() + np.sum((dk - jj * mu_j) / v)) / prec
        theta["mu"] = mean + rng.standard_normal() / np.sqrt(prec)
        phi, s2 = theta["phi"], theta["sigma_h"] ** 2
        pr = space.params[space.names.index("omega")].prior
        prec = 1 / pr.var() + (1 - phi**2) / s2 + (h.size - 1) * (1 - phi) ** 2 / s2
        num = (
            pr.mean() / pr.var()
            + (1 - phi**2) * h[0] / s2
            + (1 - phi) * np.sum(h[1:] - phi * h[:-1]) / s2
        )
        theta["omega"] = num / prec + rng.standard_normal() / np.sqrt(prec)
        omega = theta["omega"]
        hc = h - omega
        sxx = hc[:-1] @ hc[:-1]
        phi_hat = (hc[:-1] @ hc[1:]) / sxx
        sd = np.sqrt(s2 / sxx)
        lo, hi = (-1 - phi_hat) / sd, (1 - phi_hat) / sd
        phi_new = phi_hat + sd * ndtri(ndtr(lo) + rng.random() * (ndtr(hi) - ndtr(lo)))
        phi_new = float(np.clip(phi_new, -0.9999, 0.9999))
        prior_phi = space.params[space.names.index("phi")].prior
        init = lambda f: 0.5 * np.log(1 - f**2) - 0.5 * (1 - f**2) * hc[0] ** 2 / s2
        log_acc = (
            prior_phi.logpdf(phi_new)
            + init(phi_new)
            - prior_phi.logpdf(phi)
            - init(phi)
        )
        if np.log(rng.random()) < log_acc:
            theta["phi"] = phi_new
        phi = theta["phi"]
        ig = space.params[space.names.index("sigma_h")].prior
        rss = np.sum((hc[1:] - phi * hc[:-1]) ** 2) + (1 - phi**2) * hc[0] ** 2
        a_post = ig.a + 0.5 * h.size
        b_post = ig.b + 0.5 * rss
        theta["sigma_h"] = float(np.sqrt(b_post / rng.gamma(a_post)))
        if self.jumps:
            bp = space.params[space.names.index("jump_p")].prior
            a0, b0 = bp.args
            theta["jump_p"] = float(rng.beta(a0 + jj.sum(), b0 + jj.size - jj.sum()))
            theta = self._mh_jump_size(theta, traj, space, rng)
        return theta

    def _mh_jump_size(self, theta, traj, space, rng, steps=5, scale=0.3):
        idx = [space.names.index("jump_mu"), space.names.index("jump_sd")]

        def target(th):
            lp = sum(space.params[i].prior.logpdf(th[space.names[i]]) for i in idx)
            return lp + self.complete_loglik(th, traj)

        cur = target(theta)
        for _ in range(steps):
            prop = dict(theta)
            prop["jump_mu"] = theta["jump_mu"] + scale * rng.standard_normal()
            prop["jump_sd"] = theta["jump_sd"] * np.exp(scale * rng.standard_normal())
            new = target(prop)
            log_acc = new - cur + np.log(prop["jump_sd"] / theta["jump_sd"])
            if np.log(rng.random()) < log_acc:
                theta, cur = prop, new
        return theta

    def effective_drift(self, theta):
        return theta["mu"] + self._jump(theta, "jump_p") * self._jump(theta, "jump_mu")

    def simulate_forward(self, theta, x_last, horizon, rng):
        x_last = np.atleast_2d(x_last)
        n = x_last.shape[0]
        k = np.empty((n, horizon))
        h = np.empty((n, horizon))
        kp, hp = x_last[:, 0].copy(), x_last[:, 1].copy()
        omega, phi, s = theta["omega"], theta["phi"], theta["sigma_h"]
        p, mj, sj = (
            self._jump(theta, "jump_p"),
            self._jump(theta, "jump_mu"),
            self._jump(theta, "jump_sd"),
        )
        for i in range(horizon):
            hp = omega + phi * (hp - omega) + s * rng.standard_normal(n)
            jump = rng.random(n) < p
            kp = (
                kp
                + theta["mu"]
                + np.exp(0.5 * hp) * rng.standard_normal(n)
                + jump * (mj + sj * rng.standard_normal(n))
            )
            k[:, i], h[:, i] = kp, hp
        return k, h
