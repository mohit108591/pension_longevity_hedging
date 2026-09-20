import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ..filtering.kalman import LinearGaussianSSM, ffbs, kalman_filter, rts_smoother
from ..inference.parameters import InvGammaOnScale, Parameter, ParameterSpace

LOG2PI = np.log(2 * np.pi)


class LinearLeeCarterSSM:
    dim = 1
    n_noise = 1

    def __init__(
        self, data, alpha, beta, k0_mean, k0_sd=3.0, heteroscedastic=True, adapted=False
    ):
        self.data = data
        self.adapted = adapted
        self.alpha = np.asarray(alpha, float)
        self.beta = np.asarray(beta, float)
        self.y = data.log_rates.T
        self.n_steps = self.y.shape[0]
        self.k0_mean, self.k0_sd = float(k0_mean), float(k0_sd)
        inv_d = 1.0 / np.maximum(data.deaths.T, 1.0)
        self.base_var = inv_d if heteroscedastic else np.ones_like(inv_d)

    def ssm(self, theta):
        H = theta["sigma_eps"] ** 2 * self.base_var
        return LinearGaussianSSM(
            Z=self.beta[:, None],
            d=self.alpha,
            H=H,
            T=np.eye(1),
            c=np.array([theta["mu"]]),
            Q=np.array([[theta["sigma_eta"] ** 2]]),
            a0=np.array([self.k0_mean]),
            P0=np.array([[self.k0_sd**2]]),
        )

    def loglik(self, theta):
        return kalman_filter(self.y, self.ssm(theta)).loglik

    def smooth(self, theta):
        s = self.ssm(theta)
        res = kalman_filter(self.y, s)
        return rts_smoother(res, s)

    def draw_states(self, theta, rng, n_draws=1):
        s = self.ssm(theta)
        return ffbs(kalman_filter(self.y, s), s, rng, n_draws)

    def _obs_loglik(self, theta, t, k):
        var = theta["sigma_eps"] ** 2 * self.base_var[t]
        resid = (
            self.y[t][None, :] - self.alpha[None, :] - k[:, None] * self.beta[None, :]
        )
        return -0.5 * np.sum(LOG2PI + np.log(var) + resid**2 / var, axis=1)

    def _prior(self, theta, x_prev, n):
        if x_prev is None:
            return np.full(n, self.k0_mean), self.k0_sd**2
        return x_prev[:, 0] + theta["mu"], theta["sigma_eta"] ** 2

    def _predictive(self, theta, t, m, s2):
        h = theta["sigma_eps"] ** 2 * self.base_var[t]
        r = self.y[t][None, :] - self.alpha[None, :] - m[:, None] * self.beta[None, :]
        bhb = np.sum(self.beta**2 / h)
        bhr = r @ (self.beta / h)
        denom = 1 + s2 * bhb
        quad = np.sum(r**2 / h, axis=1) - s2 * bhr**2 / denom
        logdet = np.sum(np.log(h)) + np.log(denom)
        return -0.5 * (self.y.shape[1] * LOG2PI + logdet + quad)

    def _posterior(self, theta, t, m, s2):
        h = theta["sigma_eps"] ** 2 * self.base_var[t]
        v = 1.0 / (1.0 / s2 + np.sum(self.beta**2 / h))
        mean = v * (m / s2 + (self.y[t] - self.alpha) @ (self.beta / h))
        return mean, v

    def propose(self, theta, t, x_prev, z):
        n = z.shape[0]
        m, s2 = self._prior(theta, x_prev, n)
        if self.adapted:
            mean, v = self._posterior(theta, t, m, s2)
            k = mean + np.sqrt(v) * z[:, 0]
            return k[:, None], self._predictive(theta, t, m, s2)
        k = m + np.sqrt(s2) * z[:, 0]
        return k[:, None], self._obs_loglik(theta, t, k)

    def weight(self, theta, t, x_prev, x):
        x = np.atleast_2d(x)
        m, s2 = self._prior(
            theta, x_prev, x.shape[0] if x_prev is None else x_prev.shape[0]
        )
        if self.adapted:
            return self._predictive(theta, t, m, s2)
        return self._obs_loglik(theta, t, x[:, 0])

    def log_transition(self, theta, t, x_prev, x):
        s = theta["sigma_eta"]
        return stats.norm.logpdf(
            np.atleast_2d(x)[..., 0], x_prev[:, 0] + theta["mu"], s
        )

    def log_first_stage(self, theta, t, x_prev):
        m, s2 = self._prior(theta, x_prev, x_prev.shape[0])
        return self._predictive(theta, t, m, s2)

    def sort_key(self, x):
        return x[:, 0]

    def mle(self, init=None):
        x0 = np.array([-0.8, np.log(1.0), np.log(2.0)]) if init is None else init
        f = lambda u: (
            -self.loglik(
                {"mu": u[0], "sigma_eta": np.exp(u[1]), "sigma_eps": np.exp(u[2])}
            )
        )
        opt = minimize(
            f,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000},
        )
        u = opt.x
        theta = {
            "mu": u[0],
            "sigma_eta": float(np.exp(u[1])),
            "sigma_eps": float(np.exp(u[2])),
        }
        return theta, -opt.fun

    @staticmethod
    def default_space():
        return ParameterSpace(
            [
                Parameter("mu", stats.norm(0.0, 2.0), "identity", -0.5),
                Parameter("sigma_eta", InvGammaOnScale(2.0, 1.0), "log", 1.0),
                Parameter("sigma_eps", InvGammaOnScale(2.0, 2.0), "log", 1.5),
            ]
        )
