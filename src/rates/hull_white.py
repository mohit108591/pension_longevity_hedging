import numpy as np
from scipy.optimize import minimize


class HullWhite:
    def __init__(self, curve, a=0.05, sigma=0.01):
        self.curve = curve
        self.a = a
        self.sigma = sigma

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def alpha(self, t):
        a, s = self.a, self.sigma
        return self.curve.forward(t) + s**2 / (2 * a**2) * (1 - np.exp(-a * t)) ** 2

    def zcb(self, t, T, r_t):
        a, s = self.a, self.sigma
        B = self.B(t, T)
        P0T, P0t = self.curve.df(T), self.curve.df(t)
        lnA = (
            np.log(P0T / P0t)
            + B * self.curve.forward(t)
            - s**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2
        )
        return np.exp(lnA - B * np.asarray(r_t)[..., None])

    def simulate(self, horizon, n_sims, rng, steps_per_year=12):
        n = round(horizon * steps_per_year)
        dt = 1.0 / steps_per_year
        a, s = self.a, self.sigma
        e = np.exp(-a * dt)
        sd = s * np.sqrt((1 - e**2) / (2 * a))
        cov_xi = s**2 / a**2 * (dt - 2 * (1 - e) / a + (1 - e**2) / (2 * a))
        cross = s**2 / (2 * a**2) * (1 - e) ** 2
        cov = np.array([[sd**2, cross], [cross, cov_xi]])
        L = np.linalg.cholesky(cov + 1e-18 * np.eye(2))
        x = np.zeros(n_sims)
        integral = np.zeros(n_sims)
        for _ in range(n):
            z = rng.standard_normal((n_sims, 2)) @ L.T
            integral += x * (1 - e) / a + z[:, 1]
            x = x * e + z[:, 0]
        t = n * dt
        r = x + self.alpha(t)
        a_int = -np.log(self.curve.df(t)) + 0.5 * self._var_int(t)
        acc = np.exp(integral + a_int)
        return r, acc

    def _var_int(self, t):
        a, s = self.a, self.sigma
        return (
            s**2
            / a**2
            * (
                t
                + 2 / a * np.exp(-a * t)
                - 1 / (2 * a) * np.exp(-2 * a * t)
                - 3 / (2 * a)
            )
        )

    def curve_at(self, t, r_t, grid=None):
        grid = np.arange(0.5, 80.5, 0.5) if grid is None else grid
        P = self.zcb(t, t + grid, r_t)
        return grid, -np.log(P) / grid

    def calibrate_historical(self, short_rates, dt):
        r = np.asarray(short_rates, float)

        def nll(u):
            a, s, m = np.exp(u[0]), np.exp(u[1]), u[2]
            e = np.exp(-a * dt)
            v = s**2 * (1 - e**2) / (2 * a)
            mean = m + (r[:-1] - m) * e
            return 0.5 * np.sum(np.log(2 * np.pi * v) + (r[1:] - mean) ** 2 / v)

        opt = minimize(
            nll,
            [np.log(0.1), np.log(0.01), r.mean()],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )
        self.a, self.sigma = float(np.exp(opt.x[0])), float(np.exp(opt.x[1]))
        self.long_run = float(opt.x[2])
        return self
