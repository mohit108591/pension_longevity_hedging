from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..liabilities.lifetables import cohort_survival, extend_old_ages


def wang_expectation(x, lam):
    x = np.sort(np.asarray(x, float))
    n = x.size
    grid = np.arange(n + 1) / n
    F = norm.cdf(norm.ppf(np.clip(grid, 1e-12, 1 - 1e-12)) - lam)
    F[0], F[-1] = 0.0, 1.0
    return float(np.diff(F) @ x)


def market_price_of_risk_from_premium(sample, target_ratio, lo=-2.0, hi=2.0, tol=1e-8):
    base = np.mean(sample)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if wang_expectation(sample, mid) / base - 1 < target_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass
class QForward:
    age: int
    maturity: int
    notional: float = 1.0
    fixed: float = np.nan
    exposure: float = None
    seed: int = 0
    tag: str = ""

    @property
    def name(self):
        return f"qF{self.age}{self.tag}"

    def realised(self, logm, ages):
        i = int(np.where(ages == self.age)[0][0])
        m = np.exp(logm[:, i, self.maturity - 1])
        if self.exposure:
            rng = np.random.default_rng(self.seed * 1000 + self.age)
            m = rng.poisson(self.exposure * m) / self.exposure
        return 1 - np.exp(-m)

    def price(self, logm_pricing, ages, lam):
        p = 1 - self.realised(logm_pricing, ages)
        self.fixed = 1 - wang_expectation(p, lam)
        return self

    def payoff(self, logm, ages):
        return self.notional * (self.fixed - self.realised(logm, ages))


@dataclass
class SForward:
    age: int
    maturity: int
    notional: float = 1.0
    fixed: float = np.nan
    max_age: int = 110

    @property
    def name(self):
        return f"SF{self.age}"

    def realised(self, logm, ages):
        ext, ea = extend_old_ages(logm, ages, self.max_age)
        return cohort_survival(ext, ea, [self.age])[:, 0, self.maturity]

    def price(self, logm_pricing, ages, lam):
        self.fixed = wang_expectation(self.realised(logm_pricing, ages), lam)
        return self

    def payoff(self, logm, ages):
        return self.notional * (self.realised(logm, ages) - self.fixed)


@dataclass
class IndexLongevitySwap:
    age: int
    term: int
    notional: float = 1.0
    max_age: int = 110
    fixed_leg: np.ndarray = None

    @property
    def name(self):
        return f"LS{self.age}"

    def survival(self, logm, ages):
        ext, ea = extend_old_ages(logm, ages, self.max_age)
        return cohort_survival(ext, ea, [self.age])[:, 0, 1 : self.term + 1]

    def price(self, logm_pricing, ages, lam):
        s = self.survival(logm_pricing, ages)
        self.fixed_leg = np.array(
            [wang_expectation(s[:, j], lam) for j in range(s.shape[1])]
        )
        return self

    def value_at_horizon(self, logm, ages, tau, accumulation, discount_after):
        s = self.survival(logm, ages)
        net = s - self.fixed_leg[None, :]
        paid = net[:, :tau] @ accumulation[:tau]
        future = np.sum(net[:, tau:] * discount_after[..., : self.term - tau], axis=-1)
        return self.notional * (paid + future)


@dataclass
class ZeroCouponBond:
    maturity: float
    notional: float = 1.0

    @property
    def name(self):
        return f"ZCB{int(self.maturity)}"

    def pnl_at_horizon(self, tau, curve0, df_tau):
        cost = curve0.df(self.maturity) / curve0.df(tau)
        if self.maturity <= tau:
            return np.zeros_like(df_tau(1.0))
        return self.notional * (df_tau(self.maturity - tau) - cost)
