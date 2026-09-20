from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit, gammaln, logit


class InvGammaOnScale:
    def __init__(self, shape, scale):
        self.a, self.b = shape, scale

    def logpdf(self, sd):
        v = sd**2
        if sd <= 0:
            return -np.inf
        return (
            self.a * np.log(self.b)
            - gammaln(self.a)
            - (self.a + 1) * np.log(v)
            - self.b / v
            + np.log(2 * sd)
        )


class ShiftedBeta:
    def __init__(self, a, b, lo=-1.0, hi=1.0):
        self.dist = stats.beta(a, b)
        self.lo, self.hi = lo, hi

    def logpdf(self, x):
        if not self.lo < x < self.hi:
            return -np.inf
        w = self.hi - self.lo
        return self.dist.logpdf((x - self.lo) / w) - np.log(w)


TRANSFORMS = {
    "identity": (lambda u: u, lambda x: x, lambda u: 0.0),
    "log": (np.exp, np.log, lambda u: u),
    "logit": (expit, logit, lambda u: -np.logaddexp(0, u) - np.logaddexp(0, -u)),
    "tanh": (np.tanh, np.arctanh, lambda u: np.log1p(-(np.tanh(u) ** 2))),
}


@dataclass
class Parameter:
    name: str
    prior: object
    transform: str = "identity"
    init: float = 0.0


class ParameterSpace:
    def __init__(self, params):
        self.params = list(params)
        self.names = [p.name for p in self.params]

    @property
    def dim(self):
        return len(self.params)

    def constrain(self, u):
        return {
            p.name: float(TRANSFORMS[p.transform][0](ui))
            for p, ui in zip(self.params, u)
        }

    def unconstrain(self, theta):
        return np.array(
            [TRANSFORMS[p.transform][1](theta[p.name]) for p in self.params], float
        )

    def log_jacobian(self, u):
        return float(
            sum(TRANSFORMS[p.transform][2](ui) for p, ui in zip(self.params, u))
        )

    def log_prior(self, theta):
        total = 0.0
        for p in self.params:
            lp = p.prior.logpdf(theta[p.name])
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return float(total)

    def initial(self):
        return {p.name: p.init for p in self.params}

    def to_array(self, theta):
        return np.array([theta[n] for n in self.names])
