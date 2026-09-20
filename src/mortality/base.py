from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gammaln


def poisson_loglik(deaths, exposures, log_rates, mask=None):
    mu = exposures * np.exp(log_rates)
    ll = deaths * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(deaths + 1)
    if mask is not None:
        ll = ll[mask]
    return float(np.nansum(ll))


def poisson_deviance(deaths, exposures, log_rates, mask=None):
    mu = exposures * np.exp(log_rates)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(deaths > 0, deaths * np.log(deaths / mu), 0.0)
    dev = 2 * (term - (deaths - mu))
    if mask is not None:
        dev = dev[mask]
    return float(np.nansum(dev))


def cohort_index(ages, years, min_cells=3):
    coh = years[None, :] - ages[:, None]
    uniq, inv = np.unique(coh, return_inverse=True)
    inv = inv.reshape(coh.shape)
    counts = np.bincount(inv.ravel(), minlength=uniq.size)
    return uniq, inv, counts >= min_cells


class StochasticMortalityModel(ABC):
    name = "model"

    def __init__(self):
        self.data = None
        self.fitted_ = False

    @abstractmethod
    def fit(self, data): ...

    @abstractmethod
    def fitted_log_rates(self): ...

    @property
    @abstractmethod
    def n_params(self): ...

    @abstractmethod
    def simulate_log_rates(self, horizon, n_sims, rng): ...

    def mask(self):
        return np.ones(self.data.deaths.shape, bool)

    def loglik(self):
        d = self.data
        return poisson_loglik(
            d.deaths, d.exposures, self.fitted_log_rates(), self.mask()
        )

    def deviance(self):
        d = self.data
        return poisson_deviance(
            d.deaths, d.exposures, self.fitted_log_rates(), self.mask()
        )

    def bic(self):
        return self.loglik() - 0.5 * self.n_params * np.log(self.mask().sum())

    def aic(self):
        return self.loglik() - self.n_params

    def residuals(self):
        d = self.data
        mu = d.exposures * np.exp(self.fitted_log_rates())
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(d.deaths > 0, d.deaths * np.log(d.deaths / mu), 0.0)
        dev = 2 * (term - (d.deaths - mu))
        return np.sign(d.deaths - mu) * np.sqrt(np.maximum(dev, 0))

    def summary(self):
        return {
            "model": self.name,
            "loglik": self.loglik(),
            "deviance": self.deviance(),
            "n_params": self.n_params,
            "AIC": self.aic(),
            "BIC": self.bic(),
        }
