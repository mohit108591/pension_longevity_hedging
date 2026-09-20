from dataclasses import dataclass, field

import numpy as np


@dataclass
class MortalityData:
    ages: np.ndarray
    years: np.ndarray
    deaths: np.ndarray
    exposures: np.ndarray
    label: str = "population"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ages = np.asarray(self.ages, dtype=int)
        self.years = np.asarray(self.years, dtype=int)
        self.deaths = np.asarray(self.deaths, dtype=float)
        self.exposures = np.asarray(self.exposures, dtype=float)
        shape = (self.ages.size, self.years.size)
        if self.deaths.shape != shape or self.exposures.shape != shape:
            raise ValueError(f"expected arrays of shape {shape}")

    @property
    def rates(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = self.deaths / self.exposures
        return np.where(self.exposures > 0, m, np.nan)

    @property
    def log_rates(self):
        m = self.rates
        floor = np.nanmin(m[m > 0]) if np.any(m > 0) else 1e-8
        return np.log(np.where(m > 0, m, floor))

    @property
    def qx(self):
        return 1.0 - np.exp(-self.rates)

    @property
    def initial_exposures(self):
        return self.exposures + 0.5 * self.deaths

    @property
    def cohorts(self):
        return self.years[None, :] - self.ages[:, None]

    def subset(self, ages=None, years=None):
        ia = np.ones(self.ages.size, bool) if ages is None else np.isin(self.ages, ages)
        iy = (
            np.ones(self.years.size, bool)
            if years is None
            else np.isin(self.years, years)
        )
        return MortalityData(
            self.ages[ia],
            self.years[iy],
            self.deaths[np.ix_(ia, iy)],
            self.exposures[np.ix_(ia, iy)],
            self.label,
            dict(self.meta),
        )

    def window(self, age_range, year_range):
        ages = np.arange(age_range[0], age_range[1] + 1)
        years = np.arange(year_range[0], year_range[1] + 1)
        return self.subset(ages, years)

    def __repr__(self):
        return (
            f"MortalityData({self.label}, ages {self.ages[0]}-{self.ages[-1]}, "
            f"years {self.years[0]}-{self.years[-1]}, deaths={self.deaths.sum():,.0f})"
        )


@dataclass
class YieldPanel:
    dates: np.ndarray
    maturities: np.ndarray
    yields: np.ndarray

    def last_curve(self):
        return self.maturities, self.yields[-1]
