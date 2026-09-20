from dataclasses import dataclass

import numpy as np

from .containers import MortalityData, YieldPanel


@dataclass
class SyntheticTruth:
    alpha: np.ndarray
    beta: np.ndarray
    kappa: np.ndarray
    logvol: np.ndarray
    jumps: np.ndarray
    spread_alpha: np.ndarray
    spread_beta: np.ndarray
    spread_kappa: np.ndarray
    params: dict


def _gompertz_alpha(ages):
    return -10.3 + 0.095 * ages + 0.00012 * (ages - 75.0) ** 2


def _age_loading(ages):
    raw = 1.6 - 0.9 * (ages - ages[0]) / (ages[-1] - ages[0])
    return raw / raw.sum()


def simulate_sv_kappa(n_years, mu, omega, phi, sigma_h, jump_p, jump_mu, jump_sd, rng):
    h = np.empty(n_years)
    k = np.empty(n_years)
    j = rng.random(n_years) < jump_p
    h[0] = omega + sigma_h / np.sqrt(1 - phi**2) * rng.standard_normal()
    k[0] = 0.0
    for t in range(1, n_years):
        h[t] = omega + phi * (h[t - 1] - omega) + sigma_h * rng.standard_normal()
        shock = np.exp(0.5 * h[t]) * rng.standard_normal()
        if j[t]:
            shock += jump_mu + jump_sd * rng.standard_normal()
        k[t] = k[t - 1] + mu + shock
    j[0] = False
    return k - k.mean(), h, j


def _exposure_surface(ages, years, alpha, beta, kappa, base, growth):
    lograte = alpha[:, None] + beta[:, None] * kappa[None, :]
    hazard = np.exp(lograte)
    cum = np.vstack([np.zeros(years.size), np.cumsum(hazard[:-1], axis=0)])
    trend = (1 + growth) ** (years - years[0])
    return (
        base
        * np.exp(-cum)
        * trend[None, :]
        * np.exp(-0.004 * (ages[:, None] - ages[0]))
    )


def make_two_populations(
    ages=(55, 95),
    years=(1970, 2019),
    seed=7,
    reference_size=380_000,
    scheme_size=6_000,
    sv=None,
    spread=None,
):
    rng = np.random.default_rng(seed)
    age_grid = np.arange(ages[0], ages[1] + 1)
    year_grid = np.arange(years[0], years[1] + 1)
    sv = sv or {
        "mu": -0.8,
        "omega": np.log(0.8**2),
        "phi": 0.9,
        "sigma_h": 0.35,
        "jump_p": 0.04,
        "jump_mu": 2.5,
        "jump_sd": 1.0,
    }
    spread = spread or {"phi": 0.7, "sigma": 1.5}
    alpha = _gompertz_alpha(age_grid.astype(float))
    beta = _age_loading(age_grid.astype(float))
    kappa, h, jumps = simulate_sv_kappa(year_grid.size, rng=rng, **sv)
    ref_log = alpha[:, None] + beta[:, None] * kappa[None, :]
    ref_expo = _exposure_surface(
        age_grid, year_grid, alpha, beta, kappa, reference_size, 0.008
    )
    ref_deaths = rng.poisson(ref_expo * np.exp(ref_log))
    s_alpha = -0.18 + 0.0045 * (age_grid - age_grid[0])
    s_beta = np.full(age_grid.size, 1.0 / age_grid.size) * (
        1.2 - 0.4 * (age_grid - age_grid[0]) / (age_grid[-1] - age_grid[0])
    )
    s_beta /= s_beta.sum()
    s_kappa = np.empty(year_grid.size)
    s_kappa[0] = (
        spread["sigma"] / np.sqrt(1 - spread["phi"] ** 2) * rng.standard_normal()
    )
    for t in range(1, year_grid.size):
        s_kappa[t] = (
            spread["phi"] * s_kappa[t - 1] + spread["sigma"] * rng.standard_normal()
        )
    sch_log = ref_log + s_alpha[:, None] + s_beta[:, None] * s_kappa[None, :]
    sch_expo = _exposure_surface(
        age_grid, year_grid, alpha + s_alpha, beta, kappa, scheme_size, 0.02
    )
    sch_deaths = rng.poisson(sch_expo * np.exp(sch_log))
    truth = SyntheticTruth(
        alpha,
        beta,
        kappa,
        h,
        jumps,
        s_alpha,
        s_beta,
        s_kappa,
        {"sv": sv, "spread": spread},
    )
    ref = MortalityData(
        age_grid, year_grid, ref_deaths, ref_expo, "reference", {"source": "synthetic"}
    )
    sch = MortalityData(
        age_grid, year_grid, sch_deaths, sch_expo, "scheme", {"source": "synthetic"}
    )
    return ref, sch, truth


def nelson_siegel_loadings(maturities, lam):
    x = lam * np.asarray(maturities, float)
    slope = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), slope, slope - np.exp(-x)])


def make_yield_panel(n_months=240, maturities=None, seed=11, lam=0.7308):
    rng = np.random.default_rng(seed)
    tau = np.array(maturities or [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], float)
    mean = np.array([0.042, -0.017, -0.004])
    phi = np.diag([0.985, 0.965, 0.93])
    chol = np.array([[0.0010, 0, 0], [-0.0004, 0.0013, 0], [0.0001, 0.0003, 0.0022]])
    f = np.empty((n_months, 3))
    f[0] = mean
    for t in range(1, n_months):
        f[t] = mean + phi @ (f[t - 1] - mean) + chol @ rng.standard_normal(3)
    load = nelson_siegel_loadings(tau, lam)
    y = f @ load.T + 0.0005 * rng.standard_normal((n_months, tau.size))
    dates = np.arange(n_months)
    return YieldPanel(dates, tau, y), f


def derive_scheme(reference, seed=7, scale=0.015, phi=0.7, sigma=1.5):
    rng = np.random.default_rng(seed)
    ages, years = reference.ages, reference.years
    s_alpha = -0.18 + 0.0045 * (ages - ages[0])
    s_beta = np.full(ages.size, 1.0 / ages.size)
    k = np.empty(years.size)
    k[0] = sigma / np.sqrt(1 - phi**2) * rng.standard_normal()
    for t in range(1, years.size):
        k[t] = phi * k[t - 1] + sigma * rng.standard_normal()
    log_m = reference.log_rates + s_alpha[:, None] + np.outer(s_beta, k)
    expo = reference.exposures * scale
    deaths = rng.poisson(expo * np.exp(log_m))
    return MortalityData(
        ages, years, deaths, expo, "scheme", {"source": "derived", "spread_kappa": k}
    )
