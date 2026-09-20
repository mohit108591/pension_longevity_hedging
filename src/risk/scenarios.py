from dataclasses import dataclass, field

import numpy as np


@dataclass
class MortalityScenarios:
    label: str
    stochastic: np.ndarray
    at_horizon: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class HedgeProblem:
    label: str
    L: np.ndarray
    X: np.ndarray
    names: list
    L0: float
    extra: dict = field(default_factory=dict)

    def subset(self, names):
        idx = [self.names.index(n) for n in names]
        return self.L, self.X[:, idx]


def _continue_best_estimate(logm, tau, improvement):
    H = logm.shape[-1]
    steps = np.arange(1, H - tau + 1)
    out = logm.copy()
    base = logm[..., tau - 1]
    imp = improvement if improvement.ndim == 2 else improvement[None, :]
    out[..., tau:] = base[..., None] + imp[..., None] * steps
    return out


def recalibrated_improvement(stoch, first_fitted, n_hist, tau):
    return (stoch[..., tau - 1] - first_fitted[None, :]) / (n_hist - 1 + tau)


def sv_scenarios(
    model,
    chain,
    n_sims,
    horizon,
    tau,
    rng,
    fixed_theta=None,
    recalibrate=True,
    label="LC-SV posterior",
):
    draws = rng.integers(0, chain.samples.shape[0], n_sims)
    ks = np.empty((n_sims, horizon))
    drift = np.empty(n_sims)
    if fixed_theta is not None:
        theta = fixed_theta
        k, _ = model.simulate_forward(theta, chain.terminal_states[draws], horizon, rng)
        ks[:] = k
        drift[:] = model.effective_drift(theta)
    else:
        params = chain.as_dicts()
        for j, i in enumerate(draws):
            k, _ = model.simulate_forward(
                params[i], chain.terminal_states[i][None, :], horizon, rng
            )
            ks[j] = k[0]
            drift[j] = model.effective_drift(params[i])
    a, b = model.alpha, model.beta
    stoch = a[None, :, None] + b[None, :, None] * ks[:, None, :]
    if recalibrate:
        khat = model.kappa_hat
        drift = (ks[:, tau - 1] - khat[0]) / (khat.size - 1 + tau)
    at_h = _continue_best_estimate(stoch, tau, b[None, :] * drift[:, None])
    return MortalityScenarios(
        label, stoch, at_h, {"drift_at_horizon": drift, "kappa": ks}
    )


def classical_scenarios(model, n_sims, horizon, tau, rng, recalibrate=True, label=None):
    stoch = model.simulate_log_rates(horizon, n_sims, rng)
    if recalibrate:
        obs = model.data.log_rates
        improvement = recalibrated_improvement(stoch, obs[:, 0], obs.shape[1], tau)
    else:
        central = stoch.mean(0)
        improvement = (central[:, -1] - central[:, 0]) / (horizon - 1)
    at_h = _continue_best_estimate(stoch, tau, improvement)
    return MortalityScenarios(label or model.name, stoch, at_h)


def scheme_scenarios(reference, spread, rng, tau):
    n, _, H = reference.stochastic.shape
    kappa, spread_log = spread.simulate_spread(H, n, rng)
    stoch = reference.stochastic + spread_log
    k_h = kappa.copy()
    k_h[:, tau:] = (
        kappa[:, tau - 1 : tau] * spread.ts.phi ** np.arange(1, H - tau + 1)[None, :]
    )
    at_h = reference.at_horizon + spread.spread_log_rates(k_h)
    return MortalityScenarios(
        reference.label + " | scheme", stoch, at_h, {"spread_kappa": kappa}
    )


def deterministic_discount(curve0, tau, horizon):
    s = np.arange(1, horizon + 1)
    acc = curve0.df(s[:tau]) / curve0.df(tau)
    after = curve0.df(tau + s[: horizon - tau]) / curve0.df(tau)
    return (
        acc,
        after[None, :],
        lambda T: np.atleast_1d(curve0.df(tau + T) / curve0.df(tau)),
    )


def dns_discount(dns, curve0, tau, horizon, n_sims, rng):
    s = np.arange(1, horizon + 1)
    acc = curve0.df(s[:tau]) / curve0.df(tau)
    factors = dns.simulate_factors(tau, n_sims, rng)
    grid = s[: horizon - tau].astype(float)
    z = dns.zero_rates(factors, grid)
    after = np.exp(-z * grid[None, :])
    df_fn = lambda T: np.exp(
        -dns.zero_rates(factors, np.atleast_1d(float(T)))[:, 0] * T
    )
    return acc, after, df_fn, factors


def hull_white_discount(hw, curve0, tau, horizon, n_sims, rng):
    s = np.arange(1, horizon + 1)
    acc = curve0.df(s[:tau]) / curve0.df(tau)
    r, _ = hw.simulate(tau, n_sims, rng)
    grid = s[: horizon - tau].astype(float)
    after = hw.zcb(tau, tau + grid, r)
    df_fn = lambda T: hw.zcb(tau, tau + np.atleast_1d(float(T)), r)[:, 0]
    return acc, after, df_fn, r


def price_instruments(instruments, pricing_paths, ages, lam):
    for inst in instruments:
        inst.price(pricing_paths, ages, lam)
    return instruments


def instrument_matrix(instruments, paths, ages, tau, acc, after):
    cols = []
    for inst in instruments:
        if hasattr(inst, "payoff"):
            cols.append(inst.payoff(paths, ages))
        else:
            cols.append(inst.value_at_horizon(paths, ages, tau, acc, after))
    return np.column_stack(cols)


def build_problem(
    label,
    scheme,
    ages,
    tau,
    ref,
    instruments,
    acc,
    after,
    curve0,
    sch=None,
    custom=(),
    bonds=(),
    df_fn=None,
    idio_rng=None,
):
    sch = sch or ref
    val = scheme.value_at_horizon(sch.at_horizon, ages, tau, acc, after, rng=idio_rng)
    blocks = [instrument_matrix(instruments, ref.at_horizon, ages, tau, acc, after)]
    names = [i.name for i in instruments]
    if custom:
        blocks.append(instrument_matrix(custom, sch.at_horizon, ages, tau, acc, after))
        names += [i.name for i in custom]
    if bonds:
        blocks.append(
            np.column_stack([b.pnl_at_horizon(tau, curve0, df_fn) for b in bonds])
        )
        names += [b.name for b in bonds]
    H = sch.at_horizon.shape[-1]
    L0 = float(
        scheme.present_value(
            sch.stochastic.mean(0), ages, curve0.df(np.arange(1, H + 1))
        )
    )
    return HedgeProblem(
        label,
        val.value,
        np.column_stack(blocks),
        names,
        L0,
        {"paid": val.paid, "future": val.future, "survivors": val.survivors},
    )
