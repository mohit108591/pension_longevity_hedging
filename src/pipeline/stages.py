import pickle
import time

import numpy as np
import pandas as pd

from ..data.containers import YieldPanel
from ..data.hmd import load_csv_pair, load_hmd, load_nominal_curve_excel
from ..data.synthetic import derive_scheme, make_two_populations, make_yield_panel
from ..filtering.particle import (
    estimate_loglik_variance,
    particle_filter,
    tune_particle_count,
)
from ..filtering.smoothing import backward_simulation
from ..hedging import durations as dur
from ..hedging import effectiveness as eff
from ..hedging import optimizers as opt
from ..inference.diagnostics import split_rhat, summarise_chain
from ..inference.pgas import ParticleGibbs
from ..inference.pmmh import PMMH, ChainResult, KalmanMH
from ..instruments.longevity import (
    IndexLongevitySwap,
    QForward,
    SForward,
    ZeroCouponBond,
)
from ..liabilities.lifetables import (
    cohort_life_expectancy,
    cohort_survival,
    extend_old_ages,
)
from ..liabilities.scheme import PensionScheme
from ..mortality.backtest import backtest, summarise_backtest
from ..mortality.cbd import CairnsBlakeDowd
from ..mortality.lee_carter import LeeCarter
from ..mortality.multipopulation import LiLee, RelativeSpreadModel
from ..mortality.renshaw_haberman import RenshawHaberman
from ..rates.hull_white import HullWhite
from ..rates.nelson_siegel import DynamicNelsonSiegel
from ..risk import scenarios as sc
from ..statespace.lc_linear import LinearLeeCarterSSM
from ..statespace.lc_sv import PoissonLeeCarterSV
from ..utils import plotting as viz

MAIN = "LC-SV posterior"


class Context(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rng(cfg, offset):
    return np.random.default_rng(cfg.seed + offset)


def stage_data(cfg, ctx, out):
    d = cfg.data
    if d.source == "synthetic":
        ref, sch, truth = make_two_populations(
            tuple(d.ages), tuple(d.years), seed=cfg.seed
        )
    else:
        ref = load_hmd(
            d.hmd_dir,
            d.sex,
            tuple(d.ages),
            tuple(d.years),
            label="reference",
            deaths_path=d.get("deaths_path"),
            exposures_path=d.get("exposures_path"),
            population=d.get("population"),
        )
        if d.get("scheme_deaths_csv"):
            sch = load_csv_pair(d.scheme_deaths_csv, d.scheme_exposures_csv).subset(
                ref.ages, ref.years
            )
        else:
            sch = derive_scheme(ref, cfg.seed)
        truth = None
    r = cfg.rates
    if r.source == "synthetic":
        panel, _ = make_yield_panel(seed=cfg.seed + 1)
    elif r.source == "excel":
        dates, maturities, yields = load_nominal_curve_excel(
            r.yields_xlsx, r.get("yield_sheet", "4. spot curve")
        )
        panel = YieldPanel(dates, maturities, yields / r.get("yield_scale", 100.0))
    else:
        df = pd.read_csv(r.yields_csv, index_col=0)
        panel = YieldPanel(
            df.index.to_numpy(),
            df.columns.astype(float).to_numpy(),
            df.to_numpy(float) / r.get("yield_scale", 1.0),
        )
    ctx.update(ref=ref, sch=sch, truth=truth, panel=panel, ages=ref.ages)
    out.record(
        "data",
        {"reference": repr(ref), "scheme": repr(sch), "yield_obs": panel.yields.shape},
    )
    log(f"data: {ref} | {sch}")


def stage_classical(cfg, ctx, out):
    ref, sch = ctx.ref, ctx.sch
    models = {
        "LC-SVD": LeeCarter("svd"),
        "LC": LeeCarter("poisson"),
        "M5": CairnsBlakeDowd(),
        "M7": CairnsBlakeDowd(quadratic=True, cohort=True),
        "RH-H1": RenshawHaberman(),
    }
    rows = []
    for key, m in models.items():
        m.fit(ref)
        m.name = key
        rows.append(m.summary())
    table = pd.DataFrame(rows).sort_values("BIC", ascending=False)
    out.table(table, "classical_model_fit")
    out.record("classical_fit", table)
    out.figure(viz.residual_heatmap(models["LC"]), "residuals_lc")
    out.figure(viz.residual_heatmap(models["M7"]), "residuals_m7")
    c = cfg.classical
    factories = {
        "LC": lambda: LeeCarter("poisson"),
        "M7": lambda: CairnsBlakeDowd(True, True),
        "RH-H1": lambda: RenshawHaberman(),
    }
    bts = []
    for key, f in factories.items():

        def named(f=f, key=key):
            m = f()
            m.name = key
            return m

        bts.append(
            backtest(
                named,
                ref,
                c.backtest_origins,
                c.backtest_horizon,
                c.backtest_sims,
                cfg.seed,
            )
        )
    bt = summarise_backtest(pd.concat(bts))
    out.table(bt, "backtest")
    out.figure(viz.backtest_plot(bt), "backtest")
    lcm = models["LC"]
    ll = LiLee().fit([ref, sch])
    spread = RelativeSpreadModel().fit(sch, lcm.fitted_log_rates())
    li = pd.DataFrame(
        [
            {"population": p.label, **ll.explained_variance(i)}
            for i, p in enumerate([ref, sch])
        ]
    )
    out.table(li, "li_lee_explained_variance")
    out.record(
        "spread_model",
        {
            "phi": spread.ts.phi,
            "sigma": spread.ts.sigma,
            "alpha_range": [float(spread.alpha.min()), float(spread.alpha.max())],
        },
    )
    ctx.update(lc=lcm, m7=models["M7"], rh=models["RH-H1"], spread=spread)
    log(
        f"classical models fitted; best BIC = {table.iloc[0]['model']}; spread AR(1) phi={spread.ts.phi:.3f}"
    )


def stage_linear_ssm(cfg, ctx, out):
    lc, ref = ctx.lc, ctx.ref
    c = cfg.linear_ssm
    boot = LinearLeeCarterSSM(ref, lc.a, lc.b, lc.k[0], adapted=False)
    adap = LinearLeeCarterSSM(ref, lc.a, lc.b, lc.k[0], adapted=True)
    theta, ll = adap.mle()
    rng = _rng(cfg, 10)
    rows = [{"filter": "Kalman (exact)", "particles": 0, "mean": ll, "sd": 0.0}]
    for name, model, aux in [
        ("bootstrap", boot, False),
        ("fully adapted", adap, False),
        ("fully adapted APF", adap, True),
    ]:
        for N in c.particle_grid:
            m, s = estimate_loglik_variance(
                model, theta, N, c.loglik_reps, rng, auxiliary=aux
            )
            rows.append({"filter": name, "particles": N, "mean": m, "sd": s})
    pf = pd.DataFrame(rows)
    out.table(pf, "linear_pf_vs_kalman")
    space = adap.default_space()
    exact = KalmanMH(adap, space, c.mh_iter, _rng(cfg, 11)).run(theta)
    pm = PMMH(
        adap,
        space,
        c.pmmh_particles,
        c.mh_iter,
        _rng(cfg, 12),
        auxiliary=True,
        store_trajectories=False,
        label="PMMH (adapted APF)",
    ).run(theta)
    b = c.burn_in
    tab = pd.concat([summarise_chain(exact, b), summarise_chain(pm, b)])
    out.table(tab, "linear_pmmh_vs_exact")
    out.record("linear_mle", {"theta": theta, "loglik": ll})
    out.record(
        "linear_pmmh_acceptance",
        {"exact": exact.acceptance_rate, "pmmh": pm.acceptance_rate},
    )
    log(
        f"linear SSM: MLE {theta}; PMMH acc {pm.acceptance_rate:.2f} vs exact {exact.acceptance_rate:.2f}"
    )


def stage_particle_diagnostics(cfg, ctx, out):
    ref, lc = ctx.ref, ctx.lc
    model = PoissonLeeCarterSV(ref, lc.a, lc.b, lc.k, jumps=cfg.sv.jumps)
    theta = model.space().initial()
    theta["mu"] = float(lc.ts.drift[0])
    c = cfg.sv
    rng = _rng(cfg, 20)
    rows = []
    for name, kw in [
        ("guided / systematic", {}),
        ("guided / multinomial", {"resampling": "multinomial"}),
        ("guided APF", {"auxiliary": True}),
        (
            "guided / sorted, always resample",
            {"ess_threshold": 1.0, "sort_for_correlation": True},
        ),
    ]:
        for N in c.particle_grid:
            t0 = time.perf_counter()
            m, s = estimate_loglik_variance(model, theta, N, c.loglik_reps, rng, **kw)
            rows.append(
                {
                    "filter": name,
                    "particles": N,
                    "mean": m,
                    "sd": s,
                    "ms_per_run": 1000 * (time.perf_counter() - t0) / c.loglik_reps,
                }
            )
    df = pd.DataFrame(rows)
    out.table(df, "sv_loglik_variance")
    out.figure(viz.loglik_sd(df), "sv_loglik_variance")
    n_star, hist = tune_particle_count(
        model, theta, rng, target_sd=1.2, start=8, reps=c.loglik_reps
    )
    out.record("particle_tuning", {"target_sd": 1.2, "chosen_N": n_star, "path": hist})
    ctx.sv_model = model
    log(f"particle diagnostics done; N* for sd<=1.2 is {n_star}")


def _combine(chains, burn, label):
    parts = [c.thin(burn) for c in chains]
    return ChainResult(
        parts[0].names,
        np.vstack([p.samples for p in parts]),
        np.concatenate([p.loglik for p in parts]),
        np.concatenate([p.accepted for p in parts]),
        np.vstack([p.terminal_states for p in parts]),
        np.vstack([p.trajectories for p in parts]),
        sum(p.elapsed for p in parts),
        label,
    )


def _sv_setup(cfg, ctx):
    if "sv_model" not in ctx:
        lc = ctx.lc
        ctx.sv_model = PoissonLeeCarterSV(ctx.ref, lc.a, lc.b, lc.k, jumps=cfg.sv.jumps)
    model = ctx.sv_model
    space = model.space()
    init = space.initial()
    init["mu"] = float(ctx.lc.ts.drift[0])
    return model, space, init


def stage_sv_pmmh(cfg, ctx, out):
    model, space, init = _sv_setup(cfg, ctx)
    c = cfg.sv.pmmh
    ctx.chain_pmmh = PMMH(
        model,
        space,
        c.particles,
        c.iterations,
        _rng(cfg, 30),
        verbose=c.iterations // 4,
    ).run(init)


def stage_sv_cpm(cfg, ctx, out):
    model, space, init = _sv_setup(cfg, ctx)
    c = cfg.sv.cpm
    ctx.chain_cpm = PMMH(
        model,
        space,
        c.particles,
        c.iterations,
        _rng(cfg, 31),
        correlated=True,
        rho=c.rho,
        verbose=c.iterations // 4,
    ).run(init)


def stage_sv_pgas(cfg, ctx, out):
    model, space, init = _sv_setup(cfg, ctx)
    c = cfg.sv.pgas
    chains = []
    for i in range(c.chains):
        rng = _rng(cfg, 40 + i)
        start = dict(init)
        start["mu"] += 0.2 * rng.standard_normal()
        start["phi"] = float(
            np.clip(start["phi"] + 0.05 * rng.standard_normal(), 0.5, 0.98)
        )
        log(f"PGAS chain {i + 1}/{c.chains}")
        chains.append(
            ParticleGibbs(
                model, space, c.particles, c.iterations, rng, verbose=c.iterations // 2
            ).run(start)
        )
    ctx.chains_pgas = chains


def stage_sv_posterior(cfg, ctx, out):
    model, space, _ = _sv_setup(cfg, ctx)
    c = cfg.sv
    pm, cpm, pg_chains = ctx.chain_pmmh, ctx.chain_cpm, ctx.chains_pgas
    diag = pd.concat(
        [
            summarise_chain(pm, c.pmmh.burn_in),
            summarise_chain(cpm, c.cpm.burn_in),
            summarise_chain(pg_chains[0], c.pgas.burn_in, chains=pg_chains),
        ]
    )
    diag["acceptance"] = diag["sampler"].map(
        {pm.label: pm.acceptance_rate, cpm.label: cpm.acceptance_rate}
    )
    diag["elapsed_sec"] = diag["sampler"].map(
        {
            pm.label: pm.elapsed,
            cpm.label: cpm.elapsed,
            pg_chains[0].label: pg_chains[0].elapsed,
        }
    )
    out.table(diag, "sv_sampler_diagnostics")
    params = ["mu", "phi", "sigma_h"] + (["jump_p", "jump_mu"] if model.jumps else [])
    out.figure(viz.traces([pm, cpm, pg_chains[0]], params), "sv_traces")
    post_pg = _combine(pg_chains, c.pgas.burn_in, "PGAS (pooled)")
    out.figure(
        viz.posterior_densities(
            [pm.thin(c.pmmh.burn_in), cpm.thin(c.cpm.burn_in), post_pg], params
        ),
        "sv_posteriors",
    )
    source = {
        "pgas": post_pg,
        "pmmh": pm.thin(c.pmmh.burn_in),
        "cpm": cpm.thin(c.cpm.burn_in),
    }[c.posterior_source]
    years = ctx.ref.years
    truth = ctx.truth
    paths = post_pg.trajectories
    out.figure(
        viz.kappa_panel(
            years, model.kappa_hat, paths, None if truth is None else truth.kappa
        ),
        "sv_kappa",
    )
    out.figure(
        viz.volatility_panel(
            years,
            paths,
            None if truth is None else truth.logvol,
            None if truth is None else truth.jumps,
        ),
        "sv_volatility",
    )
    theta_bar = source.posterior_mean()
    fo = particle_filter(model, theta_bar, c.smoother_particles, _rng(cfg, 50))
    ffbsi = backward_simulation(model, theta_bar, fo, c.smoother_paths, _rng(cfg, 51))
    jump_prob = paths[:, :, 2].mean(0)
    rec = {
        "posterior_mean": theta_bar,
        "posterior_source": source.label,
        "rhat": {
            p: split_rhat(
                np.vstack([ch.samples[c.pgas.burn_in :, j] for ch in pg_chains])
            )
            for j, p in enumerate(space.names)
        },
        "ffbsi_vs_pgas_kappa_rmse": float(
            np.sqrt(np.mean((ffbsi[:, :, 0].mean(0) - paths[:, :, 0].mean(0)) ** 2))
        ),
        "detected_jump_years": years[jump_prob > 0.5].tolist(),
        "jump_probability_top5": {
            int(years[i]): float(jump_prob[i]) for i in np.argsort(jump_prob)[::-1][:5]
        },
        "elapsed_sec": {
            "pmmh": pm.elapsed,
            "cpm": cpm.elapsed,
            "pgas": sum(ch.elapsed for ch in pg_chains),
        },
    }
    if truth is not None:
        rec["true_jump_years"] = years[truth.jumps].tolist()
        rec["truth"] = truth.params["sv"]
        rec["logvol_corr_with_truth"] = float(
            np.corrcoef(paths[:, :, 1].mean(0), truth.logvol)[0, 1]
        )
        rec["kappa_rmse_vs_truth"] = float(
            np.sqrt(np.mean((paths[:, :, 0].mean(0) - truth.kappa) ** 2))
        )
    out.record("sv_inference", rec)
    ctx.update(posterior=source, theta_bar=theta_bar)
    for k in ("chain_pmmh", "chain_cpm", "chains_pgas"):
        ctx.pop(k)
    log(f"SV posterior mean: { {k: round(v, 3) for k, v in theta_bar.items()} }")


def stage_rates(cfg, ctx, out):
    dns = DynamicNelsonSiegel(dt=cfg.rates.dt).fit(
        ctx.panel, method=cfg.rates.dns_method
    )
    curve0 = dns.current_curve()
    hw = HullWhite(curve0).calibrate_historical(ctx.panel.yields[:, 0], cfg.rates.dt)
    out.figure(viz.yield_factors(dns.factors, dns.two_step_factors), "dns_factors")
    tab = pd.DataFrame(
        {
            "parameter": [
                "lambda",
                "mu_level",
                "mu_slope",
                "mu_curv",
                "phi_level",
                "phi_slope",
                "phi_curv",
                "obs_sd",
                "hw_a",
                "hw_sigma",
            ],
            "value": [dns.lam, *dns.mu, *np.diag(dns.Phi), dns.h, hw.a, hw.sigma],
        }
    )
    out.table(tab, "rate_models")
    out.record(
        "rates",
        {
            "dns_loglik": dns.loglik,
            "converged": dns.converged,
            "zero_curve_t0": dict(
                zip([1, 5, 10, 20, 30], curve0.zero([1, 5, 10, 20, 30]).tolist())
            ),
        },
    )
    ctx.update(dns=dns, hw=hw, curve0=curve0)
    log(f"rates: DNS lambda={dns.lam:.3f}, HW a={hw.a:.3f} sigma={hw.sigma:.4f}")


def _make_sets(cfg, ctx, which):
    h = cfg.hedging
    tau, H, n = h.horizon, h.projection_years, h.n_sims
    model = ctx.sv_model
    rng = lambda k: _rng(cfg, 100 + k)
    for name in which:
        if name == MAIN and "main_ref" in ctx:
            yield name, ctx.main_ref, ctx.main_sch
            continue
        if name == MAIN:
            ref = sc.sv_scenarios(model, ctx.posterior, n, H, tau, rng(0))
        elif name == "LC-SV fixed theta":
            ref = sc.sv_scenarios(
                model,
                ctx.posterior,
                n,
                H,
                tau,
                rng(0),
                fixed_theta=ctx.theta_bar,
                label=name,
            )
        elif name == "LC-SV no recalibration":
            ref = sc.sv_scenarios(
                model, ctx.posterior, n, H, tau, rng(0), recalibrate=False, label=name
            )
        elif name == "LC":
            ref = sc.classical_scenarios(ctx.lc, n, H, tau, rng(1), label=name)
        elif name == "M7":
            ref = sc.classical_scenarios(ctx.m7, n, H, tau, rng(2), label=name)
        elif name == "RH-H1":
            ref = sc.classical_scenarios(ctx.rh, n, H, tau, rng(3), label=name)
        else:
            raise KeyError(name)
        sch = sc.scheme_scenarios(ref, ctx.spread, rng(9), tau)
        yield name, ref, sch


def _instruments(cfg, ctx, ref_main, sch_main):
    h = cfg.hedging
    tau, ages = h.horizon, ctx.ages
    e_ref = ctx.ref.exposures[:, -1]
    e_sch = ctx.sch.exposures[:, -1]
    ex = lambda e, a: float(e[ages == a][0])
    index = [QForward(a, tau, exposure=ex(e_ref, a), seed=cfg.seed) for a in h.key_ages]
    index += [SForward(a, tau) for a in h.sforward_ages] + [
        IndexLongevitySwap(h.swap_age, h.swap_term)
    ]
    cust_obs = [
        QForward(a, tau, exposure=ex(e_sch, a), seed=cfg.seed + 1, tag="*")
        for a in h.key_ages
    ]
    cust_true = [QForward(a, tau, tag="#") for a in h.key_ages]
    lam = h.wang_lambda
    sc.price_instruments(index, ref_main.stochastic, ages, lam)
    sc.price_instruments(cust_obs + cust_true, sch_main.stochastic, ages, lam)
    return index, cust_obs + cust_true


def stage_scenarios_and_liability(cfg, ctx, out):
    h = cfg.hedging
    H = h.projection_years
    scheme = PensionScheme.synthetic(
        cfg.seed, cfg.scheme.pensioners, cfg.scheme.deferred, cfg.scheme.escalation
    )
    out.table(scheme.summary().reset_index(), "scheme_membership")
    _, ref, sch = next(_make_sets(cfg, ctx, [MAIN]))
    ages, curve0 = ctx.ages, ctx.curve0
    out.figure(
        viz.fan_chart(
            ctx.ref.years, ctx.ref.log_rates, ref.stochastic, ages, [65, 75, 85]
        ),
        "fan_chart",
    )
    ext, ea = extend_old_ages(ref.stochastic, ages)
    le65 = cohort_life_expectancy(cohort_survival(ext, ea, [65]))[:, 0]
    dfs = curve0.df(np.arange(1, H + 1))
    be = sch.stochastic.mean(0)
    cf = scheme.expected_cashflows(be, ages)
    L0 = float(cf @ dfs)
    pv_dist = scheme.present_value(sch.stochastic, ages, dfs)
    kqd = dur.key_q_durations(
        lambda lm: float(scheme.present_value(lm, ages, dfs)), be, ages, h.key_ages
    )
    krd = dur.key_rate_durations(
        lambda c: float(
            scheme.expected_cashflows(be, ages) @ c.df(np.arange(1, H + 1))
        ),
        curve0,
        h.key_rate_tenors,
    )
    out.figure(
        viz.bars(
            h.key_ages,
            kqd / 1e3,
            "Key q-durations (change in PV per unit q, thousands)",
            "m per unit q / 1000",
        ),
        "kqd",
    )
    out.figure(
        viz.bars(
            h.key_rate_tenors, -krd * 1e-4, "Key-rate DV01 of liability", "m per bp"
        ),
        "krd",
    )
    out.record(
        "liability",
        {
            "L0_best_estimate_m": L0,
            "macaulay_duration": scheme.duration(cf, dfs),
            "pv_sd_m": float(pv_dist.std()),
            "pv_99.5_minus_mean_m": float(np.quantile(pv_dist, 0.995) - pv_dist.mean()),
            "cohort_e65_mean": float(le65.mean()),
            "cohort_e65_90pct": np.quantile(le65, [0.05, 0.95]).tolist(),
            "kqd": dict(zip(map(str, h.key_ages), kqd.tolist())),
            "krd_dv01_per_bp": dict(
                zip(map(str, h.key_rate_tenors), (-krd * 1e-4).tolist())
            ),
        },
    )
    index, custom = _instruments(cfg, ctx, ref, sch)
    fixed = pd.DataFrame(
        [
            {"instrument": i.name, "fixed_rate": getattr(i, "fixed", np.nan)}
            for i in index + custom
            if hasattr(i, "fixed")
        ]
    )
    out.table(fixed, "instrument_fixed_rates")
    ctx.update(
        scheme=scheme,
        index=index,
        custom=custom,
        kqd=kqd,
        krd=krd,
        main_ref=ref,
        main_sch=sch,
        L0=L0,
    )
    log(
        f"liability L0={L0:,.1f}m, duration={scheme.duration(cf, dfs):.2f}, e65={le65.mean():.2f}"
    )


def _problem(cfg, ctx, name, ref, sch, idio=True, bonds=(), rates=None):
    h = cfg.hedging
    tau, H = h.horizon, h.projection_years
    if rates is None:
        acc, after, df_fn = sc.deterministic_discount(ctx.curve0, tau, H)
    else:
        acc, after, df_fn = rates
    idio_rng = _rng(cfg, 300) if (idio and h.idiosyncratic) else None
    return sc.build_problem(
        name,
        ctx.scheme,
        ctx.ages,
        tau,
        ref,
        ctx.index,
        acc,
        after,
        ctx.curve0,
        sch=sch,
        custom=ctx.custom,
        bonds=bonds,
        df_fn=df_fn,
        idio_rng=idio_rng,
    )


def stage_longevity_hedging(cfg, ctx, out):
    h = cfg.hedging
    ages, tau = ctx.ages, h.horizon
    prob = _problem(cfg, ctx, MAIN, ctx.main_ref, ctx.main_sch)
    qn = [f"qF{a}" for a in h.key_ages]
    L, Xq = prob.subset(qn)
    kqd_q = [
        dur.qforward_kqd(q, 1.0, ages, h.key_ages) for q in ctx.index[: len(h.key_ages)]
    ]
    strategies = {}
    strategies["q-fwd min-var"] = (qn, opt.min_variance(L, Xq, h.ridge))
    strategies["q-fwd min-CVaR"] = (
        qn,
        opt.min_cvar(L, Xq, h.cvar_beta, max_gross=h.max_gross)[0],
    )
    strategies["q-fwd mean-var"] = (qn, opt.mean_variance(L, Xq, h.risk_aversion))
    strategies["q-fwd key-q-duration"] = (
        qn,
        dur.kqd_hedge(ctx.kqd / ctx.curve0.df(tau), kqd_q),
    )
    best = max(
        qn,
        key=lambda n: eff.hedge_effectiveness(
            L, L - prob.subset([n])[1] @ opt.min_variance(*prob.subset([n]))
        )["HE_variance"],
    )
    strategies[f"single {best}"] = ([best], opt.min_variance(*prob.subset([best])))
    sn = [f"SF{a}" for a in h.sforward_ages]
    strategies["S-fwd min-var"] = (sn, opt.min_variance(*prob.subset(sn), h.ridge))
    ln = [f"LS{h.swap_age}"]
    strategies["index swap min-var"] = (ln, opt.min_variance(*prob.subset(ln)))
    cn = [f"qF{a}*" for a in h.key_ages]
    strategies["custom q-fwd (observed)"] = (
        cn,
        opt.min_variance(*prob.subset(cn), h.ridge),
    )
    tn = [f"qF{a}#" for a in h.key_ages]
    strategies["custom q-fwd (true rates)"] = (
        tn,
        opt.min_variance(*prob.subset(tn), h.ridge),
    )
    rows, hedged = [], {}
    for name, (names, w) in strategies.items():
        _, X = prob.subset(names)
        res = L - X @ w
        rows.append(
            {
                "strategy": name,
                "instruments": ",".join(names),
                **eff.hedge_effectiveness(L, res, h.alpha),
                "gross_notional_m": float(np.abs(w).sum()),
            }
        )
        hedged[name] = res
    table = pd.DataFrame(rows)
    out.table(table, "longevity_hedge_effectiveness")
    out.figure(
        viz.hedged_distributions(
            L,
            {
                k: hedged[k]
                for k in [
                    "q-fwd min-var",
                    "index swap min-var",
                    "custom q-fwd (observed)",
                ]
            },
        ),
        "hedged_distributions",
    )
    he = table.set_index("strategy")["HE_variance"]
    basis = eff.population_basis_risk(
        he["q-fwd min-var"], he["custom q-fwd (true rates)"]
    )
    basis["HE_index_qforward"] = basis["HE_index"]
    basis["HE_index_swap"] = float(he["index swap min-var"])
    basis["sampling_risk_in_custom_index"] = (
        he["custom q-fwd (true rates)"] - he["custom q-fwd (observed)"]
    )
    no_idio = _problem(cfg, ctx, MAIN, ctx.main_ref, ctx.main_sch, idio=False)
    Ln, Xn = no_idio.subset(qn)
    wn = opt.min_variance(Ln, Xn, h.ridge)
    basis["HE_without_scheme_sampling_risk"] = eff.hedge_effectiveness(
        Ln, Ln - Xn @ wn
    )["HE_variance"]
    basis["L_sd_with_sampling"] = float(L.std())
    basis["L_sd_without_sampling"] = float(Ln.std())
    out.record("basis_risk", basis)
    lam_rows = []
    w = strategies["q-fwd min-var"][1]
    for lam in h.lambda_grid:
        fixed = [
            QForward(q.age, q.maturity, exposure=q.exposure, seed=q.seed).price(
                ctx.main_ref.stochastic, ages, lam
            )
            for q in ctx.index[: len(qn)]
        ]
        Xl = np.column_stack([f.payoff(ctx.main_ref.at_horizon, ages) for f in fixed])
        lam_rows.append(
            {
                "lambda": lam,
                "expected_hedge_cost_m": float(-(Xl @ w).mean()),
                "cost_bp_of_L0": float(
                    -(Xl @ w).mean() / ctx.L0 * 1e4 * ctx.curve0.df(tau)
                ),
            }
        )
    out.table(pd.DataFrame(lam_rows), "risk_premium_sensitivity")
    ctx.update(strategies=strategies, main_problem=prob)
    log("longevity hedging: " + ", ".join(f"{k}={v:.3f}" for k, v in he.items()))


def stage_model_risk(cfg, ctx, out):
    h = cfg.hedging
    qn = [f"qF{a}" for a in h.key_ages]
    sets = [MAIN, "LC-SV fixed theta", "LC-SV no recalibration", "LC", "M7", "RH-H1"]
    probs, spread_rows = {}, []
    for name, ref, sch in _make_sets(cfg, ctx, sets):
        p = _problem(cfg, ctx, name, ref, sch)
        L, X = p.subset(qn)
        probs[name] = (L, X)
        w = opt.min_variance(L, X, h.ridge)
        spread_rows.append(
            {
                "scenario_set": name,
                "L_mean": L.mean(),
                "L_sd": L.std(),
                "L_sd_pct": 100 * L.std() / L.mean(),
                "ES99.5_centered": eff.var_es(L - L.mean(), h.alpha)[1],
                "HE_own_minvar": eff.hedge_effectiveness(L, L - X @ w)["HE_variance"],
            }
        )
        log(f"model-risk set {name}: sd={L.std():.2f}")
    out.table(pd.DataFrame(spread_rows), "scenario_set_risk")
    models = [MAIN, "LC", "M7", "RH-H1"]
    hedges = {f"min-var | {m}": opt.min_variance(*probs[m], h.ridge) for m in models}
    hedges["robust minimax"] = opt.robust_min_variance(
        [probs[m] for m in models], h0=hedges[f"min-var | {MAIN}"]
    )[0]
    hedges["model average"] = opt.bayesian_average_variance([probs[m] for m in models])
    M = np.array(
        [
            [
                eff.hedge_effectiveness(probs[m][0], probs[m][0] - probs[m][1] @ w)[
                    "HE_variance"
                ]
                for m in models
            ]
            for w in hedges.values()
        ]
    )
    mat = pd.DataFrame(M, index=list(hedges), columns=models)
    mat["worst_case"] = mat.min(1)
    out.table(mat.reset_index().rename(columns={"index": "hedge"}), "model_risk_matrix")
    out.figure(
        viz.heatmap(
            M,
            list(hedges),
            models,
            "Variance-based hedge effectiveness under model risk",
        ),
        "model_risk",
    )
    out.record("model_risk_worst_case", mat["worst_case"].to_dict())
    log(
        "model risk: worst-case HE "
        + ", ".join(f"{k}={v:.3f}" for k, v in mat["worst_case"].items())
    )


def stage_integrated(cfg, ctx, out):
    h = cfg.hedging
    tau, H, n = h.horizon, h.projection_years, h.n_sims
    bonds = [ZeroCouponBond(T) for T in h.zcb_maturities]
    qn = [f"qF{a}" for a in h.key_ages]
    bn = [b.name for b in bonds]
    krd_h, _ = dur.key_rate_hedge(
        ctx.krd, ctx.curve0, h.key_rate_tenors, h.zcb_maturities
    )
    rows = []
    engines = {
        "DNS": lambda: sc.dns_discount(ctx.dns, ctx.curve0, tau, H, n, _rng(cfg, 400))[
            :3
        ],
        "Hull-White": lambda: sc.hull_white_discount(
            ctx.hw, ctx.curve0, tau, H, n, _rng(cfg, 401)
        )[:3],
    }
    for eng, make in engines.items():
        rates = make()
        p = _problem(
            cfg,
            ctx,
            f"{MAIN} + {eng}",
            ctx.main_ref,
            ctx.main_sch,
            bonds=bonds,
            rates=rates,
        )
        L = p.L
        _, Xq = p.subset(qn)
        _, Xb = p.subset(bn)
        _, Xj = p.subset(bn + qn)
        wq = ctx.strategies["q-fwd min-var"][1]
        wb = opt.min_variance(L, Xb, h.ridge)
        wj = opt.min_variance(L, Xj, h.ridge)
        wc, _ = opt.min_cvar(L, Xj, h.cvar_beta, max_gross=h.max_gross)
        cand = {
            "longevity only (q-fwd)": Xq @ wq,
            "rates only (key-rate ZCB)": Xb @ krd_h,
            "rates only (min-var ZCB)": Xb @ wb,
            "joint min-var": Xj @ wj,
            "joint min-CVaR": Xj @ wc,
        }
        for name, pay in cand.items():
            rows.append(
                {
                    "rate_model": eng,
                    "strategy": name,
                    **eff.hedge_effectiveness(L, L - pay, h.alpha),
                }
            )
        dec = eff.decompose_residual(
            L, {"interest rates": Xb @ wb, "longevity (index)": Xq @ wq}
        )
        out.record(f"risk_decomposition_{eng}", dec)
        out.record(f"L_sd_integrated_{eng}", float(L.std()))
        log(
            f"integrated ({eng}): L sd={L.std():.1f}, decomposition={ {k: round(v, 3) for k, v in dec.items()} }"
        )
    out.record("krd_zcb_notionals", dict(zip(bn, krd_h.tolist())))
    out.table(pd.DataFrame(rows), "integrated_hedge_effectiveness")


STAGES = [
    stage_data,
    stage_classical,
    stage_linear_ssm,
    stage_particle_diagnostics,
    stage_sv_pmmh,
    stage_sv_cpm,
    stage_sv_pgas,
    stage_sv_posterior,
    stage_rates,
    stage_scenarios_and_liability,
    stage_longevity_hedging,
    stage_model_risk,
    stage_integrated,
]
OPTIONAL = {"linear_ssm", "particle_diagnostics", "model_risk", "integrated"}


def run_all(cfg, out, skip=(), resume=False, max_seconds=None):
    unknown = set(skip) - OPTIONAL
    if unknown:
        raise ValueError(f"only optional stages can be skipped: {sorted(OPTIONAL)}")
    ckpt = out.root / "checkpoint.pkl"
    ctx, done = Context(), []
    if resume and ckpt.exists():
        state = pickle.loads(ckpt.read_bytes())
        ctx, done = state["ctx"], state["done"]
        out.summary.update(state["summary"])
        log(f"resuming after: {', '.join(done)}")
    timings = out.summary.get("timings_sec", {})
    start = time.perf_counter()
    for stage in STAGES:
        name = stage.__name__.replace("stage_", "")
        if name in skip or name in done:
            continue
        if max_seconds and time.perf_counter() - start > max_seconds:
            log("time budget reached; rerun with --resume")
            return ctx, False
        t0 = time.perf_counter()
        stage(cfg, ctx, out)
        timings[name] = round(time.perf_counter() - t0, 2)
        done.append(name)
        out.record("timings_sec", timings)
        out.flush()
        ckpt.write_bytes(
            pickle.dumps({"ctx": ctx, "done": done, "summary": out.summary})
        )
    return ctx, True
