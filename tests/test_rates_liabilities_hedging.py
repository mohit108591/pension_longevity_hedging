import numpy as np

from src.data.synthetic import make_yield_panel
from src.hedging import optimizers as opt
from src.hedging.durations import triangular_weights
from src.hedging.effectiveness import hedge_effectiveness
from src.instruments.longevity import QForward, wang_expectation
from src.liabilities.lifetables import annuity_factor, cohort_survival, extend_old_ages
from src.liabilities.scheme import PensionScheme
from src.rates.curves import DiscountCurve
from src.rates.hull_white import HullWhite
from src.rates.nelson_siegel import DynamicNelsonSiegel

CURVE = DiscountCurve(
    np.arange(0.5, 80.5, 0.5),
    0.03 + 0.01 * (1 - np.exp(-np.arange(0.5, 80.5, 0.5) / 10)),
)


def test_hull_white_martingale(rng):
    hw = HullWhite(CURVE, 0.08, 0.012)
    r, bank = hw.simulate(8, 40000, rng)
    assert np.isclose(np.mean(1 / bank), CURVE.df(8), rtol=3e-3)
    P = hw.zcb(8, np.array([20.0]), r)[:, 0]
    assert np.isclose(np.mean(P / bank), CURVE.df(20), rtol=5e-3)


def test_dns_two_step():
    panel, f = make_yield_panel(n_months=120)
    dns = DynamicNelsonSiegel().fit(panel, method="two_step")
    assert dns.factors.shape == (120, 3)
    assert np.corrcoef(dns.factors[:, 0], f[:, 0])[0, 1] > 0.95


def test_zero_mortality_annuity():
    logm = np.full((61, 30), -50.0)
    surv = cohort_survival(logm, np.arange(60, 121), [60, 62])
    assert np.allclose(surv, 1.0)
    short = cohort_survival(np.full((5, 30), -50.0), np.arange(60, 65), [60])
    assert np.allclose(short[0, 6:], 0.0)
    df = CURVE.df(np.arange(1, 31))
    assert np.allclose(annuity_factor(surv, df), df.sum())


def test_survival_monotone(populations, lee_carter):
    ext, ea = extend_old_ages(lee_carter.central_log_rates(40), lee_carter.data.ages)
    s = cohort_survival(ext, ea, [60, 70, 80])
    assert np.all(np.diff(s, axis=-1) <= 1e-12)


def test_wang_transform():
    x = np.random.default_rng(0).normal(size=20000)
    assert np.isclose(wang_expectation(x, 0.0), x.mean(), atol=1e-12)
    assert np.isclose(wang_expectation(x, 0.3), 0.3, atol=0.03)


def test_min_variance_perfect_hedge(rng):
    X = rng.standard_normal((5000, 3))
    L = X @ np.array([2.0, -1.0, 0.5]) + 10
    h = opt.min_variance(L, X)
    assert np.allclose(h, [2.0, -1.0, 0.5])
    assert hedge_effectiveness(L, L - X @ h)["HE_variance"] > 0.999


def test_cvar_and_robust(rng):
    X = rng.standard_normal((2000, 2))
    L = X @ np.array([1.0, 0.5]) + 0.3 * rng.standard_normal(2000)
    h, _ = opt.min_cvar(L, X, 0.95)
    assert np.allclose(h, [1.0, 0.5], atol=0.1)
    sets = [(L, X), (L + 0.2 * X[:, 0], X)]
    hr, worst = opt.robust_min_variance(sets)
    assert 1.0 <= hr[0] <= 1.2 and worst < 0.2


def test_triangular_weights_partition():
    W = triangular_weights(np.arange(55, 96), [65, 70, 75, 80, 85])
    assert np.allclose(W.sum(0), 1.0)


def test_scheme_valuation_and_qforward(populations, lee_carter, rng):
    ref = populations[0]
    scheme = PensionScheme.synthetic(n_pensioners=500, n_deferred=200)
    sims = lee_carter.simulate_log_rates(40, 200, rng)
    acc = CURVE.df(np.arange(1, 6)) / CURVE.df(5)
    after = (CURVE.df(5 + np.arange(1, 36)) / CURVE.df(5))[None, :]
    v1 = scheme.value_at_horizon(sims, ref.ages, 5, acc, after)
    v2 = scheme.value_at_horizon(sims, ref.ages, 5, acc, after, rng=rng)
    assert v1.value.shape == (200,)
    assert np.isclose(v1.value.mean(), v2.value.mean(), rtol=0.02)
    assert v2.value.std() > v1.value.std()
    qf = QForward(75, 5).price(sims, ref.ages, 0.2)
    assert qf.fixed < QForward(75, 5).price(sims, ref.ages, 0.0).fixed
