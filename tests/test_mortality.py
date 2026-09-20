import numpy as np

from src.mortality.cbd import CairnsBlakeDowd
from src.mortality.lee_carter import LeeCarter
from src.mortality.multipopulation import RelativeSpreadModel
from src.mortality.renshaw_haberman import RenshawHaberman


def test_lee_carter_identification(lee_carter):
    assert np.isclose(lee_carter.b.sum(), 1.0)
    assert abs(lee_carter.k.mean()) < 1e-8


def test_lee_carter_recovers_truth(populations, lee_carter):
    truth = populations[2]
    assert np.corrcoef(lee_carter.k, truth.kappa)[0, 1] > 0.999
    assert np.max(np.abs(lee_carter.b - truth.beta)) < 5e-3


def test_poisson_beats_svd(populations):
    ref = populations[0]
    assert (
        LeeCarter("poisson").fit(ref).loglik()
        >= LeeCarter("svd").fit(ref).loglik() - 1e-6
    )


def test_m7_cohort_constraints(populations):
    m = CairnsBlakeDowd(quadratic=True, cohort=True).fit(populations[0])
    c = m.cohorts[m.cvalid].astype(float)
    g = m.gamma[m.cvalid]
    cc = c - c.mean()
    assert abs(g.sum()) < 1e-6
    assert abs((cc * g).sum()) < 1e-4
    assert abs((cc**2 * g).sum()) < 1e-2


def test_simulations_are_finite(populations, rng):
    ref = populations[0]
    for m in (LeeCarter(), CairnsBlakeDowd(True, True), RenshawHaberman()):
        sims = m.fit(ref).simulate_log_rates(15, 50, rng)
        assert sims.shape == (50, ref.ages.size, 15)
        assert np.isfinite(sims).all()


def test_spread_model_tracks_truth(populations, lee_carter):
    _, sch, truth = populations
    sp = RelativeSpreadModel().fit(sch, lee_carter.fitted_log_rates())
    assert np.corrcoef(sp.kappa, truth.spread_kappa)[0, 1] > 0.9
    assert 0 < sp.ts.phi < 1
