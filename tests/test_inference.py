import numpy as np
from scipy import stats

from src.inference.diagnostics import iact, split_rhat
from src.inference.parameters import (
    InvGammaOnScale,
    Parameter,
    ParameterSpace,
    ShiftedBeta,
)
from src.inference.pgas import ParticleGibbs
from src.inference.pmmh import PMMH
from src.statespace.lc_sv import PoissonLeeCarterSV


def test_transforms_roundtrip():
    space = ParameterSpace(
        [
            Parameter("a", stats.norm(), "identity", 0.3),
            Parameter("b", InvGammaOnScale(2, 1), "log", 0.7),
            Parameter("c", stats.beta(2, 2), "logit", 0.2),
            Parameter("d", ShiftedBeta(2, 2), "tanh", -0.4),
        ]
    )
    th = space.initial()
    back = space.constrain(space.unconstrain(th))
    assert all(np.isclose(th[k], back[k]) for k in th)


def test_log_jacobian_numerically():
    space = ParameterSpace(
        [
            Parameter("c", stats.beta(2, 2), "logit", 0.2),
            Parameter("d", ShiftedBeta(2, 2), "tanh", 0.4),
        ]
    )
    u = space.unconstrain(space.initial())
    eps = 1e-6
    num = 0.0
    for i, name in enumerate(space.names):
        up, dn = u.copy(), u.copy()
        up[i] += eps
        dn[i] -= eps
        num += np.log(
            (space.constrain(up)[name] - space.constrain(dn)[name]) / (2 * eps)
        )
    assert np.isclose(num, space.log_jacobian(u), atol=1e-5)


def test_inverse_gamma_scale_density_integrates():
    d = InvGammaOnScale(3.0, 0.3)
    grid = np.linspace(1e-3, 5, 20000)
    assert np.isclose(
        np.trapezoid(np.exp([d.logpdf(g) for g in grid]), grid), 1.0, atol=1e-3
    )


def test_diagnostics_on_ar1(rng):
    x = np.zeros(20000)
    for t in range(1, x.size):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    assert 14 < iact(x) < 24
    chains = rng.standard_normal((4, 2000))
    assert split_rhat(chains) < 1.01


def test_samplers_run(populations, lee_carter, rng):
    lc = lee_carter
    model = PoissonLeeCarterSV(populations[0], lc.a, lc.b, lc.k, jumps=False)
    space = model.space()
    for sampler in (
        PMMH(model, space, 16, 30, rng),
        PMMH(model, space, 16, 30, rng, correlated=True),
        ParticleGibbs(model, space, 16, 15, rng),
    ):
        res = sampler.run()
        assert np.isfinite(res.samples).all()
        assert res.trajectories.shape[1:] == (model.n_steps, 3)
