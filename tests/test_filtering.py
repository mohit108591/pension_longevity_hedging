import numpy as np

from src.filtering.kalman import ffbs, kalman_filter, rts_smoother
from src.filtering.particle import estimate_loglik_variance, particle_filter
from src.filtering.resampling import SCHEMES
from src.filtering.smoothing import backward_simulation
from src.statespace.lc_linear import LinearLeeCarterSSM
from src.statespace.lc_sv import PoissonLeeCarterSV

THETA = {"mu": -0.8, "sigma_eta": 1.0, "sigma_eps": 1.0}


def test_resampling_unbiased(rng):
    w = rng.dirichlet(np.ones(50))
    for name, f in SCHEMES.items():
        counts = np.zeros(50)
        for _ in range(400):
            counts += np.bincount(f(w, rng), minlength=50)
        assert np.allclose(counts / (400 * 50), w, atol=0.02), name


def test_adapted_pf_matches_kalman(populations, lee_carter, rng):
    lc = lee_carter
    model = LinearLeeCarterSSM(populations[0], lc.a, lc.b, lc.k[0], adapted=True)
    exact = model.loglik(THETA)
    mean, sd = estimate_loglik_variance(model, THETA, 64, 10, rng)
    assert abs(mean - exact) < 0.5
    assert sd < 0.5


def test_bootstrap_pf_is_noisier(populations, lee_carter, rng):
    lc = lee_carter
    boot = LinearLeeCarterSSM(populations[0], lc.a, lc.b, lc.k[0], adapted=False)
    adap = LinearLeeCarterSSM(populations[0], lc.a, lc.b, lc.k[0], adapted=True)
    assert (
        estimate_loglik_variance(boot, THETA, 64, 8, rng)[1]
        > estimate_loglik_variance(adap, THETA, 64, 8, rng)[1]
    )


def test_smoothers_consistent(populations, lee_carter, rng):
    lc = lee_carter
    model = LinearLeeCarterSSM(populations[0], lc.a, lc.b, lc.k[0])
    ssm = model.ssm(THETA)
    res = kalman_filter(model.y, ssm)
    a_s, _ = rts_smoother(res, ssm)
    draws = ffbs(res, ssm, rng, 300)
    assert np.allclose(draws.mean(0)[:, 0], a_s[:, 0], atol=0.3)


def test_sv_guided_filter(populations, lee_carter, rng):
    lc = lee_carter
    model = PoissonLeeCarterSV(populations[0], lc.a, lc.b, lc.k)
    theta = model.space().initial()
    out = particle_filter(model, theta, 128, rng)
    assert np.isfinite(out.loglik)
    assert np.corrcoef(out.filtering_mean()[:, 0], lc.k)[0, 1] > 0.999
    paths = backward_simulation(model, theta, out, 5, rng)
    assert paths.shape == (5, model.n_steps, 3)
