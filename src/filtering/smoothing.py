import numpy as np
from scipy.special import logsumexp


def backward_simulation(model, theta, out, n_paths, rng):
    n_steps, N, dim = out.particles.shape
    paths = np.empty((n_paths, n_steps, dim))
    idx = rng.choice(N, size=n_paths, p=np.exp(out.logw[-1]))
    paths[:, -1] = out.particles[-1, idx]
    for t in range(n_steps - 2, -1, -1):
        xt = out.particles[t]
        for j in range(n_paths):
            lw = out.logw[t] + model.log_transition(theta, t + 1, xt, paths[j, t + 1])
            p = np.exp(lw - logsumexp(lw))
            paths[j, t] = xt[rng.choice(N, p=p)]
    return paths


def fixed_lag_means(out, lag):
    n_steps, N, dim = out.particles.shape
    means = np.empty((n_steps, dim))
    for s in range(n_steps):
        end = min(s + lag, n_steps - 1)
        idx = np.arange(N)
        for t in range(end, s, -1):
            idx = out.ancestors[t, idx]
        w = np.exp(out.logw[end])
        means[s] = w @ out.particles[s, idx]
    return means


def smoothed_summary(paths, qs=(0.05, 0.5, 0.95)):
    return {"mean": paths.mean(0), "quantiles": np.quantile(paths, qs, axis=0)}
