import numpy as np
import pandas as pd
from scipy import stats


def autocorrelation(x, max_lag=None):
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf = acf / acf[0] if acf[0] > 0 else np.zeros(n)
    return acf[: (max_lag or n)]


def iact(x, c=5.0):
    rho = autocorrelation(x)
    tau = 2 * np.cumsum(rho) - 1
    for m in range(1, rho.size):
        if m >= c * tau[m]:
            return float(max(tau[m], 1.0))
    return float(max(tau[-1], 1.0))


def ess(x):
    return len(x) / iact(x)


def _rank_normalise(chains):
    flat = chains.ravel()
    r = stats.rankdata(flat).reshape(chains.shape)
    return stats.norm.ppf((r - 0.375) / (flat.size + 0.25))


def split_rhat(chains):
    chains = np.atleast_2d(np.asarray(chains, float))
    n = chains.shape[1] // 2
    split = np.vstack([chains[:, :n], chains[:, n : 2 * n]])
    z = _rank_normalise(split)
    _, n = z.shape
    b = n * z.mean(1).var(ddof=1)
    w = z.var(1, ddof=1).mean()
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def geweke_z(x, first=0.1, last=0.5):
    x = np.asarray(x, float)
    a = x[: int(first * x.size)]
    b = x[int((1 - last) * x.size) :]
    sa = np.var(a, ddof=1) * iact(a) / a.size
    sb = np.var(b, ddof=1) * iact(b) / b.size
    return float((a.mean() - b.mean()) / np.sqrt(sa + sb))


def summarise_chain(result, burn_in=0, chains=None):
    rows = []
    s = result.samples[burn_in:]
    for j, name in enumerate(result.names):
        x = s[:, j]
        tau = iact(x)
        row = {
            "param": name,
            "mean": x.mean(),
            "sd": x.std(ddof=1),
            "q05": np.quantile(x, 0.05),
            "q50": np.median(x),
            "q95": np.quantile(x, 0.95),
            "iact": tau,
            "ess": x.size / tau,
            "ess_per_sec": x.size / tau / max(result.elapsed, 1e-9),
            "geweke_z": geweke_z(x),
        }
        if chains is not None:
            row["rhat"] = split_rhat(
                np.vstack([c.samples[burn_in:, j] for c in chains])
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "sampler", result.label)
    return df
