import numpy as np


def triangular_weights(grid, keys):
    grid = np.asarray(grid, float)
    keys = np.asarray(keys, float)
    W = np.zeros((keys.size, grid.size))
    for j, k in enumerate(keys):
        left = keys[j - 1] if j > 0 else None
        right = keys[j + 1] if j < keys.size - 1 else None
        w = np.zeros_like(grid)
        w[grid == k] = 1.0
        if left is None:
            w[grid < k] = 1.0
        else:
            sel = (grid > left) & (grid < k)
            w[sel] = (grid[sel] - left) / (k - left)
        if right is None:
            w[grid > k] = 1.0
        else:
            sel = (grid > k) & (grid < right)
            w[sel] = (right - grid[sel]) / (right - k)
        W[j] = w
    return W


def bump_q(logm, weights_by_age, dq):
    q = 1 - np.exp(-np.exp(logm))
    q_new = np.clip(q + dq * weights_by_age[:, None], 1e-8, 1 - 1e-8)
    return np.log(-np.log1p(-q_new))


def key_q_durations(value_fn, logm, ages, key_ages, dq=-1e-4):
    W = triangular_weights(ages, key_ages)
    base = value_fn(logm)
    return np.array(
        [
            (value_fn(bump_q(logm, W[j], dq)) - base) / abs(dq)
            for j in range(len(key_ages))
        ]
    )


def qforward_kqd(qf, df_maturity, ages, key_ages, dq=-1e-4):
    W = triangular_weights(ages, key_ages)
    i = int(np.where(np.asarray(ages) == qf.age)[0][0])
    return -qf.notional * df_maturity * W[:, i] * dq / abs(dq)


def kqd_hedge(liability_kqd, instrument_kqd):
    A = np.column_stack(instrument_kqd)
    h, *_ = np.linalg.lstsq(A, liability_kqd, rcond=None)
    return h


def key_rate_durations(value_fn, curve, keys, bp=1e-4):
    base = value_fn(curve)
    return np.array(
        [(value_fn(curve.key_rate_bumped(k, bp, keys)) - base) / bp for k in keys]
    )


def key_rate_hedge(liability_krd, curve, keys, maturities, bp=1e-4):
    cols = []
    for T in maturities:
        f = lambda c, T=T: float(c.df(T))
        cols.append(key_rate_durations(f, curve, keys, bp))
    A = np.column_stack(cols)
    h, *_ = np.linalg.lstsq(A, liability_krd, rcond=None)
    return h, A
