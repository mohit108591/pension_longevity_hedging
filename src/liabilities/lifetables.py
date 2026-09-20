import numpy as np

LOG_MORTALITY_CAP = np.log(2.0)


def extend_old_ages(logm, ages, max_age=110, fit_ages=10, cap=LOG_MORTALITY_CAP):
    ages = np.asarray(ages)
    tail = ages[-fit_ages:].astype(float)
    y = logm[..., -fit_ages:, :]
    xc = tail - tail.mean()
    slope = np.einsum("a,...at->...t", xc, y) / (xc @ xc)
    level = y.mean(axis=-2)
    extra = np.arange(ages[-1] + 1, max_age + 1)
    if extra.size == 0:
        return logm, ages
    ext = level[..., None, :] + (extra - tail.mean())[:, None] * slope[..., None, :]
    out = np.concatenate([logm, np.minimum(ext, cap)], axis=-2)
    return out, np.concatenate([ages, extra])


def cohort_survival(logm, ages, start_ages, horizon=None):
    ages = np.asarray(ages)
    H = logm.shape[-1] if horizon is None else horizon
    start_ages = np.asarray(start_ages)
    u = np.arange(H)
    age_idx = start_ages[:, None] - ages[0] + u[None, :]
    valid = age_idx < ages.size
    age_idx = np.minimum(age_idx, ages.size - 1)
    m = np.exp(logm[..., age_idx, u[None, :]])
    m = np.where(valid, m, np.inf)
    cum = np.cumsum(m, axis=-1)
    surv = np.exp(-cum)
    ones = np.ones(surv.shape[:-1] + (1,))
    return np.concatenate([ones, surv], axis=-1)


def annuity_factor(surv, discount, escalation=0.0, deferral=None):
    H = surv.shape[-1] - 1
    s = np.arange(1, H + 1)
    pay = (1 + escalation) ** s
    if deferral is not None:
        pay = pay[None, :] * (s[None, :] >= np.asarray(deferral)[:, None])
    return np.sum(surv[..., 1:] * pay * discount[..., :H], axis=-1)


def period_life_expectancy(logm_col, ages, from_age):
    i = int(np.where(np.asarray(ages) == from_age)[0][0])
    m = np.exp(logm_col[..., i:])
    p = np.exp(-m)
    lx = np.cumprod(
        np.concatenate([np.ones(p.shape[:-1] + (1,)), p[..., :-1]], axis=-1), axis=-1
    )
    return np.sum(lx * (1 - p) / m, axis=-1)


def cohort_life_expectancy(surv):
    return 0.5 + np.sum(surv[..., 1:], axis=-1)
