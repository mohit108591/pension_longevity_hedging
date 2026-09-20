import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize


def _cov_terms(L, X):
    Xc = X - X.mean(0)
    Lc = L - L.mean()
    n = len(L)
    return Xc.T @ Xc / (n - 1), Xc.T @ Lc / (n - 1)


def min_variance(L, X, ridge=0.0):
    S, c = _cov_terms(L, X)
    scale = np.trace(S) / S.shape[0]
    return np.linalg.solve(S + ridge * scale * np.eye(S.shape[0]), c)


def mean_variance(L, X, gamma):
    S, c = _cov_terms(L, X)
    return np.linalg.solve(S, X.mean(0) / gamma + c)


def min_cvar(L, X, beta=0.99, bounds=None, max_gross=None, centred=True):
    if centred:
        L, X = L - L.mean(), X - X.mean(0)
    n, m = X.shape
    k = 1.0 / ((1 - beta) * n)
    cost = np.concatenate([np.zeros(m), [1.0], np.full(n, k)])
    A = sparse.hstack(
        [sparse.csr_matrix(-X), sparse.csr_matrix(-np.ones((n, 1))), -sparse.eye(n)]
    ).tocsr()
    b = -L
    bnds = list(bounds or [(None, None)] * m) + [(None, None)] + [(0, None)] * n
    if max_gross is not None:
        pos = sparse.hstack([sparse.eye(m), sparse.csr_matrix((m, 1 + n))])
        A = sparse.vstack([A, pos, -pos]).tocsr()
        b = np.concatenate([b, np.full(2 * m, max_gross)])
    res = linprog(cost, A_ub=A, b_ub=b, bounds=bnds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:m], float(res.fun)


def robust_min_variance(sets, relative=True, h0=None):
    terms = []
    for L, X in sets:
        S, c = _cov_terms(L, X)
        v = float(np.var(L, ddof=1))
        terms.append((S, c, v, v if relative else 1.0))
    m = terms[0][0].shape[0]
    h0 = np.zeros(m) if h0 is None else h0
    q = lambda h, t: (h @ t[0] @ h - 2 * h @ t[1] + t[2]) / t[3]
    x0 = np.append(h0, max(q(h0, t) for t in terms))
    cons = [
        {
            "type": "ineq",
            "fun": (lambda z, t=t: z[-1] - q(z[:-1], t)),
            "jac": (
                lambda z, t=t: np.append(-(2 * t[0] @ z[:-1] - 2 * t[1]) / t[3], 1.0)
            ),
        }
        for t in terms
    ]
    res = minimize(
        lambda z: z[-1],
        x0,
        jac=lambda z: np.append(np.zeros(m), 1.0),
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x[:-1], float(res.x[-1])


def bayesian_average_variance(sets, weights=None):
    w = (
        np.ones(len(sets)) / len(sets)
        if weights is None
        else np.asarray(weights) / np.sum(weights)
    )
    S_bar = sum(wi * _cov_terms(L, X)[0] for wi, (L, X) in zip(w, sets))
    c_bar = sum(wi * _cov_terms(L, X)[1] for wi, (L, X) in zip(w, sets))
    return np.linalg.solve(S_bar, c_bar)
