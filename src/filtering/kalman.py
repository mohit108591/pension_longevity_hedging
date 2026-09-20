from dataclasses import dataclass

import numpy as np

LOG2PI = np.log(2 * np.pi)


@dataclass
class LinearGaussianSSM:
    Z: np.ndarray
    d: np.ndarray
    H: np.ndarray
    T: np.ndarray
    c: np.ndarray
    Q: np.ndarray
    a0: np.ndarray
    P0: np.ndarray

    def obs_var(self, t):
        return self.H[t] if self.H.ndim == 2 else self.H

    @property
    def m(self):
        return self.T.shape[0]


@dataclass
class KalmanResult:
    loglik: float
    a_pred: np.ndarray
    P_pred: np.ndarray
    a_filt: np.ndarray
    P_filt: np.ndarray
    loglik_t: np.ndarray


def kalman_filter(y, ssm):
    n, p = y.shape
    m = ssm.m
    a_pred = np.empty((n, m))
    P_pred = np.empty((n, m, m))
    a_filt = np.empty((n, m))
    P_filt = np.empty((n, m, m))
    ll = np.zeros(n)
    a, P = ssm.a0.astype(float), ssm.P0.astype(float)
    complete = not np.isnan(y).any()
    for t in range(n):
        if t > 0:
            a = ssm.c + ssm.T @ a
            P = ssm.T @ P @ ssm.T.T + ssm.Q
        a_pred[t], P_pred[t] = a, P
        if complete:
            Z, h, v = ssm.Z, ssm.obs_var(t), y[t] - ssm.d - ssm.Z @ a
            n_obs = p
        else:
            obs = ~np.isnan(y[t])
            n_obs = int(obs.sum())
            Z, h = ssm.Z[obs], ssm.obs_var(t)[obs]
            v = y[t, obs] - ssm.d[obs] - Z @ a
        if n_obs:
            Hinv = 1.0 / h
            ZtHi = Z.T * Hinv
            Lp = np.linalg.cholesky(P)
            Pinv = np.linalg.inv(P)
            A = Pinv + ZtHi @ Z
            La = np.linalg.cholesky(A)
            A_inv = np.linalg.inv(A)
            u = ZtHi @ v
            quad = v @ (Hinv * v) - u @ A_inv @ u
            logdet = (
                np.sum(np.log(h))
                + 2 * np.sum(np.log(np.diag(Lp)))
                + 2 * np.sum(np.log(np.diag(La)))
            )
            ll[t] = -0.5 * (n_obs * LOG2PI + logdet + quad)
            a = a + A_inv @ u
            P = 0.5 * (A_inv + A_inv.T)
        a_filt[t], P_filt[t] = a, P
    return KalmanResult(float(ll.sum()), a_pred, P_pred, a_filt, P_filt, ll)


def rts_smoother(res, ssm):
    n, _ = res.a_filt.shape
    a_s = res.a_filt.copy()
    P_s = res.P_filt.copy()
    for t in range(n - 2, -1, -1):
        J = res.P_filt[t] @ ssm.T.T @ np.linalg.inv(res.P_pred[t + 1])
        a_s[t] = res.a_filt[t] + J @ (a_s[t + 1] - res.a_pred[t + 1])
        P_s[t] = res.P_filt[t] + J @ (P_s[t + 1] - res.P_pred[t + 1]) @ J.T
    return a_s, P_s


def ffbs(res, ssm, rng, n_draws=1):
    n, m = res.a_filt.shape
    out = np.empty((n_draws, n, m))
    L = np.linalg.cholesky(res.P_filt[-1] + 1e-12 * np.eye(m))
    x = res.a_filt[-1] + rng.standard_normal((n_draws, m)) @ L.T
    out[:, -1] = x
    Qinv = np.linalg.pinv(ssm.Q)
    for t in range(n - 2, -1, -1):
        Pf = res.P_filt[t]
        Pinv = np.linalg.inv(Pf + 1e-12 * np.eye(m))
        cov = np.linalg.inv(Pinv + ssm.T.T @ Qinv @ ssm.T)
        mean_part = Pinv @ res.a_filt[t]
        mean = (mean_part[None, :] + (x - ssm.c) @ Qinv.T @ ssm.T) @ cov.T
        Lc = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(m))
        x = mean + rng.standard_normal((n_draws, m)) @ Lc.T
        out[:, t] = x
    return out
