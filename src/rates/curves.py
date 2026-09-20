import numpy as np


class DiscountCurve:
    def __init__(self, maturities, zero_rates):
        t = np.asarray(maturities, float)
        z = np.asarray(zero_rates, float)
        order = np.argsort(t)
        self.t, self.z = t[order], z[order]
        self._lp = -self.z * self.t

    def zero(self, T):
        T = np.asarray(T, float)
        return np.interp(T, self.t, self.z, left=self.z[0], right=self.z[-1])

    def df(self, T):
        T = np.asarray(T, float)
        lp = np.interp(T, np.r_[0.0, self.t], np.r_[0.0, self._lp])
        beyond = T > self.t[-1]
        lp = np.where(beyond, -self.z[-1] * T, lp)
        return np.exp(lp)

    def forward(self, T, h=1e-4):
        T = np.asarray(T, float)
        return -(np.log(self.df(T + h)) - np.log(self.df(np.maximum(T - h, 0.0)))) / (
            T + h - np.maximum(T - h, 0.0)
        )

    def shifted(self, bump):
        return DiscountCurve(self.t, self.z + bump)

    def key_rate_bumped(self, key, bp=1e-4, keys=None):
        keys = np.asarray(keys, float)
        i = int(np.where(keys == key)[0][0])
        left = keys[i - 1] if i > 0 else None
        right = keys[i + 1] if i < keys.size - 1 else None
        w = np.zeros_like(self.t)
        for j, tj in enumerate(self.t):
            if tj == key:
                w[j] = 1.0
            elif left is not None and left < tj < key:
                w[j] = (tj - left) / (key - left)
            elif right is not None and key < tj < right:
                w[j] = (right - tj) / (right - key)
            elif left is None and tj < key or right is None and tj > key:
                w[j] = 1.0
        return DiscountCurve(self.t, self.z + bp * w)
