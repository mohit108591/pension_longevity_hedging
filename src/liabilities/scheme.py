from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lifetables import cohort_survival, extend_old_ages


@dataclass
class HorizonValuation:
    value: np.ndarray
    paid: np.ndarray
    future: np.ndarray
    cashflows: np.ndarray
    survivors: np.ndarray


class PensionScheme:
    def __init__(self, members, escalation=0.02, max_age=110):
        self.members = members.reset_index(drop=True)
        self.escalation = escalation
        self.max_age = max_age
        self.ages = self.members["age"].to_numpy(int)
        self.counts = self.members["count"].to_numpy(float)
        self.benefit = self.members["benefit"].to_numpy(float)
        self.ret_age = self.members["retirement_age"].to_numpy(int)

    @classmethod
    def synthetic(cls, seed=5, n_pensioners=12_000, n_deferred=6_000, escalation=0.02):
        rng = np.random.default_rng(seed)
        rows = []
        for age in range(55, 65):
            n = n_deferred / 10 * (0.8 + 0.4 * rng.random())
            rows.append((age, n, 9_000 * (1 + 0.02 * (64 - age)), 65, "deferred"))
        w = np.exp(-0.5 * ((np.arange(60, 96) - 72) / 8.0) ** 2)
        w /= w.sum()
        for age, share in zip(range(60, 96), w):
            rows.append(
                (
                    age,
                    n_pensioners * share,
                    11_000 * (1 - 0.006 * (age - 60)),
                    60,
                    "pensioner",
                )
            )
        df = pd.DataFrame(
            rows, columns=["age", "count", "benefit", "retirement_age", "status"]
        )
        return cls(df, escalation)

    def _payment_mask(self, H):
        s = np.arange(1, H + 1)
        attained = self.ages[:, None] + s[None, :]
        return (attained >= self.ret_age[:, None]).astype(float) * (
            (1 + self.escalation) ** s
        )[None, :]

    def survival(self, logm, ages):
        ext, ext_ages = extend_old_ages(logm, ages, self.max_age)
        return cohort_survival(ext, ext_ages, self.ages)

    def expected_cashflows(self, logm, ages):
        surv = self.survival(logm, ages)
        H = surv.shape[-1] - 1
        per_head = self.benefit[:, None] * self._payment_mask(H)
        return np.einsum("g,...gs,gs->...s", self.counts, surv[..., 1:], per_head) / 1e6

    def present_value(self, logm, ages, discount):
        cf = self.expected_cashflows(logm, ages)
        return np.sum(cf * discount[..., : cf.shape[-1]], axis=-1)

    def value_at_horizon(self, logm, ages, tau, accumulation, discount_after, rng=None):
        surv = self.survival(logm, ages)
        n_sims = surv.shape[0]
        H = surv.shape[-1] - 1
        per_head = self.benefit[:, None] * self._payment_mask(H) / 1e6
        if rng is None:
            lives = self.counts[None, :, None] * surv[..., 1:]
        else:
            lives = np.empty_like(surv[..., 1:])
            alive = np.broadcast_to(
                np.round(self.counts), (n_sims, self.counts.size)
            ).astype(np.int64)
            for s in range(1, tau + 1):
                p = np.divide(
                    surv[..., s],
                    surv[..., s - 1],
                    out=np.zeros_like(surv[..., s]),
                    where=surv[..., s - 1] > 0,
                )
                alive = rng.binomial(alive, np.clip(p, 0, 1))
                lives[..., s - 1] = alive
            cond = np.divide(
                surv[..., tau + 1 :],
                surv[..., tau : tau + 1],
                out=np.zeros_like(surv[..., tau + 1 :]),
                where=surv[..., tau : tau + 1] > 0,
            )
            lives[..., tau:] = alive[..., None] * cond
        cf = np.einsum("ngs,gs->ns", lives, per_head)
        paid = cf[:, :tau] @ accumulation[:tau]
        future = np.sum(cf[:, tau:] * discount_after[..., : H - tau], axis=-1)
        return HorizonValuation(
            paid + future, paid, future, cf, lives[..., tau - 1].sum(-1)
        )

    def duration(self, cf, discount):
        s = np.arange(1, cf.shape[-1] + 1)
        pv = cf * discount[: cf.shape[-1]]
        return float(np.sum(s * pv) / np.sum(pv))

    def summary(self):
        g = self.members.groupby("status").agg(
            members=("count", "sum"),
            avg_age=("age", "mean"),
            avg_benefit=("benefit", "mean"),
        )
        return g.round(1)
