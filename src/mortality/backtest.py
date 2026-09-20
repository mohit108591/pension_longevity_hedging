import numpy as np
import pandas as pd


def backtest(
    model_factory, data, origins, horizon, n_sims=500, seed=0, ages_to_score=None
):
    rng = np.random.default_rng(seed)
    rows = []
    for origin in origins:
        train = data.subset(years=data.years[data.years <= origin])
        test_years = np.arange(origin + 1, min(origin + horizon, data.years[-1]) + 1)
        if test_years.size == 0:
            continue
        model = model_factory().fit(train)
        sims = model.simulate_log_rates(test_years.size, n_sims, rng)
        realised = data.subset(years=test_years).log_rates
        ia = slice(None) if ages_to_score is None else np.isin(data.ages, ages_to_score)
        lo, med, hi = np.percentile(sims, [2.5, 50, 97.5], axis=0)
        inside = (realised >= lo) & (realised <= hi)
        pit = (sims < realised[None]).mean(0)
        for h in range(test_years.size):
            rows.append(
                {
                    "model": model.name,
                    "origin": origin,
                    "lead": h + 1,
                    "rmse": float(
                        np.sqrt(np.mean((med[ia, h] - realised[ia, h]) ** 2))
                    ),
                    "coverage95": float(inside[ia, h].mean()),
                    "pit_mean": float(pit[ia, h].mean()),
                    "pit_sd": float(pit[ia, h].std()),
                }
            )
    return pd.DataFrame(rows)


def summarise_backtest(df):
    return (
        df.groupby(["model", "lead"])[["rmse", "coverage95", "pit_mean"]]
        .mean()
        .reset_index()
    )
