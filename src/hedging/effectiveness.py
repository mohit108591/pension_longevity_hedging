import numpy as np
import pandas as pd


def var_es(loss, alpha):
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q]
    return float(q), float(tail.mean())


def risk_profile(loss, alpha=0.995):
    c = loss - loss.mean()
    v, e = var_es(c, alpha)
    return {
        "mean": float(loss.mean()),
        "sd": float(loss.std(ddof=1)),
        f"VaR{alpha}": v,
        f"ES{alpha}": e,
    }


def hedge_effectiveness(unhedged, hedged, alpha=0.995):
    u = risk_profile(unhedged, alpha)
    h = risk_profile(hedged, alpha)
    return {
        "HE_variance": 1 - (h["sd"] / u["sd"]) ** 2,
        "HE_sd": 1 - h["sd"] / u["sd"],
        "HE_VaR": 1 - h[f"VaR{alpha}"] / u[f"VaR{alpha}"],
        "HE_ES": 1 - h[f"ES{alpha}"] / u[f"ES{alpha}"],
        "hedge_cost": h["mean"] - u["mean"],
        "residual_sd": h["sd"],
        "correlation": float(np.corrcoef(unhedged, unhedged - hedged)[0, 1])
        if np.std(unhedged - hedged) > 0
        else 0.0,
    }


def evaluate_strategies(L, X, strategies, alpha=0.995):
    rows = []
    for name, h in strategies.items():
        hedged = L - X @ h
        row = {"strategy": name, **hedge_effectiveness(L, hedged, alpha)}
        row.update({f"h[{i}]": v for i, v in enumerate(h)})
        rows.append(row)
    return pd.DataFrame(rows)


def population_basis_risk(he_index, he_customised):
    return {
        "HE_index": he_index,
        "HE_customised": he_customised,
        "population_basis_risk": he_customised - he_index,
        "relative_basis_risk": 1 - he_index / he_customised
        if he_customised
        else np.nan,
    }


def decompose_residual(L, parts):
    total = np.var(L, ddof=1)
    out = {}
    running = L.copy()
    for name, x in parts.items():
        beta = np.cov(running, x)[0, 1] / np.var(x, ddof=1)
        new = running - beta * x
        out[name] = (np.var(running, ddof=1) - np.var(new, ddof=1)) / total
        running = new
    out["unexplained"] = np.var(running, ddof=1) / total
    return out
