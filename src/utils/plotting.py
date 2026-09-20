import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f3b73", "#c0392b", "#2e8b57", "#d4a017", "#6c3483", "#117a8b"]
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)


def residual_heatmap(model):
    r = model.residuals()
    r = np.where(model.mask(), r, np.nan)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(
        r,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        extent=[
            model.data.years[0],
            model.data.years[-1],
            model.data.ages[0],
            model.data.ages[-1],
        ],
    )
    ax.set_xlabel("year")
    ax.set_ylabel("age")
    ax.set_title(f"Deviance residuals: {model.name}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def kappa_panel(years, kappa_hat, paths, truth=None):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    lo, med, hi = np.quantile(paths[:, :, 0], [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(
        years, lo, hi, color=PALETTE[0], alpha=0.25, label="smoothed 90% band"
    )
    ax.plot(years, med, color=PALETTE[0], lw=1.6, label="smoothed median")
    ax.plot(years, kappa_hat, "k.", ms=4, label="Poisson LC estimate")
    if truth is not None:
        ax.plot(years, truth, color=PALETTE[1], lw=1, ls="--", label="truth")
    ax.set_title("Period index $\\kappa_t$")
    ax.legend(frameon=False)
    return fig


def volatility_panel(years, paths, true_logvol=None, true_jumps=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    vol = np.exp(0.5 * paths[:, :, 1])
    lo, med, hi = np.quantile(vol, [0.05, 0.5, 0.95], axis=0)
    axes[0].fill_between(years, lo, hi, color=PALETTE[2], alpha=0.25)
    axes[0].plot(years, med, color=PALETTE[2], lw=1.6, label="smoothed $e^{h_t/2}$")
    if true_logvol is not None:
        axes[0].plot(
            years,
            np.exp(0.5 * true_logvol),
            color=PALETTE[1],
            ls="--",
            lw=1,
            label="truth",
        )
    axes[0].set_title("Stochastic volatility of $\\Delta\\kappa_t$")
    axes[0].legend(frameon=False)
    axes[1].bar(years, paths[:, :, 2].mean(0), color=PALETTE[3], label="P(jump | data)")
    if true_jumps is not None:
        axes[1].plot(
            years[true_jumps],
            np.full(true_jumps.sum(), 1.02),
            "v",
            color=PALETTE[1],
            label="true jump",
        )
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(frameon=False)
    return fig


def traces(chains, params):
    fig, axes = plt.subplots(
        len(params), 1, figsize=(7, 1.6 * len(params)), sharex=False
    )
    for ax, p in zip(np.atleast_1d(axes), params):
        for c, col in zip(chains, PALETTE):
            j = c.names.index(p)
            ax.plot(c.samples[:, j], lw=0.5, color=col, alpha=0.8, label=c.label)
        ax.set_ylabel(p)
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=len(chains), fontsize=7)
    fig.tight_layout()
    return fig


def posterior_densities(chains, params):
    fig, axes = plt.subplots(1, len(params), figsize=(2.3 * len(params), 2.4))
    for ax, p in zip(np.atleast_1d(axes), params):
        for c, col in zip(chains, PALETTE):
            j = c.names.index(p)
            ax.hist(
                c.samples[:, j],
                bins=40,
                density=True,
                histtype="step",
                color=col,
                lw=1.2,
                label=c.label,
            )
        ax.set_title(p)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def loglik_sd(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for (name, g), col in zip(df.groupby("filter"), PALETTE):
        ax.plot(g["particles"], g["sd"], "o-", color=col, label=name)
    ax.axhline(1.2, color="grey", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("particles")
    ax.set_ylabel("sd of log-likelihood estimate")
    ax.legend(frameon=False)
    return fig


def fan_chart(years, hist_logm, future_logm, ages, which):
    fig, axes = plt.subplots(1, len(which), figsize=(3.4 * len(which), 3))
    fy = years[-1] + np.arange(1, future_logm.shape[-1] + 1)
    for ax, a in zip(np.atleast_1d(axes), which):
        i = int(np.where(ages == a)[0][0])
        ax.plot(years, np.exp(hist_logm[i]), "k.", ms=3)
        for q, alpha in [
            ((0.025, 0.975), 0.15),
            ((0.1, 0.9), 0.25),
            ((0.25, 0.75), 0.35),
        ]:
            lo, hi = np.quantile(np.exp(future_logm[:, i]), q, axis=0)
            ax.fill_between(fy, lo, hi, color=PALETTE[0], alpha=alpha, lw=0)
        ax.plot(fy, np.median(np.exp(future_logm[:, i]), axis=0), color=PALETTE[0])
        ax.set_yscale("log")
        ax.set_title(f"$m_{{{a},t}}$")
    fig.tight_layout()
    return fig


def hedged_distributions(L, hedged):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    base = L.mean()
    edges = np.linspace(*np.quantile(L - base, [0.0005, 0.9995]), 90)
    ax.hist(
        L - base, bins=edges, density=True, alpha=0.4, color="grey", label="unhedged"
    )
    for (name, h), col in zip(hedged.items(), PALETTE):
        ax.hist(
            h - base,
            bins=edges,
            density=True,
            histtype="step",
            lw=1.3,
            color=col,
            label=name,
        )
    ax.set_xlabel("loss at horizon relative to unhedged mean (m)")
    ax.legend(frameon=False, fontsize=7)
    return fig


def heatmap(matrix, rows, cols, title, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2, 0.5 * len(rows) + 1.5))
    lo, hi = np.nanmin(matrix), np.nanmax(matrix)
    im = ax.imshow(matrix, cmap="viridis", vmin=lo, vmax=hi)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            shade = "k" if matrix[i, j] > lo + 0.6 * (hi - lo) else "w"
            ax.text(
                j,
                i,
                fmt.format(matrix[i, j]),
                ha="center",
                va="center",
                color=shade,
                fontsize=8,
            )
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def bars(labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(l) for l in labels], values, color=PALETTE[0])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig


def yield_factors(factors, two_step):
    fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, ["level", "slope", "curvature"])):
        ax.plot(two_step[:, j], color="grey", lw=0.8, label="two-step OLS")
        ax.plot(factors[:, j], color=PALETTE[j], lw=1.2, label="Kalman smoothed")
        ax.set_ylabel(name)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def backtest_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for (name, g), col in zip(df.groupby("model"), PALETTE):
        axes[0].plot(g["lead"], g["rmse"], "o-", color=col, ms=3, label=name)
        axes[1].plot(g["lead"], g["coverage95"], "o-", color=col, ms=3, label=name)
    axes[0].set_title("RMSE of log m (median forecast)")
    axes[1].set_title("95% interval coverage")
    axes[1].axhline(0.95, color="grey", ls=":")
    axes[0].set_xlabel("lead (years)")
    axes[1].set_xlabel("lead (years)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
