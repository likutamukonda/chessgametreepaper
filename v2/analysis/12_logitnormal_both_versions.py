"""Step 12: the zero-one-inflated logit-normal for Stockfish, in two versions, and two five-parameter extensions.

Model (four free parameters): P(e = 0) = p0, P(e = 1) = p1, and for 0 < e < 1
    logit(e) = log( distance from the best / distance from the worst ) ~ Normal(m, s).
Versions of the data:
    measured     the errors exactly as computed;
    near-ties    errors within SNAP = 0.001 of 0 or 1 counted as exactly 0 or 1.
Extensions of the middle part (five free parameters in all):
    skew         logit(e) ~ skew-normal(location, scale, shape alpha); alpha = 0 is the logit-normal;
    tails        logit(e) ~ Student t(location, scale, degrees of freedom nu); nu -> infinity is the
                 logit-normal, small nu gives heavy tails that absorb near-ties instead of stretching s.
Each depth is fitted on its own, pooled over positions with every position weighted equally (as in steps 10
and 11). Bands: 95% bootstrap over positions for the four-parameter model.
Outputs: out/logitnormal_both_versions.csv, out/logitnormal_both_versions.png
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats
from scipy.special import expit, log_ndtr, logit, ndtr

import common as c

SNAP, B, SEED = 1e-3, 200, 20261120
GRID = np.linspace(1e-4, 0.999, 999)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def version(e, kind):
    return e if kind == "measured" else np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e))


def pooled(per_pos, idx):
    e = np.concatenate([per_pos[i] for i in idx])
    w = np.concatenate([np.full(len(per_pos[i]), 1.0 / len(per_pos[i])) for i in idx])
    return e, w


def spikes(e, w):
    W = w.sum()
    w0, w1 = w[e == 0].sum(), w[e == 1].sum()
    inner = (e > 0) & (e < 1)
    ll = w0 * np.log(w0 / W) + w1 * np.log(w1 / W) + w[inner].sum() * np.log(w[inner].sum() / W)
    return w0 / W, w1 / W, inner, ll


def fit_normal(z, wz):
    m = float(np.sum(wz * z) / wz.sum())
    s = float(np.sqrt(np.sum(wz * (z - m) ** 2) / wz.sum()))
    ll = float(np.sum(wz * stats.norm.logpdf(z, m, s)))
    return {"m": m, "s": s}, ll, lambda zz: ndtr((zz - m) / s)


def fit_skew(z, wz, start):
    def nll(th):
        loc, sc, al = th[0], math.exp(th[1]), th[2]
        u = (z - loc) / sc
        return -np.sum(wz * (math.log(2) - th[1] - 0.5 * u * u - 0.5 * math.log(2 * math.pi) + log_ndtr(al * u)))
    best = min((optimize.minimize(nll, np.array(s0), method="L-BFGS-B", bounds=[(-20, 20), (-5, 5), (-20, 20)])
                for s0 in ([start["m"], math.log(start["s"]), 0.0], [start["m"] - 1, math.log(start["s"]), 1.0],
                           [start["m"] + 1, math.log(start["s"]), -1.0])), key=lambda r: r.fun)
    loc, sc, al = best.x[0], math.exp(best.x[1]), best.x[2]
    return {"loc": loc, "scale": sc, "alpha": al}, -float(best.fun), lambda zz: stats.skewnorm.cdf(zz, al, loc, sc)


def fit_t(z, wz, start):
    def nll(th):
        return -np.sum(wz * stats.t.logpdf(z, math.exp(th[2]), th[0], math.exp(th[1])))
    best = min((optimize.minimize(nll, np.array(s0), method="L-BFGS-B", bounds=[(-20, 20), (-5, 5), (math.log(0.3), math.log(500))])
                for s0 in ([start["m"], math.log(start["s"]), math.log(5)], [start["m"], math.log(start["s"] * 0.7), math.log(2)],
                           [start["m"], math.log(start["s"]), math.log(50)])), key=lambda r: r.fun)
    loc, sc, nu = best.x[0], math.exp(best.x[1]), math.exp(best.x[2])
    return {"loc": loc, "scale": sc, "nu": nu}, -float(best.fun), lambda zz: stats.t.cdf(zz, nu, loc, sc)


def fit_all(e, w, models=("logit-normal", "skew", "tails")):
    p0, p1, inner, ll_sp = spikes(e, w)
    z, wz = logit(e[inner]), w[inner]
    jac = -float(np.sum(wz * (np.log(e[inner]) + np.log1p(-e[inner]))))      # density of e from density of logit(e)
    out = {}
    base, ll, cdf = fit_normal(z, wz)
    fits = {"logit-normal": (base, ll, cdf, 4)}
    if "skew" in models:
        p, l, f = fit_skew(z, wz, base)
        fits["skew"] = (p, l, f, 5)
    if "tails" in models:
        p, l, f = fit_t(z, wz, base)
        fits["tails"] = (p, l, f, 5)
    for name, (p, l, f, k) in fits.items():
        total = ll_sp + l + jac
        curve = p0 + (1 - p0 - p1) * f(logit(GRID))
        out[name] = {"params": {"p0": p0, "p1": p1, **p}, "loglik": total, "k": k, "curve": curve}
    return out


def observed_curve(e, w):
    order = np.argsort(e)
    es, cw = e[order], np.cumsum(w[order]) / w.sum()
    idx = np.searchsorted(es, GRID + 1e-12, side="right") - 1
    return np.where(idx >= 0, cw[np.maximum(idx, 0)], 0.0)


def main():
    run = c.load_stockfish()
    xs = run.x()
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    rows, series = [], {}
    for kind in ("measured", "near-ties"):
        for li, d in enumerate(run.levels):
            per_pos = [version(c.relative_errors(xs[p][li:li + 1])[0], kind) for p in run.pos_ids]
            e, w = pooled(per_pos, np.arange(n_pos))
            fits = fit_all(e, w)
            obs = observed_curve(e, w)
            boot = [fit_all(*pooled(per_pos, idx), models=())["logit-normal"]["params"] for idx in boots]
            band = {k: np.percentile([b_[k] for b_ in boot], [2.5, 97.5]) for k in ("p0", "p1", "m", "s")}
            for name, f in fits.items():
                gap = float(np.abs(f["curve"] - obs).max())
                aic = 2 * f["k"] - 2 * f["loglik"]
                rows.append([kind, d, name, f["k"], round(f["loglik"], 4), round(aic, 4), round(gap, 5),
                             json.dumps({k: round(float(v), 6) for k, v in f["params"].items()})]
                            + ([round(float(band[k][j]), 6) for k in ("p0", "p1", "m", "s") for j in (0, 1)] if name == "logit-normal" else [""] * 8))
                series.setdefault((kind, name), []).append({"d": d, "aic": aic, "gap": gap, **f["params"],
                                                            **({f"{k}_lo": band[k][0] for k in band} | {f"{k}_hi": band[k][1] for k in band}
                                                               if name == "logit-normal" else {})})
    c.write_csv(c.OUT / "logitnormal_both_versions.csv", rows,
                ["version", "depth", "model", "n_params", "loglik", "aic", "largest_gap", "params",
                 "p0_lo", "p0_hi", "p1_lo", "p1_hi", "m_lo", "m_hi", "s_lo", "s_hi"])
    report(series)
    plot(series)


def report(series):
    print("Zero-one-inflated logit-normal, Stockfish, each depth fitted on its own")
    for kind in ("measured", "near-ties"):
        ln = series[(kind, "logit-normal")]
        print(f"\n  {kind}:")
        for key in ("p0", "p1", "m", "s"):
            print(f"    {key:<3} " + "  ".join(f"d{r['d']}: {r[key]:.3f}" for r in ln if r["d"] in (1, 5, 10, 15, 20, 25)))
        jumps = {key: np.mean(np.abs(np.diff([r[key] for r in ln][14:]))) for key in ("m", "s")}
        print(f"    mean depth-to-depth change over depths 15-25: m {jumps['m']:.3f}, s {jumps['s']:.3f}")
        print(f"    largest gap to the observed curve: median {np.median([r['gap'] for r in ln]):.3f}, worst {max(r['gap'] for r in ln):.3f}")
        for ext in ("skew", "tails"):
            ex = series[(kind, ext)]
            daic = [a["aic"] - b["aic"] for a, b in zip(ex, ln)]
            extra = "alpha" if ext == "skew" else "nu"
            print(f"    {ext:<5} (5 parameters): AIC lower than the 4-parameter model at {sum(x < -2 for x in daic)} of 25 depths "
                  f"(median change {np.median(daic):+.1f}); largest gap median {np.median([r['gap'] for r in ex]):.3f}; "
                  f"{extra} at d1/d13/d25: " + " / ".join(f"{ex[i][extra]:.2f}" for i in (0, 12, 24)))


def plot(series):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    panels = [("p0", "p0: share exactly at 0 (as good as the best)", (0, 0.18)),
              ("p1", "p1: share exactly at 1 (as bad as the worst)", (0, 0.18)),
              ("m", "m: centre of log(distance to best / distance to worst)", (-0.6, 0.6)),
              ("s", "s: spread of that log-ratio", (2.0, 4.4))]
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        for kind, color, label in (("measured", BLUE, "as measured"), ("near-ties", ORANGE, "near-ties (< 0.001) counted as ties")):
            ln = series[(kind, "logit-normal")]
            d = [r["d"] for r in ln]
            ax.fill_between(d, [r[f"{key}_lo"] for r in ln], [r[f"{key}_hi"] for r in ln], color=color, alpha=0.15, linewidth=0)
            ax.plot(d, [r[key] for r in ln], "-o", color=color, markersize=2.5, linewidth=1.6, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(*ylim)
        if key == "m":
            ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax = axes[1, 1]
    for kind, color, label in (("measured", BLUE, "as measured"), ("near-ties", ORANGE, "near-ties counted as ties")):
        ln = series[(kind, "logit-normal")]
        ax.plot([r["d"] for r in ln], [r["gap"] for r in ln], "-o", color=color, markersize=2.5, linewidth=1.6, label=label)
    ax.set_title("Fit: largest gap between fitted and observed curve", fontsize=9)
    ax.set_ylim(0, 0.1)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax = axes[1, 2]
    for kind, color, label in (("measured", BLUE, "as measured"), ("near-ties", ORANGE, "near-ties counted as ties")):
        ln, ex = series[(kind, "logit-normal")], series[(kind, "tails")]
        ax.plot([r["d"] for r in ln], [b["aic"] - a["aic"] for a, b in zip(ln, ex)], "-o", color=color, markersize=2.5,
                linewidth=1.6, label=label)
    ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_title("Heavier tails (5th parameter): change in AIC\n(below 0 = the extra parameter helps)", fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Stockfish depth")
    fig.suptitle("Zero-one-inflated logit-normal for Stockfish, fitted at each depth, in two versions (bands: 95% bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "logitnormal_both_versions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
