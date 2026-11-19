"""Step 13: functions of depth for the four parameters of the zero-one-inflated logit-normal (Stockfish).

Parameters at each depth d (closed form, each position weighted equally, as in step 12):
    p0 = share of moves exactly at 0, p1 = share exactly at 1,
    m, s = mean and standard deviation of logit(e) = log(distance to the best / distance to the worst)
over the moves strictly between. Both versions: errors as measured, and near-ties (within 0.001 of 0 or 1)
counted as ties.

Shapes fitted to each parameter as a function of depth d:
    exponential  y = L - c * exp(-k d)                 saturating exponential (Shiku's proposal); asymptote L
    power        y = L - c * d^(-g)                    approaches its asymptote L more slowly
    logistic     y = A + (L - A) / (1 + exp(-k (d - d0)))   S-curve: flat, then a rise or fall, then flat at L
    linear       y = a + b d                           benchmark without an asymptote
    log-linear   y = a + b ln d                        benchmark without an asymptote
Fitting: weighted least squares over depths 1-25, weights 1 / se^2 with se the bootstrap standard error of
the parameter at that depth. Held-out test: fit on depths 1-20, predict 21-25, error in units of se.
Asymptote L: refitted on 200 bootstrap replicate series (positions resampled once per replicate, the same
for every depth); 95% interval, and the share of replicates in which the curve actually levels off.
For the exponential, the half-life ln(2) / k is the number of extra plies that closes half of the remaining
gap to the asymptote.
Outputs: out/depth_functions.csv, out/depth_functions.png
"""
import json
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import logit

import common as c

warnings.simplefilter("ignore", OptimizeWarning)
SNAP, B, SEED = 1e-3, 200, 20261120
PARAMS = ("p0", "p1", "m", "s")
RANGE = {"p0": (0.0, 1.0), "p1": (0.0, 1.0), "m": (-10.0, 10.0), "s": (0.0, 20.0)}
SHAPES = ("exponential", "power", "logistic", "linear", "log-linear")
COLORS = {"exponential": "#2a78d6", "power": "#eb6834", "logistic": "#1baf7a"}
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
F = {"exponential": lambda d, L, cc, k: L - cc * np.exp(-k * d),
     "power": lambda d, L, cc, g: L - cc * np.power(d, -g),
     "logistic": lambda d, A, L, k, d0: A + (L - A) / (1 + np.exp(-k * (d - d0))),
     "linear": lambda d, a, b: a + b * d,
     "log-linear": lambda d, a, b: a + b * np.log(d)}
RATE_INDEX = {"exponential": 2, "power": 2, "logistic": 2}


def parameters(per_depth, idx):
    """per_depth: list over depths of per-position error arrays; idx: positions -> dict of arrays over depths."""
    out = {k: [] for k in PARAMS}
    for per_pos in per_depth:
        e = np.concatenate([per_pos[i] for i in idx])
        w = np.concatenate([np.full(len(per_pos[i]), 1.0 / len(per_pos[i])) for i in idx])
        W = w.sum()
        out["p0"].append(w[e == 0].sum() / W)
        out["p1"].append(w[e == 1].sum() / W)
        inner = (e > 0) & (e < 1)
        z, wz = logit(e[inner]), w[inner] / w[inner].sum()
        m = float(np.sum(wz * z))
        out["m"].append(m)
        out["s"].append(float(np.sqrt(np.sum(wz * (z - m) ** 2))))
    return {k: np.array(v) for k, v in out.items()}


def fit_shape(shape, d, y, se, par, start=None):
    """Weighted least squares; returns (params, asymptote or None, levels_off)."""
    lo_y, hi_y = RANGE[par]
    if shape in ("linear", "log-linear"):
        X = np.column_stack([np.ones_like(d), d if shape == "linear" else np.log(d)])
        Wt = 1 / se ** 2
        coef = np.linalg.solve(X.T @ (X * Wt[:, None]), X.T @ (Wt * y))
        return [float(v) for v in coef], None, False
    span = float(y[-1] - y[0]) or 1e-3
    if shape == "exponential":
        bounds = ([lo_y, -50, 1e-3], [hi_y, 50, 3.0])
        starts = [start] if start is not None else [[y[-1] + span * 0.2, span, r] for r in (0.03, 0.1, 0.3)]
    elif shape == "power":
        bounds = ([lo_y, -50, 1e-3], [hi_y, 50, 5.0])
        starts = [start] if start is not None else [[y[-1] + span * 0.2, span, g] for g in (0.2, 0.6, 1.5)]
    else:
        bounds = ([lo_y, lo_y, 1e-3, -20.0], [hi_y, hi_y, 5.0, 60.0])
        starts = [start] if start is not None else [[y[0], y[-1] + span * 0.1, k, d0] for k in (0.2, 0.5) for d0 in (10, 18, 25)]
    best = None
    for s0 in starts:
        s0 = [min(max(v, lb + 1e-9), ub - 1e-9) for v, lb, ub in zip(s0, *bounds)]
        try:
            p, _ = curve_fit(F[shape], d, y, p0=s0, sigma=se, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(((F[shape](d, *p) - y) / se) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p)
    if best is None:
        return None, None, False
    p = best[1]
    rate = p[RATE_INDEX[shape]]
    asym = p[1] if shape == "logistic" else p[0]
    at_bound = rate <= 1.5e-3 or abs(asym - lo_y) < 1e-6 or abs(asym - hi_y) < 1e-6
    return [float(v) for v in p], float(asym), not at_bound


def main():
    run = c.load_stockfish()
    xs = run.x()
    d = np.array(run.levels, float)
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    rows, plot_data = [], {}
    for kind in ("measured", "near-ties"):
        snap = (lambda e: e) if kind == "measured" else (lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e)))
        per_depth = [[snap(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids] for li in range(len(d))]
        full = parameters(per_depth, np.arange(n_pos))
        reps = [parameters(per_depth, idx) for idx in boots]
        for par in PARAMS:
            y = full[par]
            rep = np.array([r[par] for r in reps])
            se = rep.std(axis=0, ddof=1)
            lo95, hi95 = np.percentile(rep, [2.5, 97.5], axis=0)
            train = d <= 20
            fits = {}
            for shape in SHAPES:
                p_all, asym, levels = fit_shape(shape, d, y, se, par)
                p_tr, _, _ = fit_shape(shape, d[train], y[train], se[train], par)
                if p_all is None or p_tr is None:
                    continue
                z_in = (F[shape](d, *p_all) - y) / se
                z_out = (F[shape](d[~train], *p_tr) - y[~train]) / se[~train]
                raw_out = F[shape](d[~train], *p_tr) - y[~train]
                lims, n_level = [], 0
                if shape in COLORS:
                    for r in rep:
                        pb, ab, lb = fit_shape(shape, d, r, se, par, start=p_all)
                        if pb is not None and lb:
                            n_level += 1
                            lims.append(ab)
                ci = np.percentile(lims, [2.5, 97.5]) if len(lims) >= 20 else (np.nan, np.nan)
                half = math.log(2) / p_all[2] if shape == "exponential" else np.nan
                fits[shape] = p_all
                rows.append([kind, par, shape, len(p_all), round(float(np.sqrt(np.mean(z_in ** 2))), 3),
                             round(float(np.sqrt(np.mean(z_out ** 2))), 3), round(float(np.sqrt(np.mean(raw_out ** 2))), 5),
                             "" if asym is None else round(asym, 5), levels,
                             "" if np.isnan(ci[0]) else round(float(ci[0]), 5), "" if np.isnan(ci[1]) else round(float(ci[1]), 5),
                             round(n_level / B, 3) if shape in COLORS else "", "" if np.isnan(half) else round(half, 2),
                             json.dumps([round(v, 6) for v in p_all])])
            plot_data[(kind, par)] = (y, lo95, hi95, fits)
    c.write_csv(c.OUT / "depth_functions.csv", rows,
                ["version", "parameter", "shape", "n_params", "rms_z_in_sample", "rms_z_heldout", "rms_heldout",
                 "asymptote", "levels_off", "asymptote_lo95", "asymptote_hi95", "share_bootstrap_levelling_off",
                 "half_life_plies", "params"])
    report(rows)
    plot(d, plot_data)


def report(rows):
    print("Functions of depth, Stockfish. rms z = root-mean-square error in units of each depth's standard error")
    print("(about 1 = within the noise). Held out: fitted on depths 1-20, predicting 21-25.")
    for kind in ("measured", "near-ties"):
        print(f"\n=== {kind} ===")
        for par in PARAMS:
            print(f"  {par}")
            for r in rows:
                if r[0] == kind and r[1] == par:
                    asym = "-" if r[7] == "" else f"{r[7]:.3f}" + ("" if r[8] else " (does not level off)")
                    ci = "" if r[9] == "" else f" [{r[9]:.3f}, {r[10]:.3f}], levels off in {r[11]:.0%} of resamples"
                    hl = "" if r[12] == "" else f"; half-life {r[12]:.1f} plies"
                    print(f"    {r[2]:<12} in-sample rms z {r[4]:>5.2f}   held-out rms z {r[5]:>6.2f}   asymptote {asym}{ci}{hl}")


def plot(d, plot_data):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    titles = {"p0": "p0 (share exactly at 0)", "p1": "p1 (share exactly at 1)", "m": "m (centre of the log-ratio)",
              "s": "s (spread of the log-ratio)"}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True)
    dd = np.linspace(1, 40, 300)
    for r_i, kind in enumerate(("measured", "near-ties")):
        for c_i, par in enumerate(PARAMS):
            ax = axes[r_i, c_i]
            y, lo, hi, fits = plot_data[(kind, par)]
            ax.errorbar(d, y, yerr=[y - lo, hi - y], fmt="o", color=INK2, markersize=2.5, elinewidth=0.8, capsize=0,
                        label="fitted at each depth (95% band)")
            for shape, color in COLORS.items():
                if shape in fits:
                    yy = F[shape](dd, *fits[shape])
                    ax.plot(dd[dd <= 25], yy[dd <= 25], color=color, linewidth=1.6, label=shape)
                    ax.plot(dd[dd >= 25], yy[dd >= 25], color=color, linewidth=1.2, linestyle=(0, (2, 2)))
            ax.axvspan(25, 40, color=GRID_C, alpha=0.35, linewidth=0)
            ax.set_title(f"{titles[par]}, {'as measured' if kind == 'measured' else 'near-ties as ties'}", fontsize=8.5)
            if r_i == 1:
                ax.set_xlabel("Stockfish depth (shaded: beyond the data)")
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Functions of depth fitted to the four parameters (curves dotted beyond depth 25, where there is no data)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "depth_functions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
