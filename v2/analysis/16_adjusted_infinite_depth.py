"""Step 16: the infinite-depth distribution with two adjustments (Stockfish, near-ties counted as ties).

1. Shallow depths. At the first few depths the search is too shallow to see past simple threats (the
   horizon effect). Checked from the data: the node count per extra ply, how far a move's value moves
   between consecutive depths, and how far each parameter sits from the trend of the deeper depths.
   The fits below start at depth 5 (depths 1-4 left out); the first depth is varied as a check.
2. A gamma kernel for s:  s(d) = c + H (d / d_peak)^a exp(a (1 - d / d_peak)),  a hump of height H at
   d_peak on top of the level c. Unlike the logistic S-curve it can rise and then fall. It tends to c as d
   goes to infinity, and also as d goes to 0, so its level at infinite depth equals its starting level.
   Compared with the logistic by weighted fit, held-out depths 21-25 (fitted on 5-20) and the bootstrap
   interval of the level.
p0, the middle share and m keep the logistic S-curve; p1 = 1 - p0 - middle. Uncertainty: the 200 bootstrap
resamples of the positions used throughout; a resample counts when every curve has a level (the logistic
levels off, the gamma kernel peaks inside the search range and its level is not at a bound).
Outputs: out/shallow_depths.csv, out/adjusted_fits.csv, out/adjusted_infinite_depth.csv,
         out/adjusted_parameters.png
"""
import collections
import csv
import importlib
import io
import warnings
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

import common as c

warnings.simplefilter("ignore", OptimizeWarning)
f13 = importlib.import_module("13_depth_functions")
f15 = importlib.import_module("15_infinite_depth")
LOG = f13.F["logistic"]
B, SEED, SNAP, FIRST = 200, 20261120, 1e-3, 5
GK_LB, GK_UB = [0.0, -20.0, 1.0, 0.01], [20.0, 20.0, 500.0, 50.0]
SHAPES = {"p0": "logistic", "middle": "logistic", "m": "logistic", "s": "gamma kernel", "s_logistic": "logistic"}
ALL = f15.ALL
TOL = (0.0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.99)
INF_X, CURVE_END = f15.INF_X, f15.CURVE_END
BLUE, ORANGE, INK, INK2, MUTED, GRID_C, SURFACE = f15.BLUE, f15.ORANGE, f15.INK, f15.INK2, f15.MUTED, f15.GRID_C, f15.SURFACE


def gamma_kernel(d, cc, H, dp, a):
    return cc + H * (d / dp) ** a * np.exp(a * (1 - d / dp))


CURVE = {"logistic": LOG, "gamma kernel": gamma_kernel}


def fit_gamma(d, y, se, start=None):
    """Weighted least squares; returns (params, level c, has a level)."""
    starts = [start] if start is not None else [[cc, H, dp, a] for cc in (1.0, 2.0, 2.4) for H in (0.5, 1.0)
                                                for dp in (20.0, 40.0, 100.0) for a in (0.2, 1.0, 3.0)]
    best = None
    for s0 in starts:
        s0 = [min(max(v, lo + 1e-9), hi - 1e-9) for v, lo, hi in zip(s0, GK_LB, GK_UB)]
        try:
            p, _ = curve_fit(gamma_kernel, d, y, p0=s0, sigma=se, bounds=(GK_LB, GK_UB), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(((gamma_kernel(d, *p) - y) / se) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p)
    if best is None:
        return None, None, False
    p = best[1]
    ok = GK_LB[0] + 1e-6 < p[0] < GK_UB[0] - 1e-6 and p[2] < GK_UB[2] - 1e-6
    return [float(v) for v in p], float(p[0]), bool(ok)


def fit(q, d, y, se, start=None):
    shape = SHAPES[q]
    if shape == "gamma kernel":
        return fit_gamma(d, y, se, start)
    return f13.fit_shape("logistic", d, y, se, "s" if q == "s_logistic" else q, start)


def data_key(q):
    return "s" if q == "s_logistic" else q


def fit_all(d, ser, se, first, starts=None):
    k = d >= first
    return {q: fit(q, d[k], ser[data_key(q)][k], se[data_key(q)][k], None if starts is None else starts[q][0])
            for q in SHAPES}


def levels(fits, s_key="s"):
    """Levels at infinite depth, or None if any curve has no level."""
    need = ("p0", "middle", "m", s_key)
    if any(fits[q][0] is None or not fits[q][2] for q in need):
        return None
    out = {"p0": fits["p0"][1], "middle": fits["middle"][1], "m": fits["m"][1], "s": fits[s_key][1]}
    out["p1"] = 1 - out["p0"] - out["middle"]
    return out


def curve_values(q, fits, dd, s_key="s"):
    if q == "p1":
        return 1 - LOG(dd, *fits["p0"][0]) - LOG(dd, *fits["middle"][0])
    key = s_key if q == "s" else q
    return CURVE[SHAPES[key]](dd, *fits[key][0])


def stockfish_setup():
    """Errors (near-ties as ties), per-depth parameters, bootstrap resamples and standard errors."""
    run = c.load_stockfish()
    xs = run.x()
    d = np.array(run.levels, float)
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    snap = lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e))
    per_depth = [[snap(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids] for li in range(len(d))]
    full = f15.series(per_depth, np.arange(n_pos))
    reps = [f15.series(per_depth, idx) for idx in boots]
    se = {q: np.array([r[q] for r in reps]).std(axis=0, ddof=1) for q in ALL}
    return run, xs, d, boots, per_depth, full, reps, se


def stockfish_limits(first=FIRST):
    """Full-data fits and levels, and per-resample fits (aligned with the resamples), from depth `first`."""
    run, xs, d, boots, per_depth, full, reps, se = stockfish_setup()
    fits = fit_all(d, full, se, first)
    boot_fits = [fit_all(d, r, se, first, starts=fits) for r in reps]
    return {"run": run, "xs": xs, "d": d, "boots": boots, "per_depth": per_depth, "full": full, "reps": reps,
            "se": se, "fits": fits, "boot_fits": boot_fits, "first": first}


def load_nodes():
    out = collections.defaultdict(dict)
    with zipfile.ZipFile(c.SF_RUN) as zf:
        for r in csv.DictReader(io.TextIOWrapper(zf.open("stockfish_evals.csv"), encoding="utf-8", newline="")):
            out[(r["pos_id"], r["move"])][int(r["depth"])] = int(r["nodes"])
    return out


def shallow_depths(S):
    """Per depth: node count per extra ply, average change in a move's value, distance of each parameter
    from the curves fitted from depth FIRST on (extended back), in standard errors."""
    d, full, se, fits, xs, run = S["d"], S["full"], S["se"], S["fits"], S["xs"], S["run"]
    nodes = load_nodes()
    rows = []
    for li, dep in enumerate(d.astype(int)):
        ratio = np.median([v[dep] / v[dep - 1] for v in nodes.values() if dep in v and dep - 1 in v and v[dep - 1] > 0]) if dep > 1 else np.nan
        if li > 0:
            ch = np.concatenate([xs[p][li] - xs[p][li - 1] for p in run.pos_ids])
            swing = float(np.mean(np.abs(ch[np.isfinite(ch)])))
        else:
            swing = np.nan
        z = {q: float((full[q][li] - curve_values(q, fits, np.array([dep], float))[0]) / se[q][li]) for q in ALL}
        rows.append([int(dep), ratio, swing] + [z[q] for q in ALL])
    c.write_csv(c.OUT / "shallow_depths.csv",
                [[r[0]] + [("" if np.isnan(v) else round(float(v), 4)) for v in r[1:]] for r in rows],
                ["depth", "median_node_ratio_per_ply", "mean_abs_change_in_value", "z_p0", "z_middle", "z_p1",
                 "z_m", "z_s"])
    return rows


def main():
    S = stockfish_limits()
    d, full, se, fits, boot_fits = S["d"], S["full"], S["se"], S["fits"], S["boot_fits"]
    rows_shallow = shallow_depths(S)

    # shape comparison for every quantity, fitted from depth FIRST: in-sample and held out (fit to 20, predict 21-25)
    k, tr, ho = d >= FIRST, (d >= FIRST) & (d <= 20), d >= 21
    fit_rows = []
    for q in SHAPES:
        y, s_ = full[data_key(q)], se[data_key(q)]
        p_tr = fit(q, d[tr], y[tr], s_[tr])[0]
        f = CURVE[SHAPES[q]]
        rss = float(np.sum(((f(d[k], *fits[q][0]) - y[k]) / s_[k]) ** 2))
        z_out = (f(d[ho], *p_tr) - y[ho]) / s_[ho]
        lv = [bf[q][1] for bf in boot_fits if bf[q][0] is not None and bf[q][2]]
        lo, hi = np.percentile(lv, [2.5, 97.5]) if len(lv) >= 20 else (np.nan, np.nan)
        fit_rows.append([data_key(q), SHAPES[q], FIRST, rss, float(np.sqrt(np.mean(z_out ** 2))), fits[q][1],
                         bool(fits[q][2]), float(lo), float(hi), len(lv) / B, fits[q][0]])
    c.write_csv(c.OUT / "adjusted_fits.csv",
                [r[:-1] + [" ".join(f"{v:.5g}" for v in r[-1])] for r in
                 [[round(v, 4) if isinstance(v, float) else v for v in row[:-1]] + [row[-1]] for row in fit_rows]],
                ["parameter", "shape", "first_depth", "weighted_rss", "rms_z_heldout_21_25", "level", "has_level",
                 "level_lo95", "level_hi95", "share_of_resamples_with_level", "params"])

    # sensitivity of the levels to the first depth (full data)
    sens = []
    for first in (1, 3, 4, 5, 6, 7, 8):
        fa = fit_all(d, full, se, first)
        for s_key in ("s", "s_logistic"):
            L = levels(fa, s_key)
            sens.append([first, SHAPES[s_key]] + ([round(L[q], 4) for q in ALL] if L else [""] * len(ALL)))

    # infinite-depth distribution: gamma kernel for s (main) and logistic for s (alternative)
    out_rows, dist = [], {}
    for s_key in ("s", "s_logistic"):
        inf = levels(fits, s_key)
        bl = [L for L in (levels(bf, s_key) for bf in boot_fits) if L is not None]
        F = f15.share_curve(inf["p0"], inf["middle"], inf["m"], inf["s"])
        Fb = np.array([f15.share_curve(L["p0"], L["middle"], L["m"], L["s"]) for L in bl])
        dist[s_key] = (inf, bl, F, Fb)
        for q in ALL:
            lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
            out_rows.append([SHAPES[s_key], "parameter", q, float(full[q][-1]), inf[q], float(lo), float(hi), len(bl) / B])
        for t in TOL:
            i = int(np.argmin(np.abs(f15.GRID - t)))
            lo, hi = np.percentile(Fb[:, i], [2.5, 97.5])
            out_rows.append([SHAPES[s_key], "share within t", t, float(np.nan), float(F[i]), float(lo), float(hi), len(bl) / B])
    c.write_csv(c.OUT / "adjusted_infinite_depth.csv",
                [[("" if isinstance(v, float) and np.isnan(v) else round(v, 4) if isinstance(v, float) else v) for v in r]
                 for r in out_rows],
                ["s_shape", "quantity", "parameter_or_t", "value_at_depth_25", "value_at_infinite_depth", "lo95", "hi95",
                 "share_of_resamples_used"])
    report(rows_shallow, fit_rows, sens, out_rows, S)
    plot(S, dist)


def report(rows_shallow, fit_rows, sens, out_rows, S):
    print(f"1. Shallow depths. Distance of each parameter from the curves fitted on depths {FIRST}-25, extended back")
    print("   (data minus curve, in standard errors), with the node count per extra ply and the average change")
    print("   in a move's value since the previous depth")
    print(f"   {'depth':>5} {'nodes x':>8} {'|change|':>9} " + " ".join(f"{q:>7}" for q in ALL))
    for r in rows_shallow[:10]:
        print(f"   {r[0]:>5} {('' if np.isnan(r[1]) else f'{r[1]:.2f}'):>8} {('' if np.isnan(r[2]) else f'{r[2]:.4f}'):>9} "
              + " ".join(f"{v:>+7.1f}" for v in r[3:]))
    print(f"   (depths 11-25: node ratio {np.nanmedian([r[1] for r in rows_shallow[10:]]):.2f}, "
          f"|change| {rows_shallow[10][2]:.4f} -> {rows_shallow[-1][2]:.4f})")
    print(f"\n2. Shapes fitted from depth {FIRST} (weighted rss in-sample; held-out rms z for depths 21-25 fitted on {FIRST}-20)")
    for r in fit_rows:
        print(f"   {r[0]:<7} {r[1]:<13} rss {r[3]:6.2f}   held-out rms z {r[4]:5.2f}   level {r[5]:7.3f}"
              f"{'' if r[6] else ' (no level)'}   95% [{r[7]:.3f}, {r[8]:.3f}]   resamples with a level {r[9]:.0%}")
        if r[1] == "gamma kernel":
            cc, H, dp, a = r[10]
            dd = np.array([25, 30, 40, 60, 100.0])
            print(f"           peak {cc + H:.3f} at depth {dp:.1f}; values at depths 25/30/40/60/100: "
                  + " ".join(f"{v:.3f}" for v in gamma_kernel(dd, *r[10])))
    print("\n3. Levels at infinite depth by first depth used (full data)")
    print(f"   {'first':>5} {'s shape':<13} " + " ".join(f"{q:>7}" for q in ALL))
    for r in sens:
        print(f"   {r[0]:>5} {r[1]:<13} " + " ".join(f"{v:>7.3f}" if v != "" else f"{'-':>7}" for v in r[2:]))
    print(f"\n4. Infinite depth, fitted from depth {FIRST}")
    for shape in ("gamma kernel", "logistic"):
        rr = [r for r in out_rows if r[0] == shape]
        print(f"   s by {shape} ({rr[0][7]:.0%} of resamples):")
        for r in rr:
            if r[1] == "parameter":
                print(f"     {r[2]:<7} depth 25 {r[3]:7.3f}   infinite depth {r[4]:7.3f}   95% [{r[5]:.3f}, {r[6]:.3f}]")
        print("     share within t: " + "  ".join(f"{r[2]}: {r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}]" for r in rr if r[1] != "parameter"))


def draw_limit(ax, x, level, lo, hi, y_lo, y_hi, clip_lo, clip_hi, color, label):
    room = 0.03 * (y_hi - y_lo)
    top, bottom = (y_hi - room if clip_hi else hi), (y_lo + room if clip_lo else lo)
    ax.plot([x, x], [bottom, top], color=color, linewidth=1.4)
    for end, clipped, val, mk, va in ((top, clip_hi, hi, "^", "top"), (bottom, clip_lo, lo, "v", "bottom")):
        if clipped:
            ax.plot(x, end, marker=mk, color=color, markersize=6)
            ax.text(x - 2.2, end, f"to {val:.2f}", ha="right", va=va, fontsize=7.5, color=INK2)
        else:
            ax.plot(x, end, marker="_", color=color, markersize=8, markeredgewidth=1.4)
    ax.plot(x, level, "o", color=color, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5, label=label)


def plot(S, dist):
    f15.style()
    d, full, se, fits = S["d"], S["full"], S["se"], S["fits"]
    inf, bl = dist["s"][0], dist["s"][1]
    inf_alt = dist["s_logistic"][0]
    dd = np.linspace(FIRST, CURVE_END, 400)
    keep = d >= FIRST
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.9))
    titles = {"p0": "p0: share exactly at 0 (S-curve)", "middle": "middle: share strictly between 0 and 1 (S-curve)",
              "p1": "p1 = 1 − p0 − middle (calculated)", "m": "m: centre of the log-ratio (S-curve)",
              "s": "s: spread of the log-ratio (gamma kernel)"}
    where = {"p0": "upper left", "middle": "lower left", "p1": "upper left", "m": "lower right", "s": "lower right"}
    for ax, q in zip(axes.flat, ALL):
        y, s_ = full[q], se[q]
        yy = curve_values(q, fits, dd)
        level = inf[q]
        lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
        extra = curve_values(q, fits, dd, "s_logistic") if q == "s" else yy
        base_lo = min(float((y - 1.96 * s_).min()), float(yy.min()), float(extra.min()), level)
        base_hi = max(float((y + 1.96 * s_).max()), float(yy.max()), float(extra.max()), level)
        span = base_hi - base_lo
        y_lo, y_hi = base_lo - 0.12 * span, base_hi + 0.12 * span
        clip_lo, clip_hi = lo < y_lo - 1.2 * span, hi > y_hi + 1.2 * span
        y_lo = y_lo if clip_lo else min(y_lo, lo - 0.06 * span)
        y_hi = y_hi if clip_hi else max(y_hi, hi + 0.06 * span)
        ax.axvspan(25, 64, color=GRID_C, alpha=0.35, linewidth=0)
        ax.errorbar(d[~keep], y[~keep], yerr=1.96 * s_[~keep], fmt="o", color=MUTED, markersize=2.5, elinewidth=0.8,
                    capsize=0, markerfacecolor=SURFACE, label="depths 1–4 (left out)")
        ax.errorbar(d[keep], y[keep], yerr=1.96 * s_[keep], fmt="o", color=INK2, markersize=2.5, elinewidth=0.8,
                    capsize=0, label="depths 5–25 (95% band)")
        ax.plot(dd[dd <= 25], yy[dd <= 25], color=BLUE, linewidth=1.8,
                label={"p1": "from the two S-curves", "s": "gamma kernel"}.get(q, "S-curve"))
        ax.plot(dd[dd >= 25], yy[dd >= 25], color=BLUE, linewidth=1.2, linestyle=(0, (2, 2)))
        if q == "s":
            ax.plot(dd, extra, color=MUTED, linewidth=1.1, linestyle=(0, (5, 3)), label=f"S-curve (level {inf_alt['s']:.2f})")
            ax.plot(INF_X, inf_alt["s"], "o", color=MUTED, markersize=5.5, markerfacecolor=SURFACE, zorder=4)
        ax.plot([40, INF_X], [level, level], color=ORANGE, linewidth=0.9, alpha=0.55)
        draw_limit(ax, INF_X, level, lo, hi, y_lo, y_hi, clip_lo, clip_hi, ORANGE, f"∞: {level:.3f}  [{lo:.3f}, {hi:.3f}]")
        ax.set_xlim(0, INF_X + 4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, INF_X])
        ax.set_xticklabels(["0", "10", "20", "30", "40", "50", "60", "∞"])
        ax.set_title(titles[q], fontsize=9)
        ax.legend(loc=where[q], frameon=False, fontsize=6.8)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.97, (
        "Stockfish, near-ties counted as ties.\n\n"
        "Curves fitted on depths 5–25; depths 1–4\n(hollow) are left out as too shallow.\n"
        "Solid over the data, dotted beyond it.\n\n"
        "∞ (orange): the level each curve tends to,\n"
        f"with its 95% bootstrap interval ({len(bl)} of {B}\n"
        "resamples: those in which every curve has\na level). A triangle: the interval runs past\n"
        "the plotted range, to the value printed.\n\n"
        "s: the gamma kernel tends to the same level\nat infinite depth as at depth 0."),
        fontsize=8.3, color=INK2, va="top", linespacing=1.3)
    for ax in (axes[1, 0], axes[1, 1], axes[0, 2]):
        ax.set_xlabel("Stockfish depth")
    fig.suptitle("Adjusted fits: depths 5–25 only, and a gamma kernel for s", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "adjusted_parameters.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
