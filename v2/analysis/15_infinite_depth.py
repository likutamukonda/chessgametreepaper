"""Step 15: the error distribution at infinite depth, from S-curves in depth (Stockfish).

Fitted with the logistic S-curve  y(d) = A + (L - A) / (1 + exp(-k (d - d0)))  as a function of depth d:
    p0      share of moves exactly at 0 (as good as the best)
    middle  share of moves strictly between 0 and 1
    m, s    mean and standard deviation of logit(e) = log(distance to the best / distance to the worst)
            over the moves strictly between
Calculated: p1 = 1 - p0 - middle (share exactly at 1).
As d goes to infinity each S-curve tends to its level L, which gives the zero-one-inflated logit-normal at
infinite depth:  P(e = 0) = p0, P(e = 1) = p1, otherwise logit(e) ~ Normal(m, s).
Its share of moves within a tolerance t is  F(t) = p0 + middle * Phi((logit t - m) / s)  for 0 < t < 1.

Benchmark: under perfect play every move wins, draws or loses, so e can only be 0, 1/2 or 1. Taking
Stockfish's own win/draw/loss verdicts at depth 25 (step 5) as the results gives the three shares such a
limit would have.

Data: errors with near-ties (within 0.001 of 0 or 1) counted as ties (the version in which s is stable);
the measured version is reported as a check. Uncertainty: 200 bootstrap resamples of the positions; within
each resample all four quantities are recomputed at every depth, refitted, and taken to infinite depth.
Resamples in which a curve does not level off are counted and left out of the intervals.
Outputs: out/infinite_depth_parameters.csv, out/infinite_depth_distribution.csv,
         out/infinite_depth_parameters.png, out/infinite_depth_distribution.png
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit, ndtr

import common as c

f05 = importlib.import_module("05_result_keeping")
f13 = importlib.import_module("13_depth_functions")
f13.RANGE["middle"] = (0.0, 1.0)
LOG = f13.F["logistic"]
B, SEED, SNAP = 200, 20261120, 1e-3
FITTED = ("p0", "middle", "m", "s")
ALL = ("p0", "middle", "p1", "m", "s")
GRID = np.concatenate([[0.0], np.linspace(1e-4, 0.9999, 2000)])
BINS = ([("exactly 0", None)]
        + [(f"{a:g}–{round(a + 0.1, 1):g}", (a, round(a + 0.1, 1))) for a in np.round(np.arange(0, 1, 0.1), 1)]
        + [("exactly 1", None)])
INF_X, CURVE_END = 70.0, 60.0
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df", "#fcfcfb"


def series(per_depth, idx):
    q = f13.parameters(per_depth, idx)
    return {"p0": q["p0"], "middle": 1 - q["p0"] - q["p1"], "p1": q["p1"], "m": q["m"], "s": q["s"]}


def share_curve(p0, middle, m, s, t=GRID):
    t = np.asarray(t, float)
    with np.errstate(divide="ignore"):
        inner = ndtr((logit(np.clip(t, 1e-300, 1 - 1e-16)) - m) / s)
    return np.where(t >= 1, 1.0, np.where(t <= 0, p0, p0 + middle * inner))


def band_probs(p0, middle, m, s):
    """Model share of moves in each error band: exactly 0, 0-0.1, ..., 0.9-1, exactly 1."""
    out = []
    for name, edges in BINS:
        if edges is None:
            out.append(p0 if name == "exactly 0" else 1 - p0 - middle)
            continue
        a, b = edges
        upper = 1.0 if b >= 1 else ndtr((logit(b) - m) / s)
        lower = 0.0 if a <= 0 else ndtr((logit(a) - m) / s)
        out.append(middle * (upper - lower))
    return np.array(out)


def observed_bands(per_pos):
    """Observed share of moves in each error band, each position weighted equally."""
    out = np.zeros(len(BINS))
    for e in per_pos:
        for i, (name, edges) in enumerate(BINS):
            if edges is None:
                out[i] += np.mean(e == (0.0 if name == "exactly 0" else 1.0))
            else:
                out[i] += np.mean((e > edges[0]) & (e <= edges[1]) & (e > 0) & (e < 1))
    return out / len(per_pos)


def perfect_play_shares(run, level_index):
    """Per position: shares of moves at e = 0, 1/2 and 1 if Stockfish's verdicts at that depth were the results."""
    wdl = f05.load_wdl(c.SF_RUN, "depth", 1)
    rows = []
    for p in run.pos_ids:
        r = f05.results(wdl[p])[level_index]
        best, worst = r.max(), r.min()
        e = np.zeros_like(r) if best == worst else (best - r) / (best - worst)
        rows.append([np.mean(e == 0), np.mean(e == 0.5), np.mean(e == 1)])
    return np.array(rows)


def run_version(kind, run, xs, d, boots):
    snap = (lambda e: e) if kind == "measured" else (lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e)))
    per_depth = [[snap(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids] for li in range(len(d))]
    full = series(per_depth, np.arange(len(run.pos_ids)))
    reps = [series(per_depth, idx) for idx in boots]
    se = {q: np.array([r[q] for r in reps]).std(axis=0, ddof=1) for q in ALL}
    fits, inf, levels_off = {}, {}, {}
    for q in FITTED:
        fits[q], inf[q], levels_off[q] = f13.fit_shape("logistic", d, full[q], se[q], q)
    inf["p1"] = 1 - inf["p0"] - inf["middle"]
    boot = []
    for r in reps:
        got = {}
        for q in FITTED:
            p, L, lv = f13.fit_shape("logistic", d, r[q], se[q], q, start=fits[q])
            if p is None or not lv:
                break
            got[q] = L
        else:
            got["p1"] = 1 - got["p0"] - got["middle"]
            boot.append(got)
    return {"full": full, "se": se, "fits": fits, "inf": inf, "levels_off": levels_off, "boot": boot,
            "per_depth": per_depth}


def main():
    run = c.load_stockfish()
    xs = run.x()
    d = np.array(run.levels, float)
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    res = {kind: run_version(kind, run, xs, d, boots) for kind in ("near-ties", "measured")}
    pp = perfect_play_shares(run, len(d) - 1)
    pp_mean = pp.mean(axis=0)
    pp_lo, pp_hi = np.percentile([pp[idx].mean(axis=0) for idx in boots], [2.5, 97.5], axis=0)

    rows = []
    for kind, r in res.items():
        for q in ALL:
            vals = [b_[q] for b_ in r["boot"]]
            lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) >= 20 else (np.nan, np.nan)
            rows.append([kind, q, "calculated" if q == "p1" else "fitted", float(r["full"][q][-1]),
                         float(r["inf"][q]), float(lo), float(hi), len(r["boot"]) / B,
                         "" if q == "p1" else r["levels_off"][q]])
    for name, v, lo, hi in zip(("share at e = 0", "share at e = 1/2", "share at e = 1"), pp_mean, pp_lo, pp_hi):
        rows.append(["perfect play, Stockfish's depth-25 verdicts as the results", name, "benchmark", "",
                     float(v), float(lo), float(hi), "", ""])
    c.write_csv(c.OUT / "infinite_depth_parameters.csv",
                [[round(v, 4) if isinstance(v, float) else v for v in row] for row in rows],
                ["version", "parameter", "fitted_or_calculated", "value_at_depth_25", "value_at_infinite_depth",
                 "lo95", "hi95", "share_of_resamples_with_all_four_levelling_off", "levels_off"])

    r = res["near-ties"]
    inf = r["inf"]
    F_inf = share_curve(inf["p0"], inf["middle"], inf["m"], inf["s"])
    F_boot = np.array([share_curve(b_["p0"], b_["middle"], b_["m"], b_["s"]) for b_ in r["boot"]])
    F_lo, F_hi = np.percentile(F_boot, [2.5, 97.5], axis=0)
    at = lambda li: {q: LOG(d[li], *r["fits"][q]) for q in FITTED}
    m25, m1 = at(len(d) - 1), at(0)
    F_25 = share_curve(m25["p0"], m25["middle"], m25["m"], m25["s"])
    F_1 = share_curve(m1["p0"], m1["middle"], m1["m"], m1["s"])
    F_obs = np.mean([(e[None, :] <= GRID[:, None] + 1e-12).mean(axis=1) for e in r["per_depth"][-1]], axis=0)
    F_pp = np.where(GRID < 0.5, pp_mean[0], pp_mean[0] + pp_mean[1])
    c.write_csv(c.OUT / "infinite_depth_distribution.csv",
                [[round(float(v), 5) for v in row] for row in zip(GRID, F_inf, F_lo, F_hi, F_25, F_obs, F_pp)],
                ["t", "share_infinite_depth", "lo95", "hi95", "share_depth25_model", "share_depth25_observed",
                 "share_perfect_play_benchmark"])
    bands = {"inf": band_probs(inf["p0"], inf["middle"], inf["m"], inf["s"]),
             "boot": np.array([band_probs(b_["p0"], b_["middle"], b_["m"], b_["s"]) for b_ in r["boot"]]),
             "d25": band_probs(m25["p0"], m25["middle"], m25["m"], m25["s"]),
             "obs": observed_bands(r["per_depth"][-1])}
    report(res, rows, inf, F_inf, F_lo, F_hi, F_25, F_obs, bands, pp_mean, pp_lo, pp_hi)
    plot_parameters(d, r)
    plot_distribution(F_inf, F_lo, F_hi, F_25, F_obs, F_1, F_pp, bands)


def report(res, rows, inf, F_inf, F_lo, F_hi, F_25, F_obs, bands, pp_mean, pp_lo, pp_hi):
    full = res["near-ties"]["full"]
    print("Near-ties version, measured at each depth: depth 1 -> depth 25")
    print("  " + "   ".join(f"{q} {full[q][0]:.3f} -> {full[q][-1]:.3f}" for q in ALL))
    print("\nLevels at infinite depth (S-curves in depth; p1 = 1 - p0 - middle)")
    for kind in ("near-ties", "measured"):
        print(f"\n  {kind}: all four S-curves level off in {len(res[kind]['boot'])} of {B} resamples")
        for r in rows:
            if r[0] == kind:
                flag = "" if r[8] in ("", True) else "   (full-data curve does not level off)"
                print(f"    {r[1]:<7} ({r[2]:<10}) depth 25: {r[3]:>7.3f}   infinite depth: {r[4]:>7.3f}   "
                      f"95% [{r[5]:.3f}, {r[6]:.3f}]{flag}")
    print("\nShare of moves within a tolerance t (near-ties version)")
    print(f"  {'t':>6} {'depth 25 observed':>18} {'depth 25 model':>15} {'infinite depth':>15} {'95% interval':>16}")
    for t in (0.0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.99):
        i = int(np.argmin(np.abs(GRID - t)))
        print(f"  {t:>6} {F_obs[i]:>18.3f} {F_25[i]:>15.3f} {F_inf[i]:>15.3f}   [{F_lo[i]:.3f}, {F_hi[i]:.3f}]")
    lo, hi = np.percentile(bands["boot"], [2.5, 97.5], axis=0)
    print("\nShare of moves by error band (near-ties version): observed d25 | model d25 -> infinite depth [95%]")
    for (name, _), o, a, b, l, h in zip(BINS, bands["obs"], bands["d25"], bands["inf"], lo, hi):
        print(f"  {name:<10} {o:.3f} | {a:.3f} -> {b:.3f}  [{l:.3f}, {h:.3f}]")
    med = float(np.exp(inf["m"]) / (1 + np.exp(inf["m"])))
    near_ends = (bands["inf"][1] + bands["inf"][-2]) / inf["middle"]
    print(f"\n  median error of a move strictly between 0 and 1 at infinite depth: {med:.3f} "
          f"(m = {inf['m']:.3f}, s = {inf['s']:.3f}); share of those within 0.1 of an end: {near_ends:.3f}")
    print("\nPerfect play, if Stockfish's depth-25 verdicts were the results (e can only be 0, 1/2 or 1)")
    for name, v, l, h in zip(("e = 0", "e = 1/2", "e = 1"), pp_mean, pp_lo, pp_hi):
        print(f"    {name:<8} {v:.3f}  95% [{l:.3f}, {h:.3f}]")


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def plot_parameters(d, r):
    style()
    dd = np.linspace(1, CURVE_END, 400)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.9))
    titles = {"p0": "p0: share exactly at 0 (fitted)", "middle": "middle: share strictly between 0 and 1 (fitted)",
              "p1": "p1 = 1 − p0 − middle (calculated)", "m": "m: centre of the log-ratio (fitted)",
              "s": "s: spread of the log-ratio (fitted)"}
    where = {"p0": "upper left", "middle": "lower left", "p1": "upper left", "m": "lower right", "s": "lower right"}
    for ax, q in zip(axes.flat, ALL):
        y, se = r["full"][q], r["se"][q]
        yy = (1 - LOG(dd, *r["fits"]["p0"]) - LOG(dd, *r["fits"]["middle"])) if q == "p1" else LOG(dd, *r["fits"][q])
        level = r["inf"][q]
        lo, hi = np.percentile([b_[q] for b_ in r["boot"]], [2.5, 97.5])
        # zoom on the data, the curve and the level; show the interval in full unless it runs far beyond
        base_lo = min(float((y - 1.96 * se).min()), float(yy.min()), level)
        base_hi = max(float((y + 1.96 * se).max()), float(yy.max()), level)
        span = base_hi - base_lo
        y_lo, y_hi = base_lo - 0.12 * span, base_hi + 0.12 * span
        clip_lo, clip_hi = lo < y_lo - 1.2 * span, hi > y_hi + 1.2 * span
        y_lo = y_lo if clip_lo else min(y_lo, lo - 0.06 * span)
        y_hi = y_hi if clip_hi else max(y_hi, hi + 0.06 * span)
        room = 0.03 * (y_hi - y_lo)
        top, bottom = (y_hi - room if clip_hi else hi), (y_lo + room if clip_lo else lo)
        ax.axvspan(25, 64, color=GRID_C, alpha=0.35, linewidth=0)
        ax.errorbar(d, y, yerr=1.96 * se, fmt="o", color=INK2, markersize=2.5, elinewidth=0.8, capsize=0,
                    label="each depth (95% band)")
        ax.plot(dd[dd <= 25], yy[dd <= 25], color=BLUE, linewidth=1.8,
                label="from the two S-curves" if q == "p1" else "S-curve")
        ax.plot(dd[dd >= 25], yy[dd >= 25], color=BLUE, linewidth=1.2, linestyle=(0, (2, 2)))
        ax.plot([25, INF_X], [level, level], color=ORANGE, linewidth=0.9, alpha=0.55)
        ax.plot([INF_X, INF_X], [bottom, top], color=ORANGE, linewidth=1.4)
        for end, clipped, val, mk, va in ((top, clip_hi, hi, "^", "top"), (bottom, clip_lo, lo, "v", "bottom")):
            if clipped:
                ax.plot(INF_X, end, marker=mk, color=ORANGE, markersize=6)
                ax.text(INF_X - 2.2, end, f"to {val:.2f}", ha="right", va=va, fontsize=7.5, color=INK2)
            else:
                ax.plot(INF_X, end, marker="_", color=ORANGE, markersize=8, markeredgewidth=1.4)
        ax.plot(INF_X, level, "o", color=ORANGE, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=5, label=f"∞: {level:.3f}  [{lo:.3f}, {hi:.3f}]")
        ax.set_xlim(0, INF_X + 4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, INF_X])
        ax.set_xticklabels(["0", "10", "20", "30", "40", "50", "60", "∞"])
        ax.set_title(titles[q], fontsize=9)
        ax.legend(loc=where[q], frameon=False, fontsize=7)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.97, (
        "Stockfish, near-ties counted as ties.\n\n"
        "Solid: S-curve fitted over depths 1–25.\n"
        "Dotted: the same curve beyond the data\n(shaded).\n\n"
        "∞ (orange): the level each curve tends to,\n"
        f"with its 95% bootstrap interval ({len(r['boot'])} of {B}\n"
        "resamples: those in which all four\nS-curves level off).\n"
        "A triangle: the interval runs past the\nplotted range, to the value printed."),
        fontsize=8.5, color=INK2, va="top", linespacing=1.35)
    for ax in (axes[1, 0], axes[1, 1], axes[0, 2]):
        ax.set_xlabel("Stockfish depth")
    fig.suptitle("The four fitted parameters and the calculated p1, and the level each tends to at infinite depth",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "infinite_depth_parameters.png", dpi=200)
    plt.close(fig)


def plot_distribution(F_inf, F_lo, F_hi, F_25, F_obs, F_1, F_pp, bands):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.fill_between(GRID, F_lo, F_hi, color=ORANGE, alpha=0.16, linewidth=0)
    ax.plot(GRID, F_inf, color=ORANGE, linewidth=2.0, zorder=4, label="infinite depth (95% band)")
    ax.plot(GRID, F_25, color=BLUE, linewidth=1.6, zorder=3, label="depth 25, model")
    ax.plot(GRID, F_obs, color=BLUE, linewidth=1.0, linestyle=(0, (2, 2)), zorder=3, label="depth 25, observed")
    ax.plot(GRID, F_1, color=MUTED, linewidth=1.1, zorder=2, label="depth 1, model")
    ax.plot(GRID, F_pp, color=INK, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2,
            label="perfect play, if Stockfish's depth-25\nverdicts are the true results")
    ax.text(0.985, 0.985, "every curve jumps to 1 at t = 1", ha="right", va="top", fontsize=7.5, color=INK2,
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("error tolerance t")
    ax.set_ylabel("share of legal moves with e ≤ t")
    ax.set_title("Share of moves within a tolerance", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax = axes[1]
    x = np.arange(len(BINS))
    lo, hi = np.percentile(bands["boot"], [2.5, 97.5], axis=0)
    top, cap = 0.45, 0.43
    ax.bar(x - 0.2, bands["d25"], width=0.38, color=BLUE, label="depth 25, model")
    ax.bar(x + 0.2, bands["inf"], width=0.38, color=ORANGE, label="infinite depth (95% interval)")
    ax.plot(x - 0.2, bands["obs"], "D", color=INK, markersize=3.5, linestyle="none", label="depth 25, observed")
    for xi, l, h in zip(x + 0.2, lo, hi):
        hh = min(h, cap)
        ax.plot([xi, xi], [l, hh], color=INK2, linewidth=0.9)
        ax.plot(xi, l, marker="_", color=INK2, markersize=5)
        if h > cap:
            ax.plot(xi, hh, marker="^", color=INK2, markersize=5)
            ax.text(xi - 0.3, hh, f"to {h:.2f}", ha="right", va="center", fontsize=7.5, color=INK2)
        else:
            ax.plot(xi, hh, marker="_", color=INK2, markersize=5)
    ax.set_ylim(0, top)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in BINS], rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("relative error e")
    ax.set_ylabel("share of legal moves")
    ax.set_title("Where the moves sit, by relative error", fontsize=9)
    ax.legend(loc="upper center", frameon=False, fontsize=7.5)
    fig.suptitle("Stockfish's error distribution at infinite depth, from the S-curve levels", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "infinite_depth_distribution.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
