"""Figures for the v2 analysis, written to out/. Run after steps 1-3.

  near_best_by_level.png   share of near-best moves by Stockfish depth and Lc0 node count, with the
                           selected depth trend and its limit
  bb_parameters.png        beta-binomial mean (mu) and precision (phi) by level, both engines
  count_fit_top.png        observed distribution of the share at the top level against the fitted models
"""
import csv
import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

cd = importlib.import_module("02_count_distributions")
tr = importlib.import_module("03_depth_trends")

# Reference palette (light mode): categorical slots 1-2, text and surface tokens.
SF_COLOR, LC0_COLOR = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2})


def read_csv(name):
    with open(c.OUT / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


TOLERANCE_COLORS = {0.05: "#86b6ef", 0.10: "#2a78d6", 0.25: "#104281"}   # ordinal blue ramp, steps 250/450/650


def near_best_by_level():
    keeping = [r for r in read_csv("result_keeping.csv") if r["rule"] == "argmax"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, run in ((axes[0], c.load_stockfish()), (axes[1], c.load_lc0())):
        counts, n = c.near_best_counts(run)
        xs = np.array(run.levels, float)
        for ti, t in enumerate(c.THRESHOLDS):
            ax.plot(xs, (counts[:, :, ti] / n[:, None]).mean(axis=0), "-o", color=TOLERANCE_COLORS[t], markersize=2.5,
                    linewidth=1.6, label=f"e ≤ {t}")
        keep = sorted([r for r in keeping if r["engine"] == run.engine], key=lambda r: int(r["level_index"]))
        ax.plot(xs, [float(r["share_keeping"]) for r in keep], "--", color=INK, linewidth=1.4,
                label="keeps the result (limit of every tolerance below ½)")
        if run.level_name == "depth":
            ax.set_xlabel("Stockfish depth")
        else:
            ax.set_xscale("log")
            ax.set_xlabel("Lc0 nodes per move")
        ax.set_title(run.engine, fontsize=9)
    axes[0].set_ylabel("Share of legal moves")
    axes[0].set_ylim(0.1, 0.65)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle("Near-best moves by search effort (264 positions; standard error about ±0.02)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(c.OUT / "near_best_by_level.png", dpi=200)
    plt.close(fig)


LEVEL_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue ramp, steps 250-650


def share_curves(errors, grid):
    """errors: list of per-position arrays of e -> average over positions of the share with e <= t, for t in grid."""
    return np.mean([(e[None, :] <= grid[:, None] + 1e-12).mean(axis=1) for e in errors], axis=0)


def limit_errors(wdl_results):
    """Per-position errors at infinite depth from win/draw/loss results (1 / 0.5 / 0): each e is 0, 1/2 or 1."""
    out = []
    for r in wdl_results:
        best, worst = r.max(), r.min()
        out.append(np.zeros(len(r)) if best == worst else (best - r) / (best - worst))
    return out


def share_by_tolerance():
    """Share of legal moves with relative error <= t, against t, at five levels per engine, plus the
    infinite-depth curve estimated from each engine's own win/draw/loss verdicts at its top level."""
    rk = importlib.import_module("05_result_keeping")
    grid = np.concatenate([[0.0], np.linspace(1e-4, 1.0, 600)])
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharey="row")
    for col, (run, zip_path, level_col, offset) in enumerate(((c.load_stockfish(), c.SF_RUN, "depth", 1),
                                                               (c.load_lc0(), c.LC0_RUN, "level", 0))):
        xs = run.x()
        wdl = rk.load_wdl(zip_path, level_col, offset)
        lim = share_curves(limit_errors([rk.results(wdl[p])[-1] for p in run.pos_ids]), grid)
        picks = [0, 6, 12, 18, 24]
        curves = [(run.levels[li], share_curves([c.relative_errors(xs[p][li:li + 1])[0] for p in run.pos_ids], grid))
                  for li in picks]
        for row, xmax in enumerate((1.0, 0.25)):
            ax = axes[row, col]
            for (level, y), color in zip(curves, LEVEL_COLORS):
                name = f"depth {level}" if run.level_name == "depth" else f"{level:,} node{'s' if level > 1 else ''}"
                ax.plot(grid, y, color=color, linewidth=1.8, label=name)
            ax.step(grid, lim, where="post", color=INK, linestyle="--", linewidth=1.3,
                    label="infinite depth (estimated)")
            ax.set_xlim(0, xmax)
            for t in c.THRESHOLDS:
                if t <= xmax:
                    ax.axvline(t, color=GRID, linewidth=1.0, zorder=0)
            if row == 0:
                ax.set_title(run.engine, fontsize=10)
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(0, 0.7)
                ax.set_xlabel("error tolerance t")
        axes[0, col].legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2, columnspacing=1.2)
    axes[0, 0].set_ylabel("share of legal moves with e ≤ t")
    axes[1, 0].set_ylabel("share of legal moves with e ≤ t\n(tolerances up to 0.25)")
    fig.suptitle("Share of moves within an error tolerance, by search effort (264 positions, common scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "share_by_tolerance.png", dpi=200)
    plt.close(fig)
    return curves


def bb_parameters():
    rows = [r for r in read_csv("count_fits.csv") if r["model"] == "beta-binomial"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.6), sharex="col")
    for col, t in enumerate(("0.05", "0.1", "0.25")):
        for row, par in enumerate(("mu", "phi")):
            ax = axes[row, col]
            for eng, color in (("Stockfish", SF_COLOR), ("Lc0", LC0_COLOR)):
                ser = sorted([r for r in rows if r["engine"] == eng and r["threshold"] == t], key=lambda r: int(r["level_index"]))
                ax.plot(range(1, 26), [float(r[par]) for r in ser], "-o", color=color, markersize=2.5, linewidth=1.5,
                        label=f"{eng} ({'depth' if eng == 'Stockfish' else 'node level'} k)")
            ax.set_title(f"{'mean chance mu' if par == 'mu' else 'precision phi'}, e ≤ {t}", fontsize=9)
            if row == 1:
                ax.set_xlabel("step k (Stockfish depth k; Lc0 level k = 1 ... 10,000 nodes)", fontsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Beta-binomial parameters by search effort", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "bb_parameters.png", dpi=200)
    plt.close(fig)


def count_fit_top():
    rows = read_csv("count_fits.csv")
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.8), sharey="row")
    for r_i, run in enumerate((c.load_stockfish(), c.load_lc0())):
        counts, n_legal = c.near_best_counts(run)
        keep = n_legal >= 2
        n = n_legal[keep] - 1
        z = np.log(n_legal[keep]) - np.log(n_legal[keep]).mean()
        top = len(run.levels) - 1
        for t_i, t in enumerate(c.THRESHOLDS):
            ax = axes[r_i, t_i]
            k = counts[keep, top, t_i] - 1
            share = (k + 1) / (n + 1)
            obs, _ = np.histogram(share, bins=bins)
            ax.bar(bins[:-1] + 0.05, obs / len(k), width=0.09, color=GRID, edgecolor=INK2, linewidth=0.5, label="observed")
            for model, color, style in (("beta-binomial", SF_COLOR, "-o"), ("zero-inflated BB", LC0_COLOR, "--s")):
                r = next(x for x in rows if x["engine"] == run.engine and x["level_index"] == str(top)
                         and x["threshold"] == str(t) and x["model"] == model)
                theta = theta_from_row(model, r)
                pm = cd.pmf_matrix(model, theta, n, z)
                exp = np.zeros(len(bins) - 1)
                for i in range(len(n)):
                    s = (np.arange(n[i] + 1) + 1) / (n[i] + 1)
                    idx = np.clip(np.digitize(s, bins) - 1, 0, len(bins) - 2)
                    np.add.at(exp, idx, pm[i, :n[i] + 1])
                ax.plot(bins[:-1] + 0.05, exp / len(k), style, color=color, markersize=3, linewidth=1.3,
                        label=f"{model} (KS p {float(r['ks_p']):.2f})")
            ax.set_title(f"{run.engine} {'depth 25' if run.engine == 'Stockfish' else '10,000 nodes'}, e ≤ {t}", fontsize=9)
            ax.legend(frameon=False, fontsize=6.5)
            if r_i == 1:
                ax.set_xlabel("share of legal moves that are near-best")
    axes[0, 0].set_ylabel("fraction of positions"); axes[1, 0].set_ylabel("fraction of positions")
    fig.suptitle("Distribution of the near-best share at the top level: observed and fitted", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "count_fit_top.png", dpi=200)
    plt.close(fig)


def theta_from_row(model, r):
    mu, phi = float(r["mu"]), float(r["phi"])
    th = [np.log(mu / (1 - mu)), np.log(phi)]
    if model == "zero-inflated BB":
        pi0 = min(max(float(r["pi0"]), 1e-9), 1 - 1e-9)
        th.append(np.log(pi0 / (1 - pi0)))
    return np.array(th)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["near_best_by_level", "share_by_tolerance", "bb_parameters", "count_fit_top"]
    for name in which:
        globals()[name]()
        print("wrote", name)
