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


def near_best_by_level():
    fits = read_csv("depth_trend_fits.csv")
    limits = read_csv("limits.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, run, color in ((axes[0], c.load_stockfish(), SF_COLOR), (axes[1], c.load_lc0(), LC0_COLOR)):
        eng = tr.Engine(run)
        counts, n = eng.counts, eng.n_legal
        share = counts[:, :, 1] / n[:, None]
        mean = share.mean(axis=0)
        se = share.std(axis=0, ddof=1) / np.sqrt(len(n))
        xs = np.array(run.levels, float)
        ax.fill_between(xs, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, mean, "o", color=color, markersize=3.5, label=f"{run.engine}, measured (95% band)")
        sel = next(r for r in fits if r["engine"] == run.engine and r["threshold"] == "0.1"
                   and r["quantity"] == "mean_share" and r["selected"] == "True")
        lim = next(r for r in limits if r["engine"] == run.engine and r["threshold"] == "0.1" and r["quantity"] == "mean_share")
        p_all = tr.fit_form(sel["shape"], eng.x[eng.use], mean[eng.use])[0]
        xx = np.linspace(eng.x[eng.use].min(), eng.x[eng.use].max() * (2.2 if run.level_name == "depth" else 1.15), 200)
        ax.plot(10 ** xx if run.level_name == "nodes" else xx, tr.predict(sel["shape"], p_all, xx), "-", color=INK2,
                linewidth=1.2, label=f"selected trend ({sel['shape']})")
        if lim["limit"]:
            ax.axhline(float(lim["limit"]), color=INK2, linestyle=":", linewidth=1.2,
                       label=f"limit {float(lim['limit']):.3f} [{float(lim['limit_lo95']):.3f}, {float(lim['limit_hi95']):.3f}]")
        if run.level_name == "depth":
            ax.axvspan(20.5, 25.5, color=GRID, alpha=0.5, linewidth=0)
            ax.text(23, 0.205, "held out", ha="center", color=INK2, fontsize=8)
            ax.set_xlabel("Stockfish depth")
        else:
            ax.set_xscale("log")
            ax.axvspan(1200, 12000, color=GRID, alpha=0.5, linewidth=0)
            ax.text(3500, 0.205, "held out", ha="center", color=INK2, fontsize=8)
            ax.set_xlabel("Lc0 nodes per move")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    axes[0].set_ylabel("Share of legal moves with e ≤ 0.10")
    axes[0].set_ylim(0.2, 0.42)
    fig.suptitle("Near-best moves by search effort (264 positions, common scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "near_best_by_level.png", dpi=200)
    plt.close(fig)


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
    which = sys.argv[1:] or ["near_best_by_level", "bb_parameters", "count_fit_top"]
    for name in which:
        globals()[name]()
        print("wrote", name)
