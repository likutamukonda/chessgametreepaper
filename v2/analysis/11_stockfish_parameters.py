"""Step 11: how the parameters of the two leading distributions change with Stockfish depth.

At every depth 1-25, the relative errors of all moves are pooled over positions (each position weighted
equally, as in step 10) and fitted with
  ZOI beta          P(e=0) = p0, P(e=1) = p1, otherwise Beta(a, b)
  ZOI logit-normal  P(e=0) = p0, P(e=1) = p1, otherwise logit(e) ~ Normal(m, s)
The spikes p0 and p1 are the same in both (the share of moves exactly at 0 and exactly at 1).
Derived: the beta part's mean a / (a + b) and precision a + b; the logit-normal part's median 1 / (1 + exp(-m)).
The logit-normal's m and s are also given with errors within SNAP = 0.001 of 0 or 1 counted as 0 or 1: a move
0.00001 away from the best is about -11.5 on the logit scale, so a handful of such near-ties moves s a lot.
For reference, the share of moves that sit at 0 and at 1 by construction (one best and one worst move per
position; every move at 0 when all moves are equal).
Error bands: 95% intervals from refitting on positions resampled with replacement (bootstrap).
Outputs: out/stockfish_parameters.csv, out/stockfish_parameters.png.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.special import betaln, expit, logit

import common as c

B, SEED = 200, 20261120
SNAP = 1e-3
SF_COLOR, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fit_level(per_pos, idx, start=None):
    """per_pos: list of per-position error arrays; idx: positions to use (with repeats for the bootstrap)."""
    e = np.concatenate([per_pos[i] for i in idx])
    w = np.concatenate([np.full(len(per_pos[i]), 1.0 / len(per_pos[i])) for i in idx])
    W = w.sum()
    p0, p1 = w[e == 0].sum() / W, w[e == 1].sum() / W
    inner = (e > 0) & (e < 1)
    x, wi = e[inner], w[inner] / w[inner].sum()
    lx, l1 = np.log(x), np.log1p(-x)
    nll = lambda th: -np.sum(wi * ((np.exp(th[0]) - 1) * lx + (np.exp(th[1]) - 1) * l1 - betaln(*np.exp(th))))
    starts = [start] if start is not None else [[0, 0], [-1, -1], [1, 1]]
    r = min((optimize.minimize(nll, np.array(s, float), method="L-BFGS-B", bounds=[(-7, 7), (-7, 7)]) for s in starts),
            key=lambda r: r.fun)
    a, b = np.exp(r.x)
    z = logit(x)
    m = float(np.sum(wi * z))
    s = float(np.sqrt(np.sum(wi * (z - m) ** 2)))
    keep = (e >= SNAP) & (e <= 1 - SNAP)                 # near-ties counted as ties: drop them from the middle
    zs, ws = logit(e[keep]), w[keep] / w[keep].sum()
    m_snap = float(np.sum(ws * zs))
    s_snap = float(np.sqrt(np.sum(ws * (zs - m_snap) ** 2)))
    return {"p0": p0, "p1": p1, "middle": 1 - p0 - p1, "a": a, "b": b, "beta_mean": a / (a + b), "precision": a + b,
            "m": m, "s": s, "logitnormal_median": float(expit(m)), "m_snapped": m_snap, "s_snapped": s_snap}, r.x


def construction_floor(per_pos):
    """Share of moves at 0 and at 1 by construction (each position weighted equally)."""
    at0 = np.mean([1.0 if np.all(e == 0) else 1.0 / len(e) for e in per_pos])
    at1 = np.mean([0.0 if np.all(e == 0) else 1.0 / len(e) for e in per_pos])
    return at0, at1


def main():
    run = c.load_stockfish()
    xs = run.x()
    depths = run.levels
    per_depth = [[c.relative_errors(xs[p][li:li + 1])[0] for p in run.pos_ids] for li in range(len(depths))]
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    keys = ["p0", "p1", "middle", "a", "b", "beta_mean", "precision", "m", "s", "logitnormal_median", "m_snapped", "s_snapped"]
    rows, est, lo, hi, floors = [], {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}, []
    for li, d in enumerate(depths):
        point, theta = fit_level(per_depth[li], np.arange(n_pos))
        bs = [fit_level(per_depth[li], idx, start=theta)[0] for idx in boots]
        f0, f1 = construction_floor(per_depth[li])
        floors.append((f0, f1))
        row = [d, round(f0, 5), round(f1, 5)]
        for k in keys:
            v = np.array([b_[k] for b_ in bs])
            l_, h_ = np.percentile(v, [2.5, 97.5])
            est[k].append(point[k]); lo[k].append(l_); hi[k].append(h_)
            row += [round(point[k], 5), round(l_, 5), round(h_, 5)]
        rows.append(row)
    header = ["depth", "at0_by_construction", "at1_by_construction"] + [f"{k}{s}" for k in keys for s in ("", "_lo95", "_hi95")]
    c.write_csv(c.OUT / "stockfish_parameters.csv", rows, header)
    plot(depths, est, lo, hi, floors)
    print("Stockfish, pooled fits by depth (95% bootstrap interval in brackets)")
    for k in keys:
        print(f"  {k:<19} " + "  ".join(f"d{d}: {est[k][d - 1]:.3f}" for d in (1, 5, 10, 15, 20, 25))
              + f"   [d25: {lo[k][24]:.3f}-{hi[k][24]:.3f}]")
    print("  by construction at 0: " + "  ".join(f"d{d}: {floors[d - 1][0]:.3f}" for d in (1, 5, 10, 15, 20, 25)))
    print("  by construction at 1: " + "  ".join(f"d{d}: {floors[d - 1][1]:.3f}" for d in (1, 5, 10, 15, 20, 25)))


def plot(depths, est, lo, hi, floors):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    d = np.array(depths)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)

    def line(ax, k, color, label):
        ax.fill_between(d, lo[k], hi[k], color=color, alpha=0.15, linewidth=0)
        ax.plot(d, est[k], "-o", color=color, markersize=2.5, linewidth=1.6, label=label)

    ax = axes[0, 0]
    line(ax, "p0", SF_COLOR, "p0: share exactly at 0 (as good as the best)")
    line(ax, "p1", ORANGE, "p1: share exactly at 1 (as bad as the worst)")
    ax.plot(d, [f[0] for f in floors], ":", color=SF_COLOR, linewidth=1.2, label="at 0 by construction")
    ax.plot(d, [f[1] for f in floors], ":", color=ORANGE, linewidth=1.2, label="at 1 by construction")
    ax.set_title("Spikes (both distributions)", fontsize=9)
    ax.set_ylim(0, 0.16)

    ax = axes[0, 1]
    line(ax, "a", SF_COLOR, "a (shape at the best end)")
    line(ax, "b", ORANGE, "b (shape at the worst end)")
    ax.axhline(1, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(25, 1.02, "1: no pile-up at that end", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_title("Beta middle: shape parameters\n(below 1 = moves pile up at that end)", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower left", frameon=False, fontsize=7)

    ax = axes[0, 2]
    line(ax, "precision", SF_COLOR, "precision a + b")
    ax.set_title("Beta middle: precision a + b\n(smaller = middle pushed to 0 and 1)", fontsize=9)
    ax.set_ylim(0.6, 1.5)

    ax = axes[1, 0]
    line(ax, "beta_mean", SF_COLOR, "beta middle: mean a/(a+b)")
    line(ax, "logitnormal_median", ORANGE, "logit-normal middle: median")
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_title("Centre of the middle part", fontsize=9)
    ax.set_ylim(0.3, 0.7)

    ax = axes[1, 1]
    line(ax, "m", SF_COLOR, "m, as measured")
    line(ax, "m_snapped", ORANGE, "m, near-ties (< 0.001) counted as ties")
    ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_title("Logit-normal middle: centre m\n(log of distance-to-best over distance-to-worst)", fontsize=9)
    ax.set_ylim(-0.6, 0.6)

    ax = axes[1, 2]
    line(ax, "s", SF_COLOR, "s, as measured")
    line(ax, "s_snapped", ORANGE, "s, near-ties (< 0.001) counted as ties")
    ax.set_title("Logit-normal middle: spread s\n(larger = middle pushed to 0 and 1)", fontsize=9)
    ax.set_ylim(2.0, 4.4)

    for ax in axes.flat:
        handles = ax.get_legend_handles_labels()[0]
        if len(handles) >= 2 and ax.get_legend() is None:
            ax.legend(loc="upper left", frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Stockfish depth")
    fig.suptitle("Parameters of the zero-one-inflated beta and logit-normal, fitted separately at each depth "
                 "(264 positions; bands: 95% bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "stockfish_parameters.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
