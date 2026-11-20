"""Step 17: Stockfish at infinite depth against Lc0.

Shiku's hypothesis: Lc0 is trying to recreate the same distribution of relative error and only reassigns
values. Both engines on the same 264 positions, the same common scale and the same zero-one-inflated
logit-normal (near-ties counted as ties). Every share curve here is the model curve
F(t) = p0 + middle * Phi((logit t - m) / s), so the comparison is like for like.
Stockfish at infinite depth: the levels of step 16 (curves fitted on depths 5-25; gamma kernel for s, with
the S-curve for s as the alternative). Lc0: its five parameters at each node level (closed form).
Compared, with paired bootstrap intervals (positions resampled once, the same for both engines):
  - the five parameters: Stockfish at depth 25 and at infinite depth against Lc0 at 1 and 10,000 nodes;
  - the share of moves within a tolerance t, and its difference from Lc0 at every t;
  - distance between two distributions = the average gap between their share curves over t in [0, 1]
    (the area between the curves, in share of moves); the largest gap is reported too. Also how the distance
    to Lc0 changes with Stockfish's depth, and how Lc0's distance to Stockfish at infinite depth changes
    with Lc0's node count.
Outputs: out/lc0_comparison.csv, out/lc0_shares.csv, out/lc0_distance_by_level.csv, out/lc0_comparison.png
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

f15 = importlib.import_module("15_infinite_depth")
f16 = importlib.import_module("16_adjusted_infinite_depth")
ALL, GRID, B, SNAP = f15.ALL, f15.GRID, f16.B, f16.SNAP
TOL = (0.0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.99)
BLUE, ORANGE, AQUA = f15.BLUE, f15.ORANGE, "#1baf7a"
INK, INK2, MUTED, GRID_C, SURFACE = f15.INK, f15.INK2, f15.MUTED, f15.GRID_C, f15.SURFACE


def curve(par):
    return f15.share_curve(par["p0"], par["middle"], par["m"], par["s"])


def gaps(Fa, Fb):
    """Average gap over t in [0, 1] (trapezoid rule) and largest gap between two share curves."""
    diff = np.abs(Fa - Fb)
    return float(np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(GRID))), float(diff.max())


def at(ser, li):
    return {q: float(ser[q][li]) for q in ALL}


def ci(vals):
    vals = [v for v in vals if v is not None]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    S = f16.stockfish_limits()
    boots, d = S["boots"], S["d"]
    lc = c.load_lc0()
    assert lc.pos_ids == S["run"].pos_ids, "the two runs must hold the same positions in the same order"
    xl = lc.x()
    snap = lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e))
    per_lc = [[snap(c.relative_errors(xl[p][li:li + 1])[0]) for p in lc.pos_ids] for li in range(len(lc.levels))]
    lc_full = f15.series(per_lc, np.arange(len(lc.pos_ids)))
    lc_reps = [f15.series(per_lc, idx) for idx in boots]
    top, d25 = len(lc.levels) - 1, len(d) - 1
    sf_full, sf_reps = S["full"], S["reps"]
    inf = {k: f16.levels(S["fits"], k) for k in ("s", "s_logistic")}
    inf_b = {k: [f16.levels(bf, k) for bf in S["boot_fits"]] for k in ("s", "s_logistic")}
    lc10, lc1, sf25 = at(lc_full, top), at(lc_full, 0), at(sf_full, d25)
    lc10_b = [at(r, top) for r in lc_reps]
    sf25_b = [at(r, d25) for r in sf_reps]
    F = {"sf_inf": curve(inf["s"]), "sf_inf_scurve": curve(inf["s_logistic"]), "sf25": curve(sf25), "lc10": curve(lc10),
         "lc1": curve(lc1)}
    Fb = {"sf_inf": [None if L is None else curve(L) for L in inf_b["s"]],
          "sf_inf_scurve": [None if L is None else curve(L) for L in inf_b["s_logistic"]],
          "sf25": [curve(b) for b in sf25_b], "lc10": [curve(b) for b in lc10_b], "lc1": [curve(at(r, 0)) for r in lc_reps]}

    # 1. the five parameters
    rows = []
    for q in ALL:
        diff_inf = [None if L is None else L[q] - l10[q] for L, l10 in zip(inf_b["s"], lc10_b)]
        diff_scurve = [None if L is None else L[q] - l10[q] for L, l10 in zip(inf_b["s_logistic"], lc10_b)]
        diff_25 = [s25[q] - l10[q] for s25, l10 in zip(sf25_b, lc10_b)]
        rows.append([q, sf25[q], *ci([b[q] for b in sf25_b]), inf["s"][q], *ci([None if L is None else L[q] for L in inf_b["s"]]),
                     inf["s_logistic"][q], lc1[q], lc10[q], *ci([b[q] for b in lc10_b]),
                     inf["s"][q] - lc10[q], *ci(diff_inf), inf["s_logistic"][q] - lc10[q], *ci(diff_scurve),
                     sf25[q] - lc10[q], *ci(diff_25)])

    # 2. shares within a tolerance, and their differences from Lc0
    share_rows = []
    for t in TOL:
        i = int(np.argmin(np.abs(GRID - t)))
        row = [t] + [float(F[k][i]) for k in ("sf25", "sf_inf", "sf_inf_scurve", "lc10")]
        for k in ("sf_inf", "sf_inf_scurve", "sf25"):
            row += [float(F[k][i] - F["lc10"][i]), *ci([None if fa is None else float(fa[i] - fl[i]) for fa, fl in zip(Fb[k], Fb["lc10"])])]
        share_rows.append(row)

    # 3. distances between distributions
    dist_rows = []
    for k, name in (("sf_inf", "Stockfish infinite depth (gamma kernel for s)"), ("sf_inf_scurve", "Stockfish infinite depth (S-curve for s)"),
                    ("sf25", "Stockfish depth 25"), ("lc1", "Lc0 1 node")):
        avg, mx = gaps(F[k], F["lc10"])
        pairs = [gaps(fa, fl) for fa, fl in zip(Fb[k], Fb["lc10"]) if fa is not None]
        dist_rows.append([name, avg, *ci([p[0] for p in pairs]), mx, *ci([p[1] for p in pairs])])
    avg, mx = gaps(F["sf_inf"], F["sf25"])
    pairs = [gaps(fa, fb) for fa, fb in zip(Fb["sf_inf"], Fb["sf25"]) if fa is not None]
    dist_rows.append(["(Stockfish infinite depth against Stockfish depth 25)", avg, *ci([p[0] for p in pairs]), mx,
                      *ci([p[1] for p in pairs])])
    by_level = []
    for li, dep in enumerate(d.astype(int)):
        pairs = [gaps(curve(at(r, li)), fl) for r, fl in zip(sf_reps, Fb["lc10"])]
        by_level.append(["Stockfish depth -> Lc0 10000 nodes", int(dep), gaps(curve(at(sf_full, li)), F["lc10"])[0],
                         *ci([p[0] for p in pairs])])
    for li, nodes in enumerate(lc.levels):
        pairs = [gaps(curve(at(r, li)), fi) for r, fi in zip(lc_reps, Fb["sf_inf"]) if fi is not None]
        by_level.append(["Lc0 nodes -> Stockfish infinite depth", int(nodes), gaps(curve(at(lc_full, li)), F["sf_inf"])[0],
                         *ci([p[0] for p in pairs])])

    # how well the model curve follows each engine's observed curve (largest gap)
    obs = lambda per_pos: np.mean([(e[None, :] <= GRID[:, None] + 1e-12).mean(axis=1) for e in per_pos], axis=0)
    fit_gap = {"Lc0 10000 nodes": float(np.max(np.abs(obs(per_lc[top]) - F["lc10"]))),
               "Stockfish depth 25": float(np.max(np.abs(obs(S["per_depth"][d25]) - F["sf25"])))}

    r4 = lambda row: [round(v, 4) if isinstance(v, float) else v for v in row]
    c.write_csv(c.OUT / "lc0_comparison.csv", [r4(r) for r in rows],
                ["parameter", "sf_depth25", "lo95", "hi95", "sf_infinite", "lo95", "hi95", "sf_infinite_scurve_s",
                 "lc0_1_node", "lc0_10000_nodes", "lo95", "hi95", "sf_inf_minus_lc0", "lo95", "hi95",
                 "sf_inf_scurve_s_minus_lc0", "lo95", "hi95", "sf_d25_minus_lc0", "lo95", "hi95"])
    c.write_csv(c.OUT / "lc0_shares.csv", [r4(r) for r in share_rows],
                ["t", "sf_depth25", "sf_infinite", "sf_infinite_scurve_s", "lc0_10000_nodes", "sf_inf_minus_lc0", "lo95", "hi95",
                 "sf_inf_scurve_s_minus_lc0", "lo95", "hi95", "sf_d25_minus_lc0", "lo95", "hi95"])
    c.write_csv(c.OUT / "lc0_distance_by_level.csv", [r4(r) for r in by_level] + [r4(["distance to Lc0 10000 nodes"] + r) for r in dist_rows],
                ["comparison", "level", "average_gap", "lo95", "hi95", "largest_gap", "lo95", "hi95"])
    report(rows, share_rows, dist_rows, by_level, fit_gap)
    plot(F, Fb, by_level, inf_b)


def report(rows, share_rows, dist_rows, by_level, fit_gap):
    print("1. The five parameters (near-ties counted as ties); 95% bootstrap intervals, paired for the differences")
    print(f"   {'':<7} {'SF d25':>7} {'SF infinite':>23} {'S-curve s':>9} {'Lc0 1':>7} {'Lc0 10000':>22} "
          f"{'SF inf - Lc0':>26} {'(S-curve s)':>26} {'SF d25 - Lc0':>25}")
    for r in rows:
        print(f"   {r[0]:<7} {r[1]:>7.3f} {r[4]:>7.3f} [{r[5]:>6.3f},{r[6]:>6.3f}] {r[7]:>9.3f} {r[8]:>7.3f} "
              f"{r[9]:>7.3f} [{r[10]:.3f},{r[11]:.3f}] {r[12]:>+8.3f} [{r[13]:+.3f},{r[14]:+.3f}] "
              f"{r[15]:>+8.3f} [{r[16]:+.3f},{r[17]:+.3f}] {r[18]:>+7.3f} [{r[19]:+.3f},{r[20]:+.3f}]")
    print("\n2. Share of moves within t (model curves) and differences from Lc0 at 10,000 nodes")
    print(f"   {'t':>5} {'SF d25':>7} {'SF inf':>7} {'S-curve s':>9} {'Lc0':>6} {'SF inf - Lc0':>26} {'(S-curve s)':>26} {'SF d25 - Lc0':>25}")
    for r in share_rows:
        print(f"   {r[0]:>5} {r[1]:>7.3f} {r[2]:>7.3f} {r[3]:>9.3f} {r[4]:>6.3f} {r[5]:>+8.3f} [{r[6]:+.3f},{r[7]:+.3f}] "
              f"{r[8]:>+8.3f} [{r[9]:+.3f},{r[10]:+.3f}] {r[11]:>+7.3f} [{r[12]:+.3f},{r[13]:+.3f}]")
    print("\n3. Distance to Lc0 at 10,000 nodes: average gap between the share curves over t in [0, 1], and the largest gap")
    for r in dist_rows:
        print(f"   {r[0]:<55} average {r[1]:.3f} [{r[2]:.3f}, {r[3]:.3f}]   largest {r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}]")
    for name in ("Stockfish depth -> Lc0 10000 nodes", "Lc0 nodes -> Stockfish infinite depth"):
        sel = [r for r in by_level if r[0] == name]
        print(f"   {name}: " + "  ".join(f"{r[1]}: {r[2]:.3f}" for r in sel[::4] + [sel[-1]]))
    print("\n   largest gap between model and observed curve: " + ", ".join(f"{k} {v:.3f}" for k, v in fit_gap.items()))


def plot(F, Fb, by_level, inf_b):
    f15.style()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    ax = axes[0]
    band = np.array([f for f in Fb["sf_inf"] if f is not None])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
    ax.fill_between(GRID, lo, hi, color=ORANGE, alpha=0.15, linewidth=0)
    ax.plot(GRID, F["sf_inf"], color=ORANGE, linewidth=2.0, zorder=4, label="Stockfish, infinite depth (95% band)")
    ax.plot(GRID, F["sf_inf_scurve"], color=ORANGE, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4,
            label="Stockfish, infinite depth, S-curve for s")
    ax.plot(GRID, F["sf25"], color=BLUE, linewidth=1.5, zorder=3, label="Stockfish, depth 25")
    ax.plot(GRID, F["lc10"], color=AQUA, linewidth=2.0, zorder=3, label="Lc0, 10,000 nodes")
    ax.text(0.985, 0.03, "every curve jumps to 1 at t = 1", ha="right", va="bottom", fontsize=7.5, color=INK2, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("error tolerance t")
    ax.set_ylabel("share of legal moves with e ≤ t")
    ax.set_title("Share of moves within a tolerance (model curves)", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)

    ax = axes[1]
    for k, color, label in (("sf_inf", ORANGE, "Stockfish infinite depth − Lc0"), ("sf25", BLUE, "Stockfish depth 25 − Lc0")):
        diffs = np.array([fa - fl for fa, fl in zip(Fb[k], Fb["lc10"]) if fa is not None])
        lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
        ax.fill_between(GRID, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(GRID, F[k] - F["lc10"], color=color, linewidth=1.8, label=f"{label} (95% band)")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("error tolerance t")
    ax.set_ylabel("difference in share of moves with e ≤ t")
    ax.set_title("Stockfish minus Lc0, by tolerance", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)

    ax = axes[2]
    sel = [r for r in by_level if r[0].startswith("Stockfish depth")]
    dep = np.array([r[1] for r in sel]); val = np.array([r[2] for r in sel])
    lo_, hi_ = np.array([r[3] for r in sel]), np.array([r[4] for r in sel])
    sh = dep < f16.FIRST
    ax.errorbar(dep[sh], val[sh], yerr=[val[sh] - lo_[sh], hi_[sh] - val[sh]], fmt="o", color=MUTED, markersize=3,
                elinewidth=0.8, capsize=0, markerfacecolor=SURFACE, label="depths 1–4")
    ax.errorbar(dep[~sh], val[~sh], yerr=[val[~sh] - lo_[~sh], hi_[~sh] - val[~sh]], fmt="o", color=BLUE, markersize=3,
                elinewidth=0.8, capsize=0, label="Stockfish at each depth (95% band)")
    pairs = [gaps(fa, fl)[0] for fa, fl in zip(Fb["sf_inf"], Fb["lc10"]) if fa is not None]
    lo_i, hi_i = np.percentile(pairs, [2.5, 97.5])
    inf_gap = gaps(F["sf_inf"], F["lc10"])[0]
    ax.plot([32, 32], [lo_i, hi_i], color=ORANGE, linewidth=1.4, zorder=4)
    ax.plot(32, inf_gap, "o", color=ORANGE, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5,
            label=f"infinite depth: {inf_gap:.3f} [{lo_i:.3f}, {hi_i:.3f}]")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 32])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "25", "∞"])
    ax.set_xlim(0, 34)
    ax.set_ylim(0, max(hi_i, hi_.max()) * 1.08)
    ax.set_xlabel("Stockfish depth")
    ax.set_ylabel("average gap to Lc0's share curve")
    ax.set_title("Distance from Stockfish to Lc0 (10,000 nodes)", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=7.3)
    fig.suptitle("Stockfish at infinite depth against Lc0 (same positions, same scale, near-ties counted as ties)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "lc0_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
