"""Step 14: which two of the spike quantities should carry the depth functions? (Stockfish, S-curve default)

The zero-one-inflated logit-normal has three shares that always add to 1: p0 (moves exactly at 0), p1
(exactly at 1) and the middle (1 - p0 - p1). Two are free; the third follows. Three ways to choose them:
    A  p0 and p1 fitted, middle derived                      (used so far)
    B  p0 and the middle fitted, p1 = 1 - p0 - middle        (Shiku's proposal)
    C  ends = p0 + p1 and split = p0 / (p0 + p1) fitted,     (the usual parametrisation of zero-one inflation)
       p0 = ends * split, p1 = ends * (1 - split)
Every quantity gets the logistic S-curve y = A + (L - A) / (1 + exp(-k (d - d0))) as a function of depth d,
fitted by weighted least squares (weights 1 / bootstrap se^2, as in step 13). For each quantity: in-sample
and held-out error (fit on depths 1-20, predict 21-25; in units of se), the level L it settles at, its 95%
bootstrap interval and how often the curve levels off at all. For p1 under A, B and C: how well the implied
p1 curve matches the measured p1, its implied level, and whether it stays between 0 and 1 out to depth 100.
Both versions of the data (as measured; near-ties counted as ties).
Outputs: out/spike_parameterisation.csv, out/spike_parameterisation.png
"""
import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

f13 = importlib.import_module("13_depth_functions")
for q in ("middle", "ends", "split"):
    f13.RANGE[q] = (0.0, 1.0)
LOG = f13.F["logistic"]
B, SEED, SNAP = 200, 20261120, 1e-3
QUANTS = ("p0", "p1", "middle", "ends", "split")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def spike_series(per_depth, idx):
    p0, p1 = [], []
    for per_pos in per_depth:
        e = np.concatenate([per_pos[i] for i in idx])
        w = np.concatenate([np.full(len(per_pos[i]), 1.0 / len(per_pos[i])) for i in idx])
        p0.append(w[e == 0].sum() / w.sum())
        p1.append(w[e == 1].sum() / w.sum())
    p0, p1 = np.array(p0), np.array(p1)
    return {"p0": p0, "p1": p1, "middle": 1 - p0 - p1, "ends": p0 + p1, "split": p0 / (p0 + p1)}


def p1_from(scheme, fits, d):
    if scheme == "A":
        return LOG(d, *fits["p1"])
    if scheme == "B":
        return 1 - LOG(d, *fits["p0"]) - LOG(d, *fits["middle"])
    return LOG(d, *fits["ends"]) * (1 - LOG(d, *fits["split"]))


def main():
    run = c.load_stockfish()
    xs = run.x()
    d = np.array(run.levels, float)
    train = d <= 20
    far = np.arange(1, 101, dtype=float)
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    rows, prow, plot_data = [], [], {}
    for kind in ("measured", "near-ties"):
        snap = (lambda e: e) if kind == "measured" else (lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e)))
        per_depth = [[snap(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids] for li in range(len(d))]
        full = spike_series(per_depth, np.arange(n_pos))
        reps = [spike_series(per_depth, idx) for idx in boots]
        se = {q: np.array([r[q] for r in reps]).std(axis=0, ddof=1) for q in QUANTS}
        fits, fits_train, boot_fits = {}, {}, {q: [] for q in QUANTS}
        for q in QUANTS:
            y = full[q]
            p_all, L, levels = f13.fit_shape("logistic", d, y, se[q], q)
            p_tr, _, _ = f13.fit_shape("logistic", d[train], y[train], se[q][train], q)
            fits[q], fits_train[q] = p_all, p_tr
            z_in = (LOG(d, *p_all) - y) / se[q]
            z_out = (LOG(d[~train], *p_tr) - y[~train]) / se[q][~train]
            lims = []
            for r in reps:
                pb, Lb, lb = f13.fit_shape("logistic", d, r[q], se[q], q, start=p_all)
                boot_fits[q].append(pb if (pb is not None and lb) else None)
                if pb is not None and lb:
                    lims.append(Lb)
            ci = np.percentile(lims, [2.5, 97.5]) if len(lims) >= 20 else (np.nan, np.nan)
            rows.append([kind, q, round(float(np.sqrt(np.mean(z_in ** 2))), 3), round(float(np.sqrt(np.mean(z_out ** 2))), 3),
                         round(L, 4), levels, round(float(ci[0]), 4), round(float(ci[1]), 4), round(len(lims) / B, 3),
                         round(float(ci[1] - ci[0]), 4)])
        for scheme in ("A", "B", "C"):
            curve = p1_from(scheme, fits, d)
            z_in = (curve - full["p1"]) / se["p1"]
            z_out = (p1_from(scheme, fits_train, d[~train]) - full["p1"][~train]) / se["p1"][~train]
            far_curve = p1_from(scheme, fits, far)
            need = {"A": ("p1",), "B": ("p0", "middle"), "C": ("ends", "split")}[scheme]
            lims = [p1_from(scheme, {q: boot_fits[q][b] for q in need}, np.array([1e4]))[0]
                    for b in range(B) if all(boot_fits[q][b] is not None for q in need)]
            ci = np.percentile(lims, [2.5, 97.5]) if len(lims) >= 20 else (np.nan, np.nan)
            prow.append([kind, scheme, round(float(np.sqrt(np.mean(z_in ** 2))), 3), round(float(np.sqrt(np.mean(z_out ** 2))), 3),
                         round(float(far_curve[-1]), 4), round(float(ci[0]), 4), round(float(ci[1]), 4),
                         bool(np.all((far_curve >= 0) & (far_curve <= 1))), round(len(lims) / B, 3)])
        plot_data[kind] = (full, se, fits)
    c.write_csv(c.OUT / "spike_parameterisation.csv",
                [["quantity"] + r for r in rows] + [["implied p1"] + r for r in prow],
                ["table", "version", "quantity_or_scheme", "rms_z_in_sample", "rms_z_heldout", "level_or_p1_at_depth_100",
                 "levels_off_or_lo95", "lo95_or_hi95", "hi95_or_valid_to_100", "share_levelling_off", "interval_width"])
    print("S-curve fitted to each spike quantity (rms z: error in units of each depth's standard error)")
    for kind in ("measured", "near-ties"):
        print(f"\n=== {kind} ===")
        for r in rows:
            if r[0] == kind:
                print(f"  {r[1]:<7} in-sample {r[2]:.2f}  held-out {r[3]:.2f}  level {r[4]:.3f}  95% [{r[6]:.3f}, {r[7]:.3f}] "
                      f"(width {r[9]:.3f})  levels off in {r[8]:.0%} of resamples")
        for r in prow:
            if r[0] == kind:
                name = {"A": "A: p1 fitted directly", "B": "B: 1 - p0 - middle", "C": "C: ends x (1 - split)"}[r[1]]
                print(f"  p1 via {name:<22} in-sample {r[2]:.2f}  held-out {r[3]:.2f}  p1 at depth 100 {r[4]:.3f}  "
                      f"implied level 95% [{r[5]:.3f}, {r[6]:.3f}]  stays in 0-1 to depth 100: {r[7]}")
    plot(d, plot_data)


def plot(d, plot_data):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    full, se, fits = plot_data["near-ties"]
    dd = np.linspace(1, 60, 400)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    titles = {"p0": "p0: share exactly at 0", "middle": "middle: share strictly between 0 and 1",
              "ends": "ends: p0 + p1", "split": "split: p0 / (p0 + p1)", "p1": "p1: share exactly at 1"}
    for ax, q in zip(axes.flat, ("p0", "middle", "p1", "ends", "split")):
        ax.errorbar(d, full[q], yerr=1.96 * se[q], fmt="o", color=INK2, markersize=2.5, elinewidth=0.8, capsize=0, label="data (95% band)")
        yy = LOG(dd, *fits[q])
        ax.plot(dd[dd <= 25], yy[dd <= 25], color=BLUE, linewidth=1.6, label="S-curve fitted to this quantity")
        ax.plot(dd[dd >= 25], yy[dd >= 25], color=BLUE, linewidth=1.2, linestyle=(0, (2, 2)))
        if q == "p1":
            for scheme, color, label in (("B", ORANGE, "p1 = 1 - p0 - middle"), ("C", AQUA, "p1 = ends x (1 - split)")):
                yb = p1_from(scheme, fits, dd)
                ax.plot(dd[dd <= 25], yb[dd <= 25], color=color, linewidth=1.4, label=label)
                ax.plot(dd[dd >= 25], yb[dd >= 25], color=color, linewidth=1.1, linestyle=(0, (2, 2)))
        ax.axvspan(25, 60, color=GRID_C, alpha=0.35, linewidth=0)
        ax.set_title(titles[q], fontsize=9)
        ax.legend(loc={"p0": "upper left", "middle": "upper right", "p1": "upper left", "ends": "upper left",
                       "split": "lower right"}[q], frameon=False, fontsize=7)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 0.9, "Near-ties counted as ties.\nCurves dotted beyond depth 25,\nwhere there is no data.\n\n"
                    "Any two of p0, p1 and the middle\ndetermine the third.", fontsize=8.5, color=INK2, va="top")
    for ax in axes[1, :2]:
        ax.set_xlabel("Stockfish depth")
    axes[0, 2].set_xlabel("Stockfish depth")
    axes[0, 2].tick_params(labelbottom=True)
    fig.suptitle("S-curves for the spike quantities, and the p1 they imply", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "spike_parameterisation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
