"""Step 20: how well the model fits, and two structural points about the formula (Stockfish, near-ties as ties).

1. The distribution at each depth (zero-one-inflated logit-normal, closed form, each position weighted equally):
   - largest gap and average gap between the fitted and observed share of moves within a tolerance
     (over t in [0, 1]), in share of moves;
   - share of moves misplaced: half the summed absolute difference between observed and fitted shares in the
     12 bands (exactly 0, ten bands of 0.1, exactly 1), i.e. the total variation distance between the binned
     distributions;
   - R-squared of the fitted band profile over the ten bands strictly between 0 and 1;
   - a test of exact fit that respects the positions: the observed-minus-fitted band shares, their covariance
     from the 200 bootstrap resamples of positions (refitting the model in each), and the Wald statistic
     W = d' S^-1 d on nine of the ten middle bands (they sum to zero), with its chi-square(9) p-value.
     Moves of one position are not independent, so a chi-square or Kolmogorov-Smirnov test on the 7,283
     moves as if they were would be invalid.
2. The depth functions (fitted on depths 5-25, as in step 16): R-squared (plain and weighted by 1/se^2) for
   each curve and for a straight line, and paired-bootstrap tests of whether each parameter still changes
   between depths 20 and 25, and whether s falls between depth 22 and depth 25.
3. Structure of p0 and p1: every position contributes its own best and worst move (1/n each, unless all its
   moves are valued the same), so part of p0 and p1 is mechanical. The rest is genuine ties. By depth and phase.
Output: out/goodness_of_fit.csv, printed summary.
"""
import importlib

import numpy as np
from scipy.stats import chi2

import common as c

f13 = importlib.import_module("13_depth_functions")
f15 = importlib.import_module("15_infinite_depth")
f16 = importlib.import_module("16_adjusted_infinite_depth")
f18 = importlib.import_module("18_phase_breakdown")
GRID = f15.GRID
ALL = f15.ALL


def fit_stats(per_pos):
    """per_pos: error arrays of one depth -> parameters, curves and band shares."""
    q = f15.series([per_pos], np.arange(len(per_pos)))
    par = {k: float(v[0]) for k, v in q.items()}
    obs_curve = np.mean([(e[None, :] <= GRID[:, None] + 1e-12).mean(axis=1) for e in per_pos], axis=0)
    mod_curve = f15.share_curve(par["p0"], par["middle"], par["m"], par["s"])
    obs_b = f15.observed_bands(per_pos)
    mod_b = f15.band_probs(par["p0"], par["middle"], par["m"], par["s"])
    return par, obs_curve, mod_curve, obs_b, mod_b


def r2(y, yhat, w=None):
    w = np.ones_like(y) if w is None else w
    ybar = np.sum(w * y) / np.sum(w)
    return float(1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - ybar) ** 2))


def boot_p(deltas):
    """Two-sided bootstrap p-value for 'no change': twice the smaller tail beyond zero (at least 1/B)."""
    deltas = np.asarray(deltas)
    tail = min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return max(2 * tail, 1 / len(deltas))


def main():
    S = f16.stockfish_limits()
    d, full, se, fits, per_depth, boots, run = S["d"], S["full"], S["se"], S["fits"], S["per_depth"], S["boots"], S["run"]
    rows = []

    # ---- 1. the distribution at each depth
    print("1. Fit of the zero-one-inflated logit-normal at each depth")
    print(f"   {'depth':>5} {'largest gap':>12} {'average gap':>12} {'misplaced':>10} {'band R2':>8} {'Wald W':>8} {'p':>9}")
    for li, dep in enumerate(d.astype(int)):
        par, oc, mc, ob, mb = fit_stats(per_depth[li])
        gap = np.abs(oc - mc)
        largest = float(gap.max())
        average = float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(GRID)))
        misplaced = float(0.5 * np.sum(np.abs(ob - mb)))
        mid = slice(1, 11)
        band_r2 = r2(ob[mid], mb[mid])
        diffs = []
        for idx in boots:
            _, _, _, ob_b, mb_b = fit_stats([per_depth[li][i] for i in idx])
            diffs.append((ob_b - mb_b)[1:10])          # nine of the ten middle bands
        diffs = np.array(diffs)
        dvec = (ob - mb)[1:10]
        cov = np.cov(diffs, rowvar=False)
        W = float(dvec @ np.linalg.solve(cov, dvec))
        p = float(chi2.sf(W, df=9))
        rows.append(["distribution", int(dep), "", largest, average, misplaced, band_r2, W, p])
        if dep in (1, 5, 10, 15, 20, 25):
            print(f"   {dep:>5} {largest:>12.3f} {average:>12.3f} {misplaced:>10.3f} {band_r2:>8.2f} {W:>8.1f} {p:>9.1e}")
        if dep == 25:
            z = dvec / diffs.std(axis=0, ddof=1)
            names = [n for n, _ in f15.BINS[1:10]]
            print("   depth 25, observed minus fitted share by band (z, using the bootstrap standard error):")
            print("     " + "  ".join(f"{n}: {v:+.3f} ({zz:+.1f})" for n, v, zz in zip(names, dvec, z))
                  + f"  0.9-1: {(ob - mb)[10]:+.3f}")
    sel = [r for r in rows if r[0] == "distribution" and r[1] >= f16.FIRST]
    print(f"   depths 5-25: largest gap {min(r[3] for r in sel):.3f}-{max(r[3] for r in sel):.3f}, "
          f"average gap {min(r[4] for r in sel):.3f}-{max(r[4] for r in sel):.3f}, misplaced {min(r[5] for r in sel):.3f}-"
          f"{max(r[5] for r in sel):.3f}, band R2 {min(r[6] for r in sel):.2f}-{max(r[6] for r in sel):.2f}, "
          f"largest p {max(r[8] for r in sel):.1e}")

    # ---- 2. the depth functions
    print("\n2. Depth functions on depths 5-25: R-squared (plain / weighted by 1/se^2), and a straight line for comparison")
    k = d >= f16.FIRST
    for q, key in (("p0", "p0"), ("middle", "middle"), ("p1", None), ("m", "m"), ("s", "s"), ("s", "s_logistic")):
        y, w = full[q][k], 1 / se[q][k] ** 2
        yhat = f16.curve_values(q, fits, d[k], "s_logistic" if key == "s_logistic" else "s")
        X = np.column_stack([np.ones(k.sum()), d[k]])
        line = X @ np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        shape = "derived from p0 and middle" if key is None else f16.SHAPES[key]
        rows.append(["depth function", q, shape, r2(y, yhat), r2(y, yhat, w), r2(y, line), r2(y, line, w), "", ""])
        print(f"   {q:<7} {shape:<27} R2 {r2(y, yhat):.3f} / {r2(y, yhat, w):.3f}    straight line {r2(y, line):.3f} / {r2(y, line, w):.3f}")

    print("\n   Is each parameter still changing at the end of the data? Paired bootstrap, same positions at both depths")
    i20, i22, i25 = list(d).index(20), list(d).index(22), list(d).index(25)
    for q in ALL:
        delta = full[q][i25] - full[q][i20]
        db = [r[q][i25] - r[q][i20] for r in S["reps"]]
        lo, hi = np.percentile(db, [2.5, 97.5])
        rows.append(["change 20->25", q, "", float(delta), float(lo), float(hi), boot_p(db), "", ""])
        print(f"   {q:<7} depth 20 -> 25: {delta:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  p = {boot_p(db):.3f}")
    delta = full["s"][i25] - full["s"][i22]
    db = [r["s"][i25] - r["s"][i22] for r in S["reps"]]
    lo, hi = np.percentile(db, [2.5, 97.5])
    rows.append(["change 22->25", "s", "", float(delta), float(lo), float(hi), boot_p(db), "", ""])
    peak = int(d[np.argmax(full["s"])])
    print(f"   s       depth 22 -> 25: {delta:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  p = {boot_p(db):.3f}   (observed peak at depth {peak})")

    # ---- 3. how much of p0 and p1 is mechanical
    print("\n3. p0 and p1: each position's own best and worst move (1/n each) against genuine ties")
    pieces = np.array([c.piece_count(run.fens[p]) for p in run.pos_ids])
    for dep in (5, 25):
        li = list(d).index(dep)
        per = per_depth[li]
        same = np.array([np.all(e == 0) for e in per])
        n = np.array([len(e) for e in per])
        floor0 = np.where(same, 1.0, 1.0 / n)
        floor1 = np.where(same, 0.0, 1.0 / n)
        p0 = np.array([np.mean(e == 0) for e in per])
        p1 = np.array([np.mean(e == 1) for e in per])
        groups = (("all", np.ones(len(per), bool)),) + tuple((g[0], (pieces >= g[1]) & (pieces <= g[2])) for g in f18.GROUPS)
        for name, sel_ in groups:
            vals = [p0[sel_].mean(), floor0[sel_].mean(), p1[sel_].mean(), floor1[sel_].mean()]
            rows.append(["p0 p1 structure", dep, name, *[float(v) for v in vals], ""])
            print(f"   depth {dep:>2} {name:<12} p0 {vals[0]:.3f} = own best {vals[1]:.3f} + ties {vals[0]-vals[1]:.3f}   "
                  f"p1 {vals[2]:.3f} = own worst {vals[3]:.3f} + ties {vals[2]-vals[3]:.3f}   (all moves equal in {same[sel_].sum()} positions)")
    c.write_csv(c.OUT / "goodness_of_fit.csv",
                [[round(v, 5) if isinstance(v, float) else v for v in r] for r in rows],
                ["block", "depth_or_parameter", "detail", "v1", "v2", "v3", "v4", "v5", "v6"])
    print("\n   (goodness_of_fit.csv: distribution rows = largest gap, average gap, misplaced, band R2, W, p; "
          "depth-function rows = R2, weighted R2, line R2, weighted line R2; change rows = delta, lo, hi, p; "
          "structure rows = p0, own best, p1, own worst)")


if __name__ == "__main__":
    main()
