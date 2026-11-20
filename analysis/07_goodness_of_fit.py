"""7. How well the model fits, and what p0 and p1 are made of (Stockfish, near-ties counted as ties).

1. The distribution at each depth:
   - largest and average gap between the fitted and observed share of moves within a tolerance (over t in [0, 1]);
   - share of moves misplaced: half the summed absolute difference between observed and fitted shares in the 12 bands
     (exactly 0, ten bands of 0.1, exactly 1), i.e. the total variation distance between the binned distributions;
   - R-squared of the fitted band profile over the ten bands strictly between 0 and 1;
   - a test of exact fit that respects the positions: the observed-minus-fitted band shares, their covariance from the
     bootstrap resamples of positions (refitting the model in each), and the Wald statistic on nine of the ten middle
     bands (they sum to zero), with its chi-square(9) p-value. Moves of one position are not independent, so a
     chi-square or Kolmogorov-Smirnov test treating the 7,283 moves as independent would be invalid.
   - at depth 25, observed minus fitted share in each middle band, with z from the bootstrap standard error.
2. The depth functions (depths 5-25): R-squared (plain and weighted by 1/se^2) of each curve and of a straight line,
   and paired-bootstrap tests of whether each parameter still changes between depths 20 and 25, and whether s falls
   between depth 22 and depth 25. The p-value is twice the smaller share of resamples on either side of zero.
3. What p0 and p1 are made of: every position contributes its own best and worst move (1/n of its moves each),
   positions in which every move has the same value count all their moves as best, and the rest are genuine ties.
Outputs: out/fit_by_depth.csv, out/band_residuals_depth25.csv, out/depth_function_checks.csv, out/p0_p1_split.csv
"""
import numpy as np
from scipy.stats import chi2

import common as c
import model as mdl

GRID = mdl.GRID


def fit_stats(per_pos):
    par = mdl.at(mdl.parameters([per_pos], np.arange(len(per_pos))), 0)
    return (par, mdl.observed_curve(per_pos), mdl.model_curve(par), mdl.observed_bands(per_pos),
            mdl.band_probs(par["p0"], par["middle"], par["m"], par["s"]))


def all_equal_kind(x, cp):
    if len(x) == 1:
        return "one legal move"
    if np.all(x == 0):
        return "every move loses (expected score 0)"
    if np.all(x == 1):
        return "every move wins (expected score 1)"
    if np.all(cp == 0):
        return "every move scores 0.00"
    return "every move has the same score"


def r2(y, yhat, w=None):
    w = np.ones_like(y) if w is None else w
    ybar = np.sum(w * y) / np.sum(w)
    return float(1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - ybar) ** 2))


def boot_p(deltas):
    deltas = np.asarray(deltas)
    return max(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)), 1 / len(deltas))


def main():
    S = mdl.limits()
    d, full, se, fits, per_depth, boots, run = S["d"], S["full"], S["se"], S["fits"], S["per_depth"], S["boots"], S["run"]

    # ---- 1. the distribution at each depth
    rows = []
    print("1. Fit of the zero-one-inflated logit-normal at each depth")
    print(f"   {'depth':>5} {'largest gap':>12} {'average gap':>12} {'misplaced':>10} {'band R2':>8} {'Wald W':>8} {'p':>9}")
    for li, dep in enumerate(d.astype(int)):
        par, oc, mc, ob, mb = fit_stats(per_depth[li])
        gap = np.abs(oc - mc)
        largest = float(gap.max())
        average = float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(GRID)))
        misplaced = float(0.5 * np.sum(np.abs(ob - mb)))
        band_r2 = r2(ob[1:11], mb[1:11])
        diffs = np.array([mdl.band_diff([per_depth[li][i] for i in idx]) for idx in boots])
        dvec = (ob - mb)[1:10]
        W = float(dvec @ np.linalg.solve(np.cov(diffs[:, 1:10], rowvar=False), dvec))
        p = float(chi2.sf(W, df=9))
        rows.append([int(dep), largest, average, misplaced, band_r2, W, p])
        if dep in (1, 5, 10, 15, 20, 25):
            print(f"   {dep:>5} {largest:>12.3f} {average:>12.3f} {misplaced:>10.3f} {band_r2:>8.2f} {W:>8.1f} {p:>9.1e}")
        if dep == 25:
            z = (ob - mb) / diffs.std(axis=0, ddof=1).clip(min=1e-12)
            band_rows = [[name, float(ob[i]), float(mb[i]), float(ob[i] - mb[i]), float(z[i]) if 0 < i < 11 else ""]
                         for i, (name, _) in enumerate(mdl.BINS)]
    c.write_csv(c.OUT / "fit_by_depth.csv", [[r[0]] + [round(v, 6) for v in r[1:6]] + [f"{r[6]:.3g}"] for r in rows],
                ["depth", "largest_gap", "average_gap", "share_misplaced", "band_r2", "wald_w", "p_exact_fit"])
    c.write_csv(c.OUT / "band_residuals_depth25.csv",
                [[r[0]] + [round(v, 6) if isinstance(v, float) else v for v in r[1:]] for r in band_rows],
                ["band", "observed", "fitted", "observed_minus_fitted", "z"])
    sel = [r for r in rows if r[0] >= mdl.FIRST]
    print(f"   depths 5-25: largest gap {min(r[1] for r in sel):.3f}-{max(r[1] for r in sel):.3f}, average gap "
          f"{min(r[2] for r in sel):.3f}-{max(r[2] for r in sel):.3f}, misplaced {min(r[3] for r in sel):.3f}-{max(r[3] for r in sel):.3f}, "
          f"band R2 {min(r[4] for r in sel):.2f}-{max(r[4] for r in sel):.2f}, largest p {max(r[6] for r in sel):.1e}")
    print("   depth 25, observed minus fitted by band (z): "
          + "  ".join(f"{r[0]}: {r[3]:+.3f}" + (f" ({r[4]:+.1f})" if r[4] != "" else "") for r in band_rows))
    li25 = list(d).index(25)
    par25 = mdl.at(full, li25)
    middle_obs = sum(r[1] for r in band_rows[1:11])
    print(f"   depth 25, in-between moves within 0.1 of either end: observed {(band_rows[1][1] + band_rows[10][1]) / middle_obs:.3f}, "
          f"model {(band_rows[1][2] + band_rows[10][2]) / par25['middle']:.3f}")

    # ---- 2. the depth functions
    print(f"\n2. Depth functions on depths {mdl.FIRST}-25: R-squared (plain / weighted) against a straight line, and change near depth 25")
    k = d >= mdl.FIRST
    i20, i22, i25 = list(d).index(20), list(d).index(22), list(d).index(25)
    checks = []
    for q, key in (("p0", "p0"), ("middle", "middle"), ("p1", None), ("m", "m"), ("s", "s"), ("s", "s_logistic")):
        y, w = full[q][k], 1 / se[q][k] ** 2
        yhat = mdl.curve_values(q, fits, d[k], "s_logistic" if key == "s_logistic" else "s")
        X = np.column_stack([np.ones(k.sum()), d[k]])
        line = X @ np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        shape = "1 - p0 - middle" if key is None else mdl.SHAPES[key]
        db = [r[q][i25] - r[q][i20] for r in S["reps"]]
        lo, hi = np.percentile(db, [2.5, 97.5])
        checks.append([q, shape, r2(y, yhat), r2(y, yhat, w), r2(y, line), r2(y, line, w), "20-25",
                       float(full[q][i25] - full[q][i20]), float(lo), float(hi), boot_p(db)])
        print(f"   {q:<7} {shape:<16} R2 {checks[-1][2]:.3f} / {checks[-1][3]:.3f}   line {checks[-1][4]:.3f} / {checks[-1][5]:.3f}   "
              f"depth 20 -> 25: {checks[-1][7]:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {checks[-1][10]:.3f}")
    db = [r["s"][i25] - r["s"][i22] for r in S["reps"]]
    lo, hi = np.percentile(db, [2.5, 97.5])
    checks.append(["s", "observed", "", "", "", "", "22-25", float(full["s"][i25] - full["s"][i22]), float(lo), float(hi), boot_p(db)])
    print(f"   s       depth 22 -> 25: {checks[-1][7]:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {checks[-1][10]:.3f} "
          f"(observed peak at depth {int(d[np.argmax(full['s'])])})")
    c.write_csv(c.OUT / "depth_function_checks.csv", [[round(v, 6) if isinstance(v, float) else v for v in r] for r in checks],
                ["parameter", "depth_function", "r2", "r2_weighted", "r2_line", "r2_line_weighted", "change_depths",
                 "change", "change_lo95", "change_hi95", "p_no_change"])

    # ---- 3. what p0 and p1 are made of
    print("\n3. p0 and p1: own best / worst move (1/n each), positions with every move equal, and genuine ties")
    pieces = np.array([c.piece_count(run.fens[p]) for p in run.pos_ids])
    split_rows = []
    for dep in (5, 25):
        li = list(d).index(dep)
        per = per_depth[li]
        n = np.array([len(e) for e in per])
        same = np.array([np.all(e == 0) for e in per])
        single = np.where(same, 0.0, 1.0 / n)
        ties0 = np.array([0.0 if s_ else (np.sum(e == 0) - 1) / len(e) for e, s_ in zip(per, same)])
        ties1 = np.array([0.0 if s_ else (np.sum(e == 1) - 1) / len(e) for e, s_ in zip(per, same)])
        groups = (("all", np.ones(len(per), bool)),) + tuple((g[0], (pieces >= g[1]) & (pieces <= g[2])) for g in mdl.PHASES)
        for name, g in groups:
            vals = [float(np.mean([np.mean(e == 0) for e, keep in zip(per, g) if keep])), float(single[g].mean()),
                    float(same[g].mean()), float(ties0[g].mean()),
                    float(np.mean([np.mean(e == 1) for e, keep in zip(per, g) if keep])), float(single[g].mean()),
                    float(ties1[g].mean()), int(same[g].sum())]
            split_rows.append([dep, name, int(g.sum())] + vals)
            print(f"   depth {dep:>2} {name:<12} p0 {vals[0]:.3f} = own best {vals[1]:.3f} + all moves equal {vals[2]:.3f} + ties {vals[3]:.3f}"
                  f"   p1 {vals[4]:.3f} = own worst {vals[5]:.3f} + ties {vals[6]:.3f}   ({vals[7]} positions with all moves equal)")
        if dep == 25:
            for p, s_ in zip(run.pos_ids, same):
                if s_:
                    print(f"      position {p}: {len(run.cp[p][li])} legal moves, {c.piece_count(run.fens[p])} pieces, "
                          f"{all_equal_kind(S['xs'][p][li], run.cp[p][li])}")
    c.write_csv(c.OUT / "p0_p1_split.csv", [[round(v, 6) if isinstance(v, float) else v for v in r] for r in split_rows],
                ["depth", "pieces", "positions", "p0", "own_best_move", "all_moves_equal", "ties_with_best", "p1",
                 "own_worst_move", "ties_with_worst", "positions_with_all_moves_equal"])


if __name__ == "__main__":
    main()
