"""9. Robustness checks (Stockfish; Lc0 only as a side check).

1. Scale. The steepness of the expected-score curve is a modelling choice. At depth 25 (near-ties counted as ties):
   the observed share of moves within 0.05, 0.10, 0.25 and 0.5 and the five parameters, for half, one and twice the
   Lichess steepness, and for Stockfish's own win/draw/loss scale (w + d/2) / 1000.
2. Near-ties. The analysis as measured (no snapping) against near-ties counted as ties: the depth-25 parameters, how
   much s moves between consecutive depths, and the levels at infinite depth (both depth functions for s).
   (The model comparison for both versions is in 01_candidate_distributions.py.)
3. First depth. The depth functions fitted from depth 1 and from depths 3-8 instead of 5: the levels at infinite depth.
4. Lc0. The same positions scored by Lc0 (10,000 nodes per move), on the same scale and with the same model: the observed
   share of moves within t against Stockfish at depth 25 (paired bootstrap intervals for the differences), the
   parameters at 1 and at 10,000 nodes, and how far Lc0's distribution moves between 1 and 10,000 nodes (average gap
   between the share curves over t in [0, 1]).
Outputs: out/robustness_scale.csv, out/robustness_near_ties.csv, out/robustness_first_depth.csv, out/robustness_lc0.csv
"""
import numpy as np

import common as c
import model as mdl

TOL = (0.05, 0.10, 0.25, 0.5)
ORDER = ("p0", "p1", "middle", "m", "s")


def shares(per_pos, t):
    return np.array([np.mean(e <= t + 1e-12) for e in per_pos])


def avg_gap(Fa, Fb):
    diff = np.abs(Fa - Fb)
    return float(np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(mdl.GRID)))


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def scale_check(S):
    run, boots, n = S["run"], S["boots"], len(S["run"].pos_ids)
    li = run.levels.index(25)
    rows = []
    print("1. Scale, depth 25: observed share within t (bootstrap se) and the parameters")
    for name, steep, own in (("steepness 0.5", 0.5, False), ("Lichess steepness", 1.0, False),
                             ("steepness 2", 2.0, False), ("Stockfish win/draw/loss", 1.0, True)):
        xs = run.x(steep, own)
        per = [mdl.snap(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids]
        for t in TOL:
            sh = shares(per, t)
            rows.append([name, f"within {t:g}", float(sh.mean()), float(np.std([sh[i].mean() for i in boots], ddof=1))])
        par = mdl.parameters([per], np.arange(n))
        reps = [mdl.parameters([per], i) for i in boots]
        for q in ORDER:
            rows.append([name, q, float(par[q][0]), float(np.std([r[q][0] for r in reps], ddof=1))])
        get = {r[1]: r[2] for r in rows if r[0] == name}
        print(f"   {name:<24} " + "  ".join(f"within {t:g} {get[f'within {t:g}']:.3f}" for t in TOL) + "   "
              + "  ".join(f"{q} {get[q]:.3f}" for q in ORDER))
    c.write_csv(c.OUT / "robustness_scale.csv", [[r[0], r[1], round(r[2], 6), round(r[3], 6)] for r in rows],
                ["scale", "quantity", "value_depth25", "se"])


def near_ties_check(S):
    rows = []
    print("\n2. Near-ties counted as ties against as measured")
    raw = mdl.errors_by_depth(S["run"], S["xs"], near_ties=False)
    for dep in (5, 25):
        per = raw[S["run"].levels.index(dep)]
        touched = float(np.mean([np.mean(((e > 0) & (e < mdl.SNAP)) | ((e < 1) & (e > 1 - mdl.SNAP))) for e in per]))
        rows.append(["near-ties as ties", f"share of moves within {mdl.SNAP:g} of 0 or 1 but not at it, depth {dep}",
                     touched, "", "", ""])
        print(f"   depth {dep}: the rule moves {touched:.2%} of moves (each position weighted equally)")
    for name, T in (("near-ties as ties", S), ("as measured", mdl.limits(near_ties=False))):
        d, full, se = T["d"], T["full"], T["se"]
        i25 = list(d).index(25)
        k = d >= mdl.FIRST
        for q in ORDER:
            rows.append([name, f"{q} at depth 25", float(full[q][i25]), float(se[q][i25]), "", ""])
        jump = float(np.mean(np.abs(np.diff(full["s"][k]))))
        rows.append([name, "mean |change in s| between consecutive depths 5-25", jump, "", "", ""])
        for s_key in ("s", "s_logistic"):
            inf = mdl.levels(T["fits"], s_key)
            bl = mdl.boot_levels(T, s_key)
            label = mdl.SHAPES[s_key]
            for q in ORDER:
                lo, hi = ci([L[q] for L in bl])
                rows.append([name, f"{q} at infinite depth ({label} for s)", inf[q], "", lo, hi])
            v = float(mdl.model_curve(inf, np.array([0.10]))[0])
            lo, hi = ci([mdl.model_curve(L, np.array([0.10]))[0] for L in bl])
            rows.append([name, f"within 0.1 at infinite depth ({label} for s)", v, "", lo, hi])
            rows.append([name, f"resamples with every level ({label} for s)", len(bl) / mdl.B, "", "", ""])
        got = {r[1]: r for r in rows if r[0] == name}
        print(f"   {name:<18} depth 25: " + "  ".join(f"{q} {got[f'{q} at depth 25'][2]:.3f}" for q in ORDER)
              + f"   mean |change in s| {jump:.3f}")
        for s_key in ("s", "s_logistic"):
            label = mdl.SHAPES[s_key]
            print(f"   {'':<18} infinite depth ({label} for s): "
                  + "  ".join(f"{q} {got[f'{q} at infinite depth ({label} for s)'][2]:.3f}" for q in ORDER)
                  + f"   within 0.1 {got[f'within 0.1 at infinite depth ({label} for s)'][2]:.3f}"
                  + f"   ({got[f'resamples with every level ({label} for s)'][2]:.0%} of resamples)")
    c.write_csv(c.OUT / "robustness_near_ties.csv",
                [[r[0], r[1]] + [round(v, 6) if isinstance(v, float) else v for v in r[2:]] for r in rows],
                ["version", "quantity", "value", "se", "lo95", "hi95"])


def first_depth_check(S):
    d, full, se = S["d"], S["full"], S["se"]
    rows = []
    print("\n3. Levels at infinite depth by the first depth used for the depth functions")
    print(f"   {'first':>5} {'s by':<13} " + " ".join(f"{q:>7}" for q in ORDER) + f" {'within 0.1':>11}")
    for first in (1, 3, 4, 5, 6, 7, 8):
        fits = mdl.fit_all(d, full, se, first)
        for s_key in ("s", "s_logistic"):
            L = mdl.levels(fits, s_key)
            vals = [L[q] for q in ORDER] + [float(mdl.model_curve(L, np.array([0.10]))[0])] if L else [np.nan] * 6
            rows.append([first, mdl.SHAPES[s_key]] + vals)
            print(f"   {first:>5} {mdl.SHAPES[s_key]:<13} " + " ".join(f"{v:>7.3f}" for v in vals[:5]) + f" {vals[5]:>11.3f}")
    c.write_csv(c.OUT / "robustness_first_depth.csv",
                [[r[0], r[1]] + [round(float(v), 5) for v in r[2:]] for r in rows],
                ["first_depth", "s_depth_function"] + [f"{q}_infinite" for q in ORDER] + ["within_0.1_infinite"])


def lc0_check(S):
    run, boots, per_sf = S["run"], S["boots"], S["per_depth"]
    lc = c.load_lc0()
    assert lc.pos_ids == run.pos_ids, "the two runs must hold the same positions in the same order"
    per_lc = mdl.errors_by_depth(lc)
    top, i25 = len(lc.levels) - 1, run.levels.index(25)
    sf25, lc10, lc1 = per_sf[i25], per_lc[top], per_lc[0]
    rows = []
    print(f"\n4. Lc0 ({lc.levels[top]:,} nodes) against Stockfish (depth 25), same positions and scale; "
          "paired 95% intervals for the differences")
    for t in (0.0, 0.05, 0.10, 0.25, 0.5):
        a, b = shares(sf25, t), shares(lc10, t)
        lo, hi = ci([a[i].mean() - b[i].mean() for i in boots])
        rows.append([f"share within {t:g}", float(a.mean()), float(b.mean()), float(shares(lc1, t).mean()),
                     float(a.mean() - b.mean()), lo, hi])
        print(f"   within {t:<5g} Stockfish {a.mean():.3f}   Lc0 {b.mean():.3f}   difference {a.mean() - b.mean():+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]   (Lc0 at 1 node {shares(lc1, t).mean():.3f})")
    n = len(run.pos_ids)
    P = {k: mdl.at(mdl.parameters([v], np.arange(n)), 0) for k, v in (("sf25", sf25), ("lc10", lc10), ("lc1", lc1))}
    reps = [(mdl.at(mdl.parameters([sf25], i), 0), mdl.at(mdl.parameters([lc10], i), 0)) for i in boots]
    for q in ORDER:
        lo, hi = ci([a[q] - b[q] for a, b in reps])
        rows.append([q, P["sf25"][q], P["lc10"][q], P["lc1"][q], P["sf25"][q] - P["lc10"][q], lo, hi])
    print("   parameters (Stockfish d25 / Lc0 10,000 nodes / Lc0 1 node): "
          + "  ".join(f"{q} {P['sf25'][q]:.3f} / {P['lc10'][q]:.3f} / {P['lc1'][q]:.3f}" for q in ORDER))
    g_obs = avg_gap(mdl.observed_curve(lc1), mdl.observed_curve(lc10))
    g_mod = avg_gap(mdl.model_curve(P["lc1"]), mdl.model_curve(P["lc10"]))
    g_sf = avg_gap(mdl.observed_curve(sf25), mdl.observed_curve(lc10))
    rows.append(["average gap, Lc0 1 node vs 10,000 nodes (observed curves)", g_obs, "", "", "", "", ""])
    rows.append(["average gap, Lc0 1 node vs 10,000 nodes (model curves)", g_mod, "", "", "", "", ""])
    rows.append(["average gap, Stockfish depth 25 vs Lc0 10,000 nodes (observed curves)", g_sf, "", "", "", "", ""])
    print(f"   average gap between share curves: Lc0 1 vs 10,000 nodes {g_obs:.4f} (observed), {g_mod:.4f} (model); "
          f"Stockfish d25 vs Lc0 10,000 nodes {g_sf:.4f}")
    c.write_csv(c.OUT / "robustness_lc0.csv", [[r[0]] + [round(v, 6) if isinstance(v, float) else v for v in r[1:]] for r in rows],
                ["quantity", "stockfish_depth25", "lc0_10000_nodes", "lc0_1_node", "stockfish_minus_lc0", "lo95", "hi95"])


def main():
    S = mdl.limits()
    scale_check(S)
    near_ties_check(S)
    first_depth_check(S)
    lc0_check(S)


if __name__ == "__main__":
    main()
