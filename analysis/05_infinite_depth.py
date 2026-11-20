"""5. Depth functions and the distribution at infinite depth (Stockfish, near-ties counted as ties).

Shallow depths. At the first few depths the search is too shallow to see past simple threats (the horizon effect).
Checked from the data: the node count per extra ply, how far a move's value moves between consecutive depths, and how
far each parameter at each depth sits from the depth functions fitted on depths 5-25 (in standard errors). The fits
start at depth 5.

Depth functions (model.py): logistic S-curves for p0, the middle share and m; a gamma kernel for s, with the S-curve
for s as the alternative; p1 = 1 - p0 - middle. For each: weighted residual sum of squares on depths 5-25, error on
held-out depths 21-25 when fitted on 5-20, the level at infinite depth and its 95% bootstrap interval.

Infinite depth: the zero-one-inflated logit-normal with every parameter at its level, for both depth functions of s,
with 95% intervals from the resamples in which every curve has a level: the parameters and the share of moves within
t = 0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.99.
Outputs: out/shallow_depths.csv, out/depth_functions.csv, out/infinite_depth.csv
"""
import collections
import csv
import io
import zipfile

import numpy as np

import common as c
import model as mdl

TOL = (0.0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.99)


def load_nodes():
    out = collections.defaultdict(dict)
    with zipfile.ZipFile(c.SF_RUN) as zf:
        for r in csv.DictReader(io.TextIOWrapper(zf.open("stockfish_evals.csv"), encoding="utf-8", newline="")):
            out[(r["pos_id"], r["move"])][int(r["depth"])] = int(r["nodes"])
    return out


def shallow_depths(S):
    d, full, se, fits, xs, run = S["d"], S["full"], S["se"], S["fits"], S["xs"], S["run"]
    nodes = load_nodes()
    rows = []
    for li, dep in enumerate(d.astype(int)):
        ratio = (np.median([v[dep] / v[dep - 1] for v in nodes.values() if dep in v and dep - 1 in v and v[dep - 1] > 0])
                 if dep > 1 else np.nan)
        if li > 0:
            ch = np.concatenate([xs[p][li] - xs[p][li - 1] for p in run.pos_ids])
            swing = float(np.mean(np.abs(ch[np.isfinite(ch)])))
        else:
            swing = np.nan
        z = {q: float((full[q][li] - mdl.curve_values(q, fits, np.array([dep], float))[0]) / se[q][li]) for q in mdl.QUANTS}
        rows.append([int(dep), ratio, swing] + [z[q] for q in mdl.QUANTS])
    c.write_csv(c.OUT / "shallow_depths.csv",
                [[r[0]] + [("" if np.isnan(v) else round(float(v), 4)) for v in r[1:]] for r in rows],
                ["depth", "median_node_ratio_per_ply", "mean_abs_change_in_value"] + [f"z_{q}" for q in mdl.QUANTS])
    return rows


def main():
    S = mdl.limits()
    d, full, se, fits, boot_fits = S["d"], S["full"], S["se"], S["fits"], S["boot_fits"]
    rows_shallow = shallow_depths(S)

    k, tr, ho = d >= mdl.FIRST, (d >= mdl.FIRST) & (d <= 20), d >= 21
    fit_rows = []
    for q, shape in mdl.SHAPES.items():
        key = mdl.data_key(q)
        y, s_ = full[key], se[key]
        p_tr = mdl.fit_shape(shape, d[tr], y[tr], s_[tr], key)[0]
        f = mdl.SHAPE_F[shape]
        rss = float(np.sum(((f(d[k], *fits[q][0]) - y[k]) / s_[k]) ** 2))
        z_out = (f(d[ho], *p_tr) - y[ho]) / s_[ho]
        lv = [bf[q][1] for bf in boot_fits if bf[q][0] is not None and bf[q][2]]
        lo, hi = np.percentile(lv, [2.5, 97.5]) if len(lv) >= 20 else (np.nan, np.nan)
        fit_rows.append([key, shape, rss, float(np.sqrt(np.mean(z_out ** 2))), fits[q][1], bool(fits[q][2]),
                         float(lo), float(hi), len(lv) / mdl.B, " ".join(f"{v:.5g}" for v in fits[q][0])])
    c.write_csv(c.OUT / "depth_functions.csv", [[round(v, 4) if isinstance(v, float) else v for v in r] for r in fit_rows],
                ["parameter", "shape", "weighted_rss", "rms_z_heldout_21_25", "level", "has_level", "level_lo95",
                 "level_hi95", "share_of_resamples_with_level", "params"])

    out_rows = []
    for s_key in ("s", "s_logistic"):
        inf = mdl.levels(fits, s_key)
        bl = mdl.boot_levels(S, s_key)
        for q in mdl.QUANTS:
            lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
            out_rows.append([mdl.SHAPES[s_key], "parameter", q, float(full[q][-1]), inf[q], float(lo), float(hi), len(bl) / mdl.B])
        for t in TOL:
            vals = [mdl.model_curve(L, np.array([t]))[0] for L in bl]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out_rows.append([mdl.SHAPES[s_key], "share within t", t, float(mdl.model_curve(mdl.at(full, len(d) - 1), np.array([t]))[0]),
                             float(mdl.model_curve(inf, np.array([t]))[0]), float(lo), float(hi), len(bl) / mdl.B])
    c.write_csv(c.OUT / "infinite_depth.csv", [[round(v, 5) if isinstance(v, float) else v for v in r] for r in out_rows],
                ["s_shape", "quantity", "parameter_or_t", "model_at_depth_25", "infinite_depth", "lo95", "hi95",
                 "share_of_resamples_used"])
    report(rows_shallow, fit_rows, out_rows)


def report(rows_shallow, fit_rows, out_rows):
    print("Shallow depths: node count per extra ply, mean change in a move's value since the previous depth, and each")
    print(f"parameter's distance from the depth functions fitted on depths {mdl.FIRST}-25 (data minus curve, standard errors)")
    print(f"  {'depth':>5} {'nodes x':>8} {'|change|':>9} " + " ".join(f"{q:>7}" for q in mdl.QUANTS))
    for r in rows_shallow[:8]:
        print(f"  {r[0]:>5} {('' if np.isnan(r[1]) else f'{r[1]:.2f}'):>8} {('' if np.isnan(r[2]) else f'{r[2]:.4f}'):>9} "
              + " ".join(f"{v:>+7.1f}" for v in r[3:]))
    ratios = [r[1] for r in rows_shallow[4:]]
    print(f"  from depth 5 on: node ratio between {min(ratios):.2f} and {max(ratios):.2f}")
    print(f"\nDepth functions (depths {mdl.FIRST}-25)")
    for r in fit_rows:
        print(f"  {r[0]:<7} {r[1]:<13} weighted rss {r[2]:6.2f}  held out {r[3]:5.2f}  level {r[4]:7.3f}  "
              f"95% [{r[6]:.3f}, {r[7]:.3f}]  resamples with a level {r[8]:.0%}")
        if r[1] == "gamma kernel":
            p = [float(v) for v in r[9].split()]
            print(f"           peak {p[0] + p[1]:.3f} at depth {p[2]:.1f}")
    for shape in ("gamma kernel", "logistic"):
        rr = [r for r in out_rows if r[0] == shape]
        print(f"\nInfinite depth with the {shape} for s ({rr[0][7]:.0%} of resamples)")
        for r in rr:
            label = r[2] if r[1] == "parameter" else f"within {r[2]}"
            print(f"  {label:<12} depth 25 (model) {r[3]:7.3f}   infinite depth {r[4]:7.3f}   95% [{r[5]:.3f}, {r[6]:.3f}]")


if __name__ == "__main__":
    main()
