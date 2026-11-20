"""4. Which two of the three shares should carry the depth functions? (Stockfish, near-ties as ties, depths 5-25)

p0 (moves exactly at 0), p1 (exactly at 1) and the middle share (1 - p0 - p1) always add to 1, so two are free.
Three ways to choose them, each with logistic S-curves in depth:
    A  p0 and p1 fitted, middle = 1 - p0 - p1
    B  p0 and the middle fitted, p1 = 1 - p0 - middle          (used in the paper)
    C  ends = p0 + p1 and split = p0 / (p0 + p1) fitted, p0 = ends x split, p1 = ends x (1 - split)
For each quantity: in-sample and held-out error (fit depths 5-20, predict 21-25; in standard errors), level and its
95% bootstrap interval. For p1 under A, B and C: how well the implied curve follows the measured p1, its implied
level and interval, and whether it stays between 0 and 1 out to depth 100.
Output: out/spike_parameterisation.csv
"""
import numpy as np

import common as c
import model as mdl

LOG = mdl.SHAPE_F["logistic"]
QUANTS = ("p0", "p1", "middle", "ends", "split")
for q in ("ends", "split"):
    mdl.RANGE[q] = (0.0, 1.0)


def spike_series(params):
    p0, p1 = params["p0"], params["p1"]
    return {"p0": p0, "p1": p1, "middle": 1 - p0 - p1, "ends": p0 + p1, "split": p0 / (p0 + p1)}


def p1_from(scheme, fits, d):
    if scheme == "A":
        return LOG(d, *fits["p1"])
    if scheme == "B":
        return 1 - LOG(d, *fits["p0"]) - LOG(d, *fits["middle"])
    return LOG(d, *fits["ends"]) * (1 - LOG(d, *fits["split"]))


def main():
    S = mdl.setup()
    d = S["d"]
    k, tr, ho = d >= mdl.FIRST, (d >= mdl.FIRST) & (d <= 20), d >= 21
    far = np.arange(1, 101, dtype=float)
    full = spike_series(S["full"])
    reps = [spike_series(r) for r in S["reps"]]
    se = {q: np.array([r[q] for r in reps]).std(axis=0, ddof=1) for q in QUANTS}
    rows, fits, fits_tr, boot_fits = [], {}, {}, {q: [] for q in QUANTS}
    print(f"S-curves for the share quantities, depths {mdl.FIRST}-25 (errors in standard errors)")
    for q in QUANTS:
        y = full[q]
        p_all, L, off = mdl.fit_shape("logistic", d[k], y[k], se[q][k], q)
        p_tr, _, _ = mdl.fit_shape("logistic", d[tr], y[tr], se[q][tr], q)
        fits[q], fits_tr[q] = p_all, p_tr
        z_in = (LOG(d[k], *p_all) - y[k]) / se[q][k]
        z_out = (LOG(d[ho], *p_tr) - y[ho]) / se[q][ho]
        lv = []
        for r in reps:
            pb, lb, ob = mdl.fit_shape("logistic", d[k], r[q][k], se[q][k], q, start=p_all)
            boot_fits[q].append(pb if (pb is not None and ob) else None)
            if pb is not None and ob:
                lv.append(lb)
        lo, hi = np.percentile(lv, [2.5, 97.5]) if len(lv) >= 20 else (np.nan, np.nan)
        rows.append(["quantity", q, round(float(np.sqrt(np.mean(z_in ** 2))), 3), round(float(np.sqrt(np.mean(z_out ** 2))), 3),
                     round(L, 4), round(float(lo), 4), round(float(hi), 4), round(len(lv) / mdl.B, 3), ""])
        print(f"  {q:<7} in-sample {np.sqrt(np.mean(z_in ** 2)):.2f}  held out {np.sqrt(np.mean(z_out ** 2)):.2f}  level {L:.3f}  "
              f"95% [{lo:.3f}, {hi:.3f}]  levels off in {len(lv) / mdl.B:.0%} of resamples")
    for scheme in ("A", "B", "C"):
        curve = p1_from(scheme, fits, d[k])
        z_in = (curve - full["p1"][k]) / se["p1"][k]
        z_out = (p1_from(scheme, fits_tr, d[ho]) - full["p1"][ho]) / se["p1"][ho]
        far_curve = p1_from(scheme, fits, far)
        need = {"A": ("p1",), "B": ("p0", "middle"), "C": ("ends", "split")}[scheme]
        lv = [p1_from(scheme, {q: boot_fits[q][b] for q in need}, np.array([1e4]))[0]
              for b in range(mdl.B) if all(boot_fits[q][b] is not None for q in need)]
        lo, hi = np.percentile(lv, [2.5, 97.5]) if len(lv) >= 20 else (np.nan, np.nan)
        valid = bool(np.all((far_curve >= 0) & (far_curve <= 1)))
        rows.append(["p1 via scheme", scheme, round(float(np.sqrt(np.mean(z_in ** 2))), 3), round(float(np.sqrt(np.mean(z_out ** 2))), 3),
                     round(float(p1_from(scheme, fits, np.array([1e4]))[0]), 4), round(float(lo), 4), round(float(hi), 4),
                     round(len(lv) / mdl.B, 3), valid])
        name = {"A": "p1 fitted directly", "B": "1 - p0 - middle", "C": "ends x (1 - split)"}[scheme]
        print(f"  p1 via {scheme} ({name:<18}) in-sample {np.sqrt(np.mean(z_in ** 2)):.2f}  held out {np.sqrt(np.mean(z_out ** 2)):.2f}  "
              f"implied level {p1_from(scheme, fits, np.array([1e4]))[0]:.3f}  95% [{lo:.3f}, {hi:.3f}]  stays in 0-1 to depth 100: {valid}")
    c.write_csv(c.OUT / "spike_parameterisation.csv", rows,
                ["table", "quantity_or_scheme", "rms_z_in_sample", "rms_z_heldout", "level", "lo95", "hi95",
                 "share_of_resamples", "valid_to_depth_100"])


if __name__ == "__main__":
    main()
