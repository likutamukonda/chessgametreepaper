"""3. Which function of depth? (Stockfish, near-ties counted as ties, depths 5-25)

Shapes fitted to p0, the middle share, m and s as functions of depth d:
    exponential   y = L - c exp(-k d)                          saturating exponential; level L
    power         y = L - c d^(-g)                             slower power-law approach; level L
    logistic      y = A + (L - A) / (1 + exp(-k (d - d0)))     S-curve; level L
    gamma kernel  s = c + H (d / d_p)^a exp(a (1 - d / d_p))   rise and fall (s only); level c
    linear        y = a + b d                                  benchmark without a level
Weighted least squares on depths 5-25 (weights 1 / se^2). Held-out test: fit depths 5-20, predict 21-25.
Errors are in units of each depth's bootstrap standard error (about 1 = within the noise). Levels: 95% bootstrap
interval over the resamples in which the curve levels off.
Output: out/depth_shapes.csv
"""
import numpy as np

import common as c
import model as mdl

SHAPES = {"p0": ("exponential", "power", "logistic", "linear"),
          "middle": ("exponential", "power", "logistic", "linear"),
          "m": ("exponential", "power", "logistic", "linear"),
          "s": ("exponential", "power", "logistic", "gamma kernel", "linear")}


def main():
    S = mdl.setup()
    d, full, se, reps = S["d"], S["full"], S["se"], S["reps"]
    k, tr, ho = d >= mdl.FIRST, (d >= mdl.FIRST) & (d <= 20), d >= 21
    rows = []
    print(f"Functions of depth, depths {mdl.FIRST}-25 (rms z: error in standard errors; held out: fit to 20, predict 21-25)")
    for q, shapes in SHAPES.items():
        print(f"  {q}")
        y, s_ = full[q], se[q]
        for shape in shapes:
            p_all, level, off = mdl.fit_shape(shape, d[k], y[k], s_[k], q)
            p_tr, _, _ = mdl.fit_shape(shape, d[tr], y[tr], s_[tr], q)
            f = mdl.SHAPE_F[shape]
            z_in = (f(d[k], *p_all) - y[k]) / s_[k]
            z_out = (f(d[ho], *p_tr) - y[ho]) / s_[ho]
            lv = []
            if shape != "linear":
                for r in reps:
                    pb, lb, ob = mdl.fit_shape(shape, d[k], r[q][k], s_[k], q, start=p_all)
                    if pb is not None and ob:
                        lv.append(lb)
            lo, hi = np.percentile(lv, [2.5, 97.5]) if len(lv) >= 20 else (np.nan, np.nan)
            rows.append([q, shape, len(p_all), round(float(np.sqrt(np.mean(z_in ** 2))), 3),
                         round(float(np.sqrt(np.mean(z_out ** 2))), 3), "" if level is None else round(level, 4), off,
                         "" if np.isnan(lo) else round(float(lo), 4), "" if np.isnan(hi) else round(float(hi), 4),
                         round(len(lv) / mdl.B, 3)])
            lvl = "-" if level is None else f"{level:.3f}" + ("" if off else " (no level)")
            ci = "" if np.isnan(lo) else f"  95% [{lo:.3f}, {hi:.3f}], levels off in {len(lv) / mdl.B:.0%} of resamples"
            print(f"    {shape:<13} in-sample {np.sqrt(np.mean(z_in ** 2)):5.2f}  held out {np.sqrt(np.mean(z_out ** 2)):5.2f}  level {lvl}{ci}")
    c.write_csv(c.OUT / "depth_shapes.csv", rows, ["parameter", "shape", "n_params", "rms_z_in_sample", "rms_z_heldout",
                                                    "level", "levels_off", "level_lo95", "level_hi95", "share_levelling_off"])


if __name__ == "__main__":
    main()
