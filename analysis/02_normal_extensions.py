"""2. Does the normal part of the logit-normal need skewness or heavier tails?

The zero-one-inflated logit-normal has four free parameters: P(e = 0) = p0, P(e = 1) = p1 and, for 0 < e < 1,
logit(e) = log(distance to the best / distance to the worst) ~ Normal(m, s). Two five-parameter extensions of
the normal part:
    skew    logit(e) ~ skew-normal(location, scale, shape alpha); alpha = 0 is the normal;
    tails   logit(e) ~ Student t(location, scale, degrees of freedom nu); nu -> infinity is the normal.
Each depth is fitted on its own by weighted maximum likelihood (each position weighted equally), errors as measured
and with near-ties (within 0.001 of 0 or 1) counted as ties. An extension counts as helping at a depth when it
lowers the AIC by more than 2.
Output: out/normal_extensions.csv
"""
import json
import math

import numpy as np
from scipy import optimize, stats
from scipy.special import log_ndtr, logit

import common as c
import model as mdl


def fit_normal(z, wz):
    m = float(np.sum(wz * z) / wz.sum())
    s = float(np.sqrt(np.sum(wz * (z - m) ** 2) / wz.sum()))
    return {"m": m, "s": s}, float(np.sum(wz * stats.norm.logpdf(z, m, s)))


def fit_skew(z, wz, start):
    def nll(th):
        loc, sc, al = th[0], math.exp(th[1]), th[2]
        u = (z - loc) / sc
        return -np.sum(wz * (math.log(2) - th[1] - 0.5 * u * u - 0.5 * math.log(2 * math.pi) + log_ndtr(al * u)))
    best = min((optimize.minimize(nll, np.array(s0), method="L-BFGS-B", bounds=[(-20, 20), (-5, 5), (-20, 20)])
                for s0 in ([start["m"], math.log(start["s"]), 0.0], [start["m"] - 1, math.log(start["s"]), 1.0],
                           [start["m"] + 1, math.log(start["s"]), -1.0])), key=lambda r: r.fun)
    return {"loc": best.x[0], "scale": math.exp(best.x[1]), "alpha": best.x[2]}, -float(best.fun)


def fit_t(z, wz, start):
    def nll(th):
        return -np.sum(wz * stats.t.logpdf(z, math.exp(th[2]), th[0], math.exp(th[1])))
    best = min((optimize.minimize(nll, np.array(s0), method="L-BFGS-B", bounds=[(-20, 20), (-5, 5), (math.log(0.3), math.log(500))])
                for s0 in ([start["m"], math.log(start["s"]), math.log(5)], [start["m"], math.log(start["s"] * 0.7), math.log(2)],
                           [start["m"], math.log(start["s"]), math.log(50)])), key=lambda r: r.fun)
    return {"loc": best.x[0], "scale": math.exp(best.x[1]), "nu": math.exp(best.x[2])}, -float(best.fun)


def main():
    run = c.load_stockfish()
    xs = run.x()
    rows = []
    for version, near_ties in (("measured", False), ("near-ties", True)):
        per_depth = mdl.errors_by_depth(run, xs, near_ties)
        for li, depth in enumerate(run.levels):
            e = np.concatenate(per_depth[li])
            w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in per_depth[li]])
            inner = (e > 0) & (e < 1)
            z, wz = logit(e[inner]), w[inner]
            base, ll0 = fit_normal(z, wz)
            for name, (params, ll, k) in (("normal", (base, ll0, 4)), ("skew", (*fit_skew(z, wz, base), 5)),
                                          ("tails", (*fit_t(z, wz, base), 5))):
                # the point masses and the change of variable from logit(e) to e are the same for all three,
                # so the difference in AIC comes from the normal part alone
                rows.append([version, depth, name, k, round(ll, 4), round(2 * k - 2 * ll, 4),
                             json.dumps({kk: round(float(vv), 5) for kk, vv in params.items()})])
    c.write_csv(c.OUT / "normal_extensions.csv", rows, ["version", "depth", "model", "n_params", "loglik_middle",
                                                         "aic_middle", "params"])
    print("Extensions of the normal part: depths (of 25) where the extra parameter lowers AIC by more than 2")
    for version in ("measured", "near-ties"):
        sel = [r for r in rows if r[0] == version]
        base = {r[1]: r[5] for r in sel if r[2] == "normal"}
        for ext in ("skew", "tails"):
            d_aic = [r[5] - base[r[1]] for r in sel if r[2] == ext]
            print(f"  {version:<10} {ext:<6} helps at {sum(x < -2 for x in d_aic):>2} depths; median change in AIC {np.median(d_aic):+.1f}")


if __name__ == "__main__":
    main()
