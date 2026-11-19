"""Step 10: which distribution fits the relative error best, level by level?

At every Stockfish depth and every Lc0 node level separately, the relative errors of all moves are pooled
over positions, each move weighted 1 / (legal moves in its position) so that every position counts
equally. The weighted distribution is then exactly the average share-by-tolerance curve
(out/share_by_tolerance.png). Each candidate is fitted by maximum likelihood with its own parameters at
each level; depth (or node count) is a hyperparameter here, not part of the model.

Candidates, all on [0, 1] with probability at exactly 0 and exactly 1:
  ZOI beta            P(e=0) = p0, P(e=1) = p1, otherwise Beta(a, b)                        4 parameters
  ZOI Kumaraswamy     the same with a Kumaraswamy(a, b) in the middle                          4
  ZOI logit-normal    the same with a logit-normal(m, s) in the middle (= Johnson SB on 0-1)    4
  censored normal     a normal(m, s) cut off at 0 and 1; the tails beyond pile up as the spikes 2
  censored logistic   the same with a logistic(m, s)                                           2
  ordered beta        Kubinec's model: cut points k1 < k2 on one scale eta give the spikes;
                      the middle is Beta with mean logistic(eta) and precision phi              4
  two-beta mixture    ZOI with a mixture of two betas in the middle; a flexibility benchmark,
                      not a candidate                                                          7

Comparison: weighted log-likelihood (each position contributes total weight 1, so AIC and BIC use the
number of positions), and the gap between the fitted and the observed curve: largest gap, root-mean-
square gap over t in [0, 0.999], and the gap at t = 0.05, 0.10, 0.25 and 0.5. Moves in one position are
not independent, so the likelihood ranks the models but its absolute size is not a proper likelihood.
Sensitivity: the same fits with every error within SNAP = 0.001 of 0 or 1 counted as exactly 0 or 1.
Lc0 rates about a quarter of all moves within 0.0001 of its worst move, a near-wall just below 1 that
no smooth middle part can follow; this check shows how the ranking changes when such moves count as
equal to the worst (out/error_distribution_fits_snapped.csv).
Output: out/error_distribution_fits.csv
"""
import json
import math

import numpy as np
from scipy import optimize
from scipy.special import betainc, betaln, expit, log_ndtr, logit, ndtr

import common as c

GRID = np.concatenate([[0.0], np.linspace(1e-4, 0.999, 999)])
CHECKS = (0.05, 0.10, 0.25, 0.5)
SNAP = 1e-3
MODELS = ["ZOI beta", "ZOI Kumaraswamy", "ZOI logit-normal", "censored normal", "censored logistic",
          "ordered beta", "two-beta mixture (benchmark)"]
N_PARAMS = {"ZOI beta": 4, "ZOI Kumaraswamy": 4, "ZOI logit-normal": 4, "censored normal": 2,
            "censored logistic": 2, "ordered beta": 4, "two-beta mixture (benchmark)": 7}


def level_data(run, li):
    xs = run.x()
    e, w = [], []
    for p in run.pos_ids:
        v = c.relative_errors(xs[p][li:li + 1])[0]
        e.append(v)
        w.append(np.full(len(v), 1.0 / len(v)))
    return np.concatenate(e), np.concatenate(w)


def minimise(f, starts, bounds):
    best = None
    for s in starts:
        r = optimize.minimize(f, np.array(s, float), method="L-BFGS-B", bounds=bounds)
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


# ---- the middle parts (weighted maximum likelihood on 0 < e < 1) ------------------------------------
def middle_beta(x, w):
    lx, l1 = np.log(x), np.log1p(-x)
    r = minimise(lambda th: -np.sum(w * ((np.exp(th[0]) - 1) * lx + (np.exp(th[1]) - 1) * l1 - betaln(*np.exp(th)))),
                 ([0, 0], [-1, -1], [1, 1]), [(-7, 7), (-7, 7)])
    a, b = np.exp(r.x)
    return {"a": a, "b": b}, -r.fun, lambda t: betainc(a, b, t)


def middle_kumaraswamy(x, w):
    lx = np.log(x)

    def nll(th):
        a, b = np.exp(th)
        # log(1 - x^a) computed as log(-expm1(a log x)), which stays finite for x just below 1
        return -np.sum(w * (np.log(a) + np.log(b) + (a - 1) * lx + (b - 1) * np.log(-np.expm1(a * lx))))
    r = minimise(nll, ([0, 0], [-1, -1], [1, 1]), [(-7, 7), (-7, 7)])
    a, b = np.exp(r.x)
    return {"a": a, "b": b}, -r.fun, lambda t: 1.0 - np.power(1.0 - np.power(t, a), b)


def middle_logitnormal(x, w):
    z, lj = logit(x), np.log(x) + np.log1p(-x)
    m = float(np.sum(w * z) / np.sum(w))
    s = float(np.sqrt(np.sum(w * (z - m) ** 2) / np.sum(w)))          # closed-form weighted MLE
    ll = float(np.sum(w * (-np.log(s) - 0.5 * math.log(2 * math.pi) - lj - (z - m) ** 2 / (2 * s * s))))
    return {"m": m, "s": s}, ll, lambda t: ndtr((logit(np.clip(t, 1e-300, 1 - 1e-16)) - m) / s)


def middle_beta2(x, w):
    lx, l1 = np.log(x), np.log1p(-x)

    def nll(th):
        q = expit(th[0])
        a1, b1, a2, b2 = np.exp(th[1:])
        l_1 = (a1 - 1) * lx + (b1 - 1) * l1 - betaln(a1, b1)
        l_2 = (a2 - 1) * lx + (b2 - 1) * l1 - betaln(a2, b2)
        return -np.sum(w * np.logaddexp(np.log(q) + l_1, np.log1p(-q) + l_2))
    starts = ([0, 0, 1.5, 1.5, 0], [0, -0.5, 1, 1, -0.5], [-1, 0, 2, 0.5, -1], [1, -1, 0.5, 2, 1])
    r = minimise(nll, starts, [(-8, 8)] + [(-7, 7)] * 4)
    q = expit(r.x[0])
    a1, b1, a2, b2 = np.exp(r.x[1:])
    return ({"weight1": q, "a1": a1, "b1": b1, "a2": a2, "b2": b2}, -r.fun,
            lambda t: q * betainc(a1, b1, t) + (1 - q) * betainc(a2, b2, t))


def fit_inflated(e, w, middle):
    W = w.sum()
    w0, w1 = w[e == 0].sum(), w[e == 1].sum()
    p0, p1 = w0 / W, w1 / W
    inner = (e > 0) & (e < 1)
    params, ll_mid, cdf_mid = middle(e[inner], w[inner])
    ll = w0 * np.log(p0) + w1 * np.log(p1) + w[inner].sum() * np.log(1 - p0 - p1) + ll_mid
    params = {"p0": p0, "p1": p1, **params}
    return params, float(ll), (lambda t: np.where(t >= 1, 1.0, p0 + (1 - p0 - p1) * cdf_mid(np.minimum(t, 1 - 1e-16))))


# ---- censored models ----------------------------------------------------------------------------------
def fit_censored(e, w, kind):
    zero, one, inner = e == 0, e == 1, (e > 0) & (e < 1)
    x, wi, w0, w1 = e[inner], w[inner], w[zero].sum(), w[one].sum()

    if kind == "normal":
        lcdf = log_ndtr
        lpdf = lambda z: -0.5 * z * z - 0.5 * math.log(2 * math.pi)
        cdf = ndtr
    else:
        lcdf = lambda z: -np.logaddexp(0.0, -z)
        lpdf = lambda z: -z - 2 * np.logaddexp(0.0, -z)
        cdf = expit

    def nll(th):
        m, s = th[0], np.exp(th[1])
        return -(w0 * lcdf((0 - m) / s) + w1 * lcdf(-(1 - m) / s) + np.sum(wi * (lpdf((x - m) / s) - np.log(s))))
    r = minimise(nll, ([0.5, 0], [0.3, -0.5], [0.6, 0.5], [0.5, 1.0]), [(-20, 20), (-6, 4)])
    m, s = r.x[0], float(np.exp(r.x[1]))
    return {"m": m, "s": s}, -float(r.fun), (lambda t: np.where(t >= 1, 1.0, cdf((t - m) / s)))


# ---- ordered beta (Kubinec) ---------------------------------------------------------------------------
def fit_ordered_beta(e, w):
    zero, one, inner = e == 0, e == 1, (e > 0) & (e < 1)
    x, wi, w0, w1 = e[inner], w[inner], w[zero].sum(), w[one].sum()
    lx, l1 = np.log(x), np.log1p(-x)

    def parts(th):
        eta, k1, k2, phi = th[0], th[1], th[1] + np.exp(th[2]), np.exp(th[3])
        p0, p1 = 1 - expit(eta - k1), expit(eta - k2)
        return eta, k1, k2, phi, p0, p1

    def nll(th):
        eta, k1, k2, phi, p0, p1 = parts(th)
        mu = expit(eta)
        a, b = mu * phi, (1 - mu) * phi
        mid = (a - 1) * lx + (b - 1) * l1 - betaln(a, b)
        return -(w0 * np.log(p0) + w1 * np.log(p1) + wi.sum() * np.log(1 - p0 - p1) + np.sum(wi * mid))
    r = minimise(nll, ([0, -2, 1, 0], [0.5, -3, 1.5, -0.5], [-0.5, -2, 1, 0.5]), [(-10, 10), (-15, 15), (-8, 4), (-7, 7)])
    eta, k1, k2, phi, p0, p1 = parts(r.x)
    mu = expit(eta)
    a, b = mu * phi, (1 - mu) * phi
    return ({"eta": eta, "k1": k1, "k2": k2, "phi": phi, "p0": p0, "p1": p1}, -float(r.fun),
            (lambda t: np.where(t >= 1, 1.0, p0 + (1 - p0 - p1) * betainc(a, b, np.minimum(t, 1 - 1e-16)))))


def fit_model(name, e, w):
    if name == "ZOI beta":
        return fit_inflated(e, w, middle_beta)
    if name == "ZOI Kumaraswamy":
        return fit_inflated(e, w, middle_kumaraswamy)
    if name == "ZOI logit-normal":
        return fit_inflated(e, w, middle_logitnormal)
    if name == "censored normal":
        return fit_censored(e, w, "normal")
    if name == "censored logistic":
        return fit_censored(e, w, "logistic")
    if name == "ordered beta":
        return fit_ordered_beta(e, w)
    return fit_inflated(e, w, middle_beta2)


def observed_cdf(e, w, t):
    order = np.argsort(e)
    es, cw = e[order], np.cumsum(w[order]) / w.sum()
    idx = np.searchsorted(es, t + 1e-12, side="right") - 1
    return np.where(idx >= 0, cw[np.maximum(idx, 0)], 0.0)


def main():
    runs = (c.load_stockfish(), c.load_lc0())
    for snap, out_name in ((0.0, "error_distribution_fits.csv"), (SNAP, "error_distribution_fits_snapped.csv")):
        print("\n=== " + ("errors as measured" if snap == 0 else f"errors within {snap:g} of 0 or 1 counted as 0 or 1") + " ===")
        fit_all(runs, snap, out_name)


def fit_all(runs, snap, out_name):
    rows = []
    for run in runs:
        for li, level in enumerate(run.levels):
            e, w = level_data(run, li)
            if snap:
                e = np.where(e < snap, 0.0, np.where(e > 1 - snap, 1.0, e))
            obs = observed_cdf(e, w, GRID)
            n_pos = len(run.pos_ids)
            for name in MODELS:
                params, ll, cdf = fit_model(name, e, w)
                gap = cdf(GRID) - obs
                at = [float(cdf(np.array([t]))[0] - observed_cdf(e, w, np.array([t]))[0]) for t in CHECKS]
                k = N_PARAMS[name]
                rows.append([run.engine, li, level, name, k, round(ll, 4), round(2 * k - 2 * ll, 4),
                             round(k * math.log(n_pos) - 2 * ll, 4), round(float(np.abs(gap).max()), 5),
                             round(float(GRID[np.abs(gap).argmax()]), 3), round(float(np.sqrt(np.mean(gap ** 2))), 5)]
                            + [round(v, 5) for v in at] + [json.dumps({kk: round(float(vv), 6) for kk, vv in params.items()})])
    header = ["engine", "level_index", "level", "model", "n_params", "loglik", "aic", "bic", "largest_gap",
              "t_of_largest_gap", "rms_gap"] + [f"gap_t{t}" for t in CHECKS] + ["params"]
    c.write_csv(c.OUT / out_name, rows, header)
    summarise(rows)


def summarise(rows):
    import collections
    print("Fits at every level (25 per engine), pooled over positions with each position weighted equally")
    for eng in ("Stockfish", "Lc0"):
        sel = [r for r in rows if r[0] == eng]
        by_level = collections.defaultdict(list)
        for r in sel:
            by_level[r[1]].append(r)
        cand = [m for m in MODELS if "benchmark" not in m]
        aic_w = collections.Counter(min((r for r in fits if r[3] in cand), key=lambda r: r[6])[3] for fits in by_level.values())
        bic_w = collections.Counter(min((r for r in fits if r[3] in cand), key=lambda r: r[7])[3] for fits in by_level.values())
        print(f"\n{eng}: lowest AIC among the candidates {dict(aic_w)}; lowest BIC {dict(bic_w)}")
        print(f"  {'model':<30}{'median dAIC':>12}{'median largest gap':>20}{'median rms gap':>16}   gap at t=0.05/0.10/0.25/0.5 (median)")
        for m in MODELS:
            rs = [r for r in sel if r[3] == m]
            daic = [r[6] - min(x[6] for x in by_level[r[1]] if x[3] in cand) for r in rs]
            print(f"  {m:<30}{np.median(daic):>12.1f}{np.median([r[8] for r in rs]):>20.3f}{np.median([r[10] for r in rs]):>16.3f}"
                  "   " + " / ".join(f"{np.median([r[11 + i] for r in rs]):+.3f}" for i in range(4)))


if __name__ == "__main__":
    main()
