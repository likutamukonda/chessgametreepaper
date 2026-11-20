"""1. Which distribution describes the relative error best? (Table 1 of the paper)

At every Stockfish depth separately, the relative errors of all legal moves are pooled over positions, each move
weighted 1 / (legal moves in its position) so that every position counts equally. Each candidate is fitted by
weighted maximum likelihood with its own parameters at each depth.

Candidates, all on [0, 1] with probability at exactly 0 and exactly 1:
  ZOI beta            P(e=0) = p0, P(e=1) = p1, otherwise Beta(a, b)                          4 parameters
  ZOI Kumaraswamy     the same with a Kumaraswamy(a, b) in the middle                            4
  ZOI logit-normal    the same with a logit-normal(m, s) in the middle                           4
  censored normal     a normal(m, s) cut off at 0 and 1; the tails beyond pile up at 0 and 1     2
  censored logistic   the same with a logistic(m, s)                                             2
  ordered beta        Kubinec's model: cut points k1 < k2 on one scale eta give the point masses;
                      the middle is Beta with mean logistic(eta) and precision phi                4
  two-beta mixture    ZOI with a mixture of two betas in the middle; a flexibility benchmark      7

Comparison: AIC (moves within a position are not independent, so the likelihood ranks the models but its absolute
size is not a proper likelihood) and the largest vertical gap between the fitted and observed share of moves within
a tolerance, over t in [0, 0.999]. Both versions of the data: errors as measured, and near-ties (within 0.001 of 0
or 1) counted as ties.
Output: out/candidate_distributions.csv
"""
import collections
import json
import math

import numpy as np
from scipy import optimize
from scipy.special import betainc, betaln, expit, log_ndtr, logit, ndtr

import common as c
import model as mdl

GRID = np.concatenate([[0.0], np.linspace(1e-4, 0.999, 999)])
MODELS = ["ZOI logit-normal", "ZOI beta", "ordered beta", "ZOI Kumaraswamy", "censored normal", "censored logistic",
          "two-beta mixture (benchmark)"]
N_PARAMS = {"ZOI beta": 4, "ZOI Kumaraswamy": 4, "ZOI logit-normal": 4, "censored normal": 2,
            "censored logistic": 2, "ordered beta": 4, "two-beta mixture (benchmark)": 7}


def minimise(f, starts, bounds):
    best = None
    for s in starts:
        r = optimize.minimize(f, np.array(s, float), method="L-BFGS-B", bounds=bounds)
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


# ---- the middle parts (weighted maximum likelihood on 0 < e < 1)
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


# ---- censored models
def fit_censored(e, w, kind):
    zero, one, inner = e == 0, e == 1, (e > 0) & (e < 1)
    x, wi, w0, w1 = e[inner], w[inner], w[zero].sum(), w[one].sum()
    if kind == "normal":
        lcdf, cdf = log_ndtr, ndtr
        lpdf = lambda z: -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    else:
        lcdf, cdf = (lambda z: -np.logaddexp(0.0, -z)), expit
        lpdf = lambda z: -z - 2 * np.logaddexp(0.0, -z)

    def nll(th):
        m, s = th[0], np.exp(th[1])
        return -(w0 * lcdf((0 - m) / s) + w1 * lcdf(-(1 - m) / s) + np.sum(wi * (lpdf((x - m) / s) - np.log(s))))
    r = minimise(nll, ([0.5, 0], [0.3, -0.5], [0.6, 0.5], [0.5, 1.0]), [(-20, 20), (-6, 4)])
    m, s = r.x[0], float(np.exp(r.x[1]))
    return {"m": m, "s": s}, -float(r.fun), (lambda t: np.where(t >= 1, 1.0, cdf((t - m) / s)))


# ---- ordered beta (Kubinec)
def fit_ordered_beta(e, w):
    zero, one, inner = e == 0, e == 1, (e > 0) & (e < 1)
    x, wi, w0, w1 = e[inner], w[inner], w[zero].sum(), w[one].sum()
    lx, l1 = np.log(x), np.log1p(-x)

    def parts(th):
        eta, k1, k2, phi = th[0], th[1], th[1] + np.exp(th[2]), np.exp(th[3])
        return eta, k1, k2, phi, 1 - expit(eta - k1), expit(eta - k2)

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
    run = c.load_stockfish()
    xs = run.x()
    rows = []
    for version, near_ties in (("measured", False), ("near-ties", True)):
        per_depth = mdl.errors_by_depth(run, xs, near_ties)
        for li, depth in enumerate(run.levels):
            e = np.concatenate(per_depth[li])
            w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in per_depth[li]])
            obs = observed_cdf(e, w, GRID)
            for name in MODELS:
                params, ll, cdf = fit_model(name, e, w)
                gap = cdf(GRID) - obs
                k = N_PARAMS[name]
                rows.append([version, depth, name, k, round(ll, 4), round(2 * k - 2 * ll, 4),
                             round(float(np.abs(gap).max()), 5),
                             json.dumps({kk: round(float(vv), 6) for kk, vv in params.items()})])
    c.write_csv(c.OUT / "candidate_distributions.csv", rows,
                ["version", "depth", "model", "n_params", "loglik", "aic", "largest_gap", "params"])
    summarise(rows)


def summarise(rows):
    print("Candidate distributions, fitted at each of the 25 depths (each position weighted equally)")
    for version in ("measured", "near-ties"):
        sel = [r for r in rows if r[0] == version]
        by_depth = collections.defaultdict(list)
        for r in sel:
            by_depth[r[1]].append(r)
        cand = [m for m in MODELS if "benchmark" not in m]
        wins = collections.Counter(min((r for r in fits if r[2] in cand), key=lambda r: r[5])[2] for fits in by_depth.values())
        print(f"\n  {version}: lowest AIC among the six candidates {dict(wins)}")
        for m in MODELS:
            gaps = [r[6] for r in sel if r[2] == m]
            print(f"    {m:<30} median largest gap {np.median(gaps):.3f}  (range {min(gaps):.3f}-{max(gaps):.3f})")


if __name__ == "__main__":
    main()
