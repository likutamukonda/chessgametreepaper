"""Step 10: which distribution best describes the relative error, fitted separately at every level?

Data. At each Stockfish depth and Lc0 node level, the relative errors of the "other" moves of each
position: one best move (e = 0) and one worst move (e = 1) are left out, because every position has
them by construction (when all moves are equal, only one best move is left out and the rest are ties
at 0). What remains can still have exact ties at 0 (as good as the best) and at 1 (as bad as the worst).
The share of moves within a tolerance follows directly: share(t) = (1 + m * F(t)) / n for t < 1, with
n legal moves, m other moves and F the distribution of the other moves' errors.

Families, all with a probability mass at 0, one at 1, and a density in between:
  ZOIB         zero-one-inflated beta: free masses pi0, pi1, Beta(a, b) in between (4 parameters).
               At a single level the ordered beta (Kubinec) is the same distribution, reparametrised;
               the two differ only once depth enters as a covariate.
  ZOI-Kuma     zero-one-inflated Kumaraswamy: free masses, Kumaraswamy(a, b) in between (4).
  ZOI-logitN   zero-one-inflated logit-normal: free masses, logistic(Normal(mu, sigma)) in between (4).
               (Johnson's SB distribution on [0, 1] is the same family.)
  cens-normal  censored normal: a normal latent value, piled up at 0 below 0 and at 1 above 1 (2).
  cens-logist  censored logistic: the same with a logistic latent value (2).
  ZOI-2logitN  zero-one-inflated mixture of two logit-normals (7): two clusters of errors; fitted by EM
               with each spread held at 0.3 or more on the logit scale.
  ZOI-2beta    zero-one-inflated mixture of two betas (7): a flexible benchmark, pooled fits only.

Fits, every one by maximum likelihood with its own parameters:
  by level     all positions pooled (every move counts once).
  by position  each position separately (positions with >= 10 other moves and >= 5 strictly between
               0 and 1), all families except the two-beta mixture.
Comparison: log-likelihood, AIC and BIC; the largest distance between the fitted and the empirical
distribution (Kolmogorov-Smirnov distance); and the largest gap between the share curve the fit implies
and the observed one (each position weighted equally, as in share_by_tolerance.png).

Outputs: out/error_fits_by_level.csv, out/error_fits_by_position.csv, printed summary.
"""
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import optimize, stats
from scipy.special import betainc, betaln, expit

import common as c

FAMILIES = ["ZOIB", "ZOI-Kuma", "ZOI-logitN", "cens-normal", "cens-logist", "ZOI-2logitN", "ZOI-2beta"]
PER_POSITION = FAMILIES[:-1]          # the two-beta mixture is fitted to the pooled data only
N_PARAMS = {"ZOIB": 4, "ZOI-Kuma": 4, "ZOI-logitN": 4, "cens-normal": 2, "cens-logist": 2, "ZOI-2logitN": 7, "ZOI-2beta": 7}
MIN_OTHER, MIN_INTERIOR = 10, 5
GRID = np.concatenate([[0.0], np.linspace(1e-4, 0.999, 1200)])


def other_errors(e):
    e = np.sort(e)
    if len(e) < 2:
        return e[:0]
    return e[1:-1] if e[-1] == 1.0 else e[1:]


# ---------------------------------------------------------------- the families
def spikes(n0, n1, n):
    """Free masses at 0 and 1 (maximum likelihood = observed fractions) and their log-likelihood."""
    pi0, pi1 = n0 / n, n1 / n
    m = n - n0 - n1
    ll = (n0 * math.log(pi0) if n0 else 0.0) + (n1 * math.log(pi1) if n1 else 0.0) + (m * math.log(1 - pi0 - pi1) if m else 0.0)
    return ll, pi0, pi1


def minimise(nll, starts, bounds):
    best = None
    for s in starts:
        try:
            r = optimize.minimize(nll, np.array(s, float), method="L-BFGS-B", bounds=bounds)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


def fit_family(fam, x):
    """x: errors in [0, 1]. Returns (loglik, params, cdf) where cdf(t) gives P(e <= t) for t in [0, 1)."""
    n = len(x)
    n0, n1 = int(np.sum(x == 0.0)), int(np.sum(x == 1.0))
    inner = x[(x > 0.0) & (x < 1.0)]
    m = len(inner)
    lx, l1x = np.log(inner), np.log1p(-inner)
    if fam in ("ZOIB", "ZOI-Kuma", "ZOI-logitN", "ZOI-2logitN", "ZOI-2beta"):
        ll_s, pi0, pi1 = spikes(n0, n1, n)
        body = 1.0 - pi0 - pi1
        if fam == "ZOIB":
            s1, s2 = lx.sum(), l1x.sum()
            r = minimise(lambda th: -((np.exp(th[0]) - 1) * s1 + (np.exp(th[1]) - 1) * s2 - m * betaln(*np.exp(th))),
                         ([0, 0], [-1, -1], [1, 1]), [(-7, 7), (-7, 7)])
            a, b = np.exp(r.x)
            return ll_s - r.fun, {"pi0": pi0, "pi1": pi1, "a": a, "b": b}, lambda t: pi0 + body * betainc(a, b, t)
        if fam == "ZOI-Kuma":
            def nll(th):
                a, b = np.exp(th)
                xa = np.minimum(np.exp(a * lx), 1 - 1e-15)
                return -(m * (th[0] + th[1]) + (a - 1) * lx.sum() + (b - 1) * np.log1p(-xa).sum())
            r = minimise(nll, ([0, 0], [-1, -1], [1, 1], [-1, 1]), [(-7, 7), (-7, 7)])
            a, b = np.exp(r.x)
            return ll_s - r.fun, {"pi0": pi0, "pi1": pi1, "a": a, "b": b}, \
                lambda t: pi0 + body * (1 - (1 - np.clip(t, 0, 1) ** a) ** b)
        if fam == "ZOI-logitN":
            y = lx - l1x
            mu, sd = float(y.mean()), max(float(y.std()), 1e-6)
            ll = float(np.sum(stats.norm.logpdf(y, mu, sd) - lx - l1x))
            return ll_s + ll, {"pi0": pi0, "pi1": pi1, "mu": mu, "sigma": sd}, \
                lambda t: pi0 + body * np.where(t > 0, stats.norm.cdf((np.log(np.maximum(t, 1e-300)) - np.log1p(-np.minimum(t, 1 - 1e-12)) - mu) / sd), 0.0)
        if fam == "ZOI-2logitN":
            y = lx - l1x
            ll_y, (w, m1, s1, m2, s2) = em_two_normals(y)
            def cdf2(t, w=w, m1=m1, s1=s1, m2=m2, s2=s2):
                z = np.log(np.maximum(t, 1e-300)) - np.log1p(-np.minimum(t, 1 - 1e-12))
                g = w * stats.norm.cdf((z - m1) / s1) + (1 - w) * stats.norm.cdf((z - m2) / s2)
                return pi0 + body * np.where(t > 0, g, 0.0)
            return ll_s + ll_y - float(np.sum(lx + l1x)), {"pi0": pi0, "pi1": pi1, "w": w, "mu1": m1, "sigma1": s1,
                                                          "mu2": m2, "sigma2": s2}, cdf2
        # two-beta mixture
        def comp(a, b):
            return (a - 1) * lx + (b - 1) * l1x - betaln(a, b)
        def nll(th):
            w = expit(th[0]); a1, b1, a2, b2 = np.exp(th[1:])
            return -np.sum(np.logaddexp(np.log(w) + comp(a1, b1), np.log1p(-w) + comp(a2, b2)))
        r = minimise(nll, ([0, np.log(.5), np.log(2), np.log(2), np.log(.5)], [0, np.log(.7), np.log(3), np.log(5), np.log(1.2)],
                           [1, np.log(.4), np.log(1.5), np.log(1.5), np.log(.4)], [-1, np.log(1), np.log(8), np.log(.6), np.log(.6)]),
                     [(-8, 8)] + [(-6, 6)] * 4)
        w = float(expit(r.x[0])); a1, b1, a2, b2 = (float(v) for v in np.exp(r.x[1:]))
        return ll_s - r.fun, {"pi0": pi0, "pi1": pi1, "w": w, "a1": a1, "b1": b1, "a2": a2, "b2": b2}, \
            lambda t: pi0 + body * (w * betainc(a1, b1, t) + (1 - w) * betainc(a2, b2, t))
    # censored latent distributions: the masses at 0 and 1 come from the same curve as the middle
    dist = stats.norm if fam == "cens-normal" else stats.logistic
    def nll(th):
        mu, s = th[0], math.exp(th[1])
        ll = (n0 * dist.logcdf(-mu / s) if n0 else 0.0) + (n1 * dist.logsf((1 - mu) / s) if n1 else 0.0)
        return -(ll + np.sum(dist.logpdf((inner - mu) / s)) - m * th[1])
    r = minimise(nll, ([0.2, np.log(0.3)], [0.5, np.log(0.5)], [0.8, np.log(0.3)], [0.5, np.log(1.5)], [0.0, np.log(1.0)]),
                 [(-5, 6), (-6, 4)])
    mu, s = float(r.x[0]), float(math.exp(r.x[1]))
    return -r.fun, {"mu": mu, "scale": s}, lambda t: dist.cdf((t - mu) / s)


def em_two_normals(y, floor=0.3, iters=500):
    """Two-component normal mixture by EM (spreads held at `floor` or more); returns (loglik, params)."""
    sd = max(float(y.std()), floor)
    neg, pos = y[y < 0], y[y >= 0]
    starts = [(0.5, float(np.quantile(y, 0.25)), sd / 2, float(np.quantile(y, 0.75)), sd / 2),
              (0.5, float(y.mean()) - sd, sd / 2, float(y.mean()) + sd, sd / 2)]
    if len(neg) >= 2 and len(pos) >= 2:
        starts.append((len(neg) / len(y), float(neg.mean()), max(float(neg.std()), floor), float(pos.mean()), max(float(pos.std()), floor)))
    best = None
    for w, m1, s1, m2, s2 in starts:
        prev = -np.inf
        for _ in range(iters):
            l1 = np.log(w) + stats.norm.logpdf(y, m1, s1)
            l2 = np.log1p(-w) + stats.norm.logpdf(y, m2, s2)
            tot = np.logaddexp(l1, l2)
            ll = float(tot.sum())
            r = np.exp(l1 - tot)
            w = float(np.clip(r.mean(), 0.02, 0.98))
            m1, m2 = float(np.sum(r * y) / r.sum()), float(np.sum((1 - r) * y) / (1 - r).sum())
            s1 = max(float(np.sqrt(np.sum(r * (y - m1) ** 2) / r.sum())), floor)
            s2 = max(float(np.sqrt(np.sum((1 - r) * (y - m2) ** 2) / (1 - r).sum())), floor)
            if ll - prev < 1e-9:
                break
            prev = ll
        ll = float(np.sum(np.logaddexp(np.log(w) + stats.norm.logpdf(y, m1, s1), np.log1p(-w) + stats.norm.logpdf(y, m2, s2))))
        if best is None or ll > best[0]:
            best = (ll, (w, m1, s1, m2, s2) if m1 <= m2 else (1 - w, m2, s2, m1, s1))
    return best


# ---------------------------------------------------------------- one level
def fit_level(task):
    engine, li, level, others, n_legal, observed_curve = task
    pooled = np.concatenate([o for o in others if len(o)])
    n_obs = len(pooled)
    emp = np.searchsorted(np.sort(pooled), GRID, side="right") / n_obs
    rows_level = []
    for fam in FAMILIES:
        ll, params, cdf = fit_family(fam, pooled)
        q = N_PARAMS[fam]
        F = cdf(GRID)
        ks = float(np.max(np.abs(F - emp)))
        implied = np.mean([np.ones_like(GRID) if n == 1 else (1 + len(o) * F) / n for o, n in zip(others, n_legal)], axis=0)
        gap = np.abs(implied - observed_curve)
        at = [float(gap[np.argmin(np.abs(GRID - t))]) for t in c.THRESHOLDS]
        rows_level.append([engine, li, level, fam, q, n_obs, int(np.sum(pooled == 0)), int(np.sum(pooled == 1)), round(ll, 2),
                           round(2 * q - 2 * ll, 2), round(q * math.log(n_obs) - 2 * ll, 2), round(ks, 4),
                           round(float(gap.max()), 4)] + [round(v, 4) for v in at] + [json.dumps({k: round(float(v), 5) for k, v in params.items()})])
    # each position separately
    use = [i for i, o in enumerate(others) if len(o) >= MIN_OTHER and np.sum((o > 0) & (o < 1)) >= MIN_INTERIOR]
    per = {fam: {"ll": 0.0, "aic": 0.0, "wins": 0, "curves": []} for fam in PER_POSITION}
    for i in use:
        aics = {}
        for fam in PER_POSITION:
            ll, params, cdf = fit_family(fam, others[i])
            aic = 2 * N_PARAMS[fam] - 2 * ll
            per[fam]["ll"] += ll; per[fam]["aic"] += aic; aics[fam] = aic
            per[fam]["curves"].append((1 + len(others[i]) * cdf(GRID)) / n_legal[i])
        per[min(aics, key=aics.get)]["wins"] += 1
    obs_use = np.mean([np.mean(np.concatenate([[0.0], others[i], [1.0]])[:, None] <= GRID[None, :] + 1e-12, axis=0)
                       for i in use], axis=0) if use else None
    rows_pos = []
    for fam in PER_POSITION:
        gap = float(np.max(np.abs(np.mean(per[fam]["curves"], axis=0) - obs_use))) if use else float("nan")
        rows_pos.append([engine, li, level, fam, len(use), round(per[fam]["ll"], 2), round(per[fam]["aic"], 2),
                         per[fam]["wins"], round(gap, 4)])
    return rows_level, rows_pos


def tasks():
    out = []
    for run in (c.load_stockfish(), c.load_lc0()):
        xs = run.x()
        n_legal = [len(run.moves[p]) for p in run.pos_ids]
        for li, level in enumerate(run.levels):
            errs = [c.relative_errors(xs[p][li:li + 1])[0] for p in run.pos_ids]
            observed = np.mean([(e[:, None] <= GRID[None, :] + 1e-12).mean(axis=0) for e in errs], axis=0)
            out.append((run.engine, li, level, [other_errors(e) for e in errs], n_legal, observed))
    return out


def summarise(rows_level, rows_pos):
    import collections
    print("By level (all positions pooled): which family has the lowest AIC and BIC, out of 25 levels")
    for eng in ("Stockfish", "Lc0"):
        by = collections.defaultdict(list)
        for r in rows_level:
            if r[0] == eng:
                by[r[1]].append(r)
        aic_w = collections.Counter(min(v, key=lambda r: r[9])[3] for v in by.values())
        bic_w = collections.Counter(min(v, key=lambda r: r[10])[3] for v in by.values())
        print(f"  {eng}: AIC {dict(aic_w)} | BIC {dict(bic_w)}")
    print("\nBy level, at five levels: AIC above the best family / KS distance / largest gap in the share curve")
    for eng, picks in (("Stockfish", (0, 6, 12, 18, 24)), ("Lc0", (0, 6, 12, 18, 24))):
        for li in picks:
            fits = [r for r in rows_level if r[0] == eng and r[1] == li]
            best = min(r[9] for r in fits)
            print(f"  {eng:<9} {fits[0][2]:>6}: " + "  ".join(f"{r[3]} {r[9] - best:>7.0f}/{r[11]:.3f}/{r[12]:.3f}" for r in fits))
    print("\nBy position (each position its own parameters), summed over positions: AIC above the best family, and wins")
    for eng, picks in (("Stockfish", (0, 6, 12, 18, 24)), ("Lc0", (0, 6, 12, 18, 24))):
        for li in picks:
            fits = [r for r in rows_pos if r[0] == eng and r[1] == li]
            best = min(r[6] for r in fits)
            print(f"  {eng:<9} {fits[0][2]:>6} ({fits[0][4]} positions): " +
                  "  ".join(f"{r[3]} {r[6] - best:>6.0f} ({r[7]} wins, gap {r[8]:.3f})" for r in fits))


def main():
    work = tasks()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fit_level, work))
    rows_level = [r for rl, _ in results for r in rl]
    rows_pos = [r for _, rp in results for r in rp]
    c.write_csv(c.OUT / "error_fits_by_level.csv", rows_level,
                ["engine", "level_index", "level", "family", "n_params", "n_moves", "n_at_0", "n_at_1", "loglik", "aic", "bic",
                 "ks_distance", "largest_share_gap"] + [f"share_gap_t{t}" for t in c.THRESHOLDS] + ["params"])
    c.write_csv(c.OUT / "error_fits_by_position.csv", rows_pos,
                ["engine", "level_index", "level", "family", "positions", "total_loglik", "total_aic", "positions_won",
                 "largest_share_gap"])
    summarise(rows_level, rows_pos)


if __name__ == "__main__":
    main()
