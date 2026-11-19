"""Step 2: which distribution fits the number of near-best moves, at every depth and node level?

Data for one engine, level and tolerance: in each position, K = the number of moves other than the
best one with e <= tolerance, out of n = legal moves - 1. The best move always has e = 0, so the
share of near-best moves (K + 1) / (n + 1) can never fall below 1 / legal moves; modelling K out of
n respects that and copes with positions having different numbers of legal moves. Positions with a
single legal move carry no information and are left out.

Candidate models, all fitted by maximum likelihood:
  binomial                  every other move has the same chance p of being near-best, everywhere
  beta-binomial             that chance varies between positions as Beta(mu*phi, (1 - mu)*phi)
  zero-inflated BB          plus an extra probability pi0 that only the best move is near-best (K = 0)
  zero-and-all-inflated BB  plus also an extra probability pi1 that every move is near-best (K = n)
  BB with legal-move trend  beta-binomial whose mean depends on the number of legal moves:
                            logit(mu) = a + b * (log(legal moves) - its average)

Goodness of fit: randomised probability-integral transform (PIT) values, which are uniform on (0, 1)
when the model is right, tested with Kolmogorov-Smirnov (approximate, since parameters are fitted).

Output: out/count_fits.csv, one row per engine, level, tolerance and model.
"""
import numpy as np
from scipy import optimize, stats
from scipy.special import betaln, expit, gammaln, logit

import common as c

MODELS = ["binomial", "beta-binomial", "zero-inflated BB", "zero-and-all-inflated BB", "BB with legal-move trend"]
N_PARAMS = {"binomial": 1, "beta-binomial": 2, "zero-inflated BB": 3, "zero-and-all-inflated BB": 4,
            "BB with legal-move trend": 3}
BOUNDS = {"binomial": [(-15, 15)],
          "beta-binomial": [(-12, 12), (-7, 12)],
          "zero-inflated BB": [(-12, 12), (-7, 12), (-15, 15)],
          "zero-and-all-inflated BB": [(-12, 12), (-7, 12), (-15, 15), (-15, 15)],
          "BB with legal-move trend": [(-12, 12), (-10, 10), (-7, 12)]}


def log_choose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logpmf(model, theta, k, n, z):
    """Log-probability of K = k out of n under a model; works element-wise on broadcast arrays."""
    with np.errstate(all="ignore"):
        if model == "binomial":
            p = expit(theta[0])
            return log_choose(n, k) + k * np.log(p) + (n - k) * np.log1p(-p)
        if model == "BB with legal-move trend":
            mu, phi = expit(theta[0] + theta[1] * z), np.exp(theta[2])
        else:
            mu, phi = expit(theta[0]), np.exp(theta[1])
        a, b = mu * phi, (1.0 - mu) * phi
        base = log_choose(n, k) + betaln(k + a, n - k + b) - betaln(a, b)
        if model in ("beta-binomial", "BB with legal-move trend"):
            return base
        if model == "zero-inflated BB":
            pi0 = expit(theta[2])
            body = np.log1p(-pi0) + base
            return np.where(k == 0, np.logaddexp(np.log(pi0), body), body)
        w = np.exp(np.array([theta[2], theta[3], 0.0]))
        pi0, pi1, rest = w / w.sum()
        body = np.log(rest) + base
        body = np.where(k == 0, np.logaddexp(np.log(pi0), body), body)
        return np.where(k == n, np.logaddexp(np.log(pi1), body), body)


def params_dict(model, theta):
    d = {"mu": "", "phi": "", "pi0": "", "pi1": "", "trend_b": ""}
    if model == "binomial":
        d["mu"] = expit(theta[0])
    elif model == "BB with legal-move trend":
        d["mu"], d["trend_b"], d["phi"] = expit(theta[0]), theta[1], np.exp(theta[2])
    else:
        d["mu"], d["phi"] = expit(theta[0]), np.exp(theta[1])
        if model == "zero-inflated BB":
            d["pi0"] = expit(theta[2])
        elif model == "zero-and-all-inflated BB":
            w = np.exp(np.array([theta[2], theta[3], 0.0]))
            d["pi0"], d["pi1"] = (w / w.sum())[:2]
    return d


def fit(model, k, n, z):
    p0 = float(np.clip(k.sum() / n.sum(), 1e-6, 1 - 1e-6))
    if model == "binomial":
        theta = np.array([logit(p0)])
        return theta, float(logpmf(model, theta, k, n, z).sum())
    starts = []
    for lphi in (np.log(0.5), np.log(2.0), np.log(10.0)):
        if model == "BB with legal-move trend":
            starts.append([logit(p0), 0.0, lphi])
        else:
            starts.append([logit(p0), lphi] + {"beta-binomial": [], "zero-inflated BB": [-2.0],
                                               "zero-and-all-inflated BB": [-2.0, -2.0]}[model])
    best = None
    for s in starts:
        res = optimize.minimize(lambda th: -logpmf(model, th, k, n, z).sum(), np.array(s, float),
                                method="L-BFGS-B", bounds=BOUNDS[model])
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    return best.x, -float(best.fun)


def pmf_matrix(model, theta, n, z):
    """Array (positions, max n + 1): probability of each K for each position (0 beyond its n)."""
    ks = np.arange(n.max() + 1)[None, :].astype(float)
    lp = logpmf(model, theta, ks, n[:, None].astype(float), z[:, None])
    return np.where(ks <= n[:, None], np.exp(lp), 0.0)


def diagnostics(model, theta, k, n, z, seed):
    pm = pmf_matrix(model, theta, n, z)
    cdf = np.cumsum(pm, axis=1)
    rows = np.arange(len(n))
    upper = cdf[rows, k]
    lower = np.where(k > 0, cdf[rows, np.maximum(k - 1, 0)], 0.0)
    u = lower + np.random.default_rng(seed).random(len(n)) * (upper - lower)
    ks_stat, ks_p = stats.kstest(np.clip(u, 0, 1), "uniform")
    ks_range = np.arange(pm.shape[1])
    mean_share = float(np.mean((1 + (pm * ks_range).sum(axis=1)) / (n + 1)))
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "exp_K0": float(pm[:, 0].mean()), "exp_Kn": float(pm[rows, n].mean()), "exp_share": mean_share}


def run_engine(run, rows, seed_base):
    counts, n_legal = c.near_best_counts(run)
    keep = n_legal >= 2
    n = n_legal[keep] - 1
    z = np.log(n_legal[keep]) - np.log(n_legal[keep]).mean()
    for li, level in enumerate(run.levels):
        for ti, t in enumerate(c.THRESHOLDS):
            k = counts[keep, li, ti] - 1
            assert (k >= 0).all() and (k <= n).all()
            obs = {"obs_K0": float(np.mean(k == 0)), "obs_Kn": float(np.mean(k == n)),
                   "obs_share": float(np.mean((k + 1) / (n + 1)))}
            for mi, model in enumerate(MODELS):
                theta, ll = fit(model, k, n, z)
                q = N_PARAMS[model]
                aic, bic = 2 * q - 2 * ll, q * np.log(len(k)) - 2 * ll
                dg = diagnostics(model, theta, k, n, z, seed_base + 1000 * li + 10 * ti + mi)
                p = params_dict(model, theta)
                rows.append([run.engine, li, level, t, model, q, round(ll, 3), round(aic, 3), round(bic, 3),
                             round(dg["ks_stat"], 4), round(dg["ks_p"], 4)]
                            + [("" if p[x] == "" else round(float(p[x]), 5)) for x in ("mu", "phi", "pi0", "pi1", "trend_b")]
                            + [round(obs["obs_K0"], 4), round(dg["exp_K0"], 4), round(obs["obs_Kn"], 4),
                               round(dg["exp_Kn"], 4), round(obs["obs_share"], 4), round(dg["exp_share"], 4), len(k)])


HEADER = ["engine", "level_index", "level", "threshold", "model", "n_params", "loglik", "aic", "bic", "ks_stat",
          "ks_p", "mu", "phi", "pi0", "pi1", "trend_b", "obs_K0", "exp_K0", "obs_Kn", "exp_Kn", "obs_share",
          "exp_share", "positions"]


def summarise(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[0], r[1], r[3])].append(r)
    print("Models with the lowest AIC / BIC, counted over the 25 levels")
    for eng in ("Stockfish", "Lc0"):
        for t in c.THRESHOLDS:
            aic_w = collections.Counter(); bic_w = collections.Counter(); ks_ok = collections.Counter()
            for li in range(25):
                fits = by[(eng, li, t)]
                aic_w[min(fits, key=lambda r: r[7])[4]] += 1
                bic_w[min(fits, key=lambda r: r[8])[4]] += 1
                for r in fits:
                    ks_ok[r[4]] += r[10] >= 0.05
            print(f"  {eng:<9} e<={t:<4} AIC: {dict(aic_w)}")
            print(f"  {'':<9} {'':<7} BIC: {dict(bic_w)}")
            print(f"  {'':<9} {'':<7} levels where KS p >= 0.05: " + ", ".join(f"{m} {ks_ok[m]}" for m in MODELS))
    print("\nTop level (Stockfish depth 25, Lc0 10,000 nodes): delta AIC from the best, KS p, K=0 and K=n observed vs model")
    for eng in ("Stockfish", "Lc0"):
        for t in c.THRESHOLDS:
            fits = by[(eng, 24, t)]
            best_aic = min(r[7] for r in fits)
            print(f"  {eng} e<={t}: positions {fits[0][-1]}, observed K=0 {fits[0][16]:.3f}, K=n {fits[0][18]:.3f}")
            for r in fits:
                extra = f" mu {r[11]:.3f} phi {r[12] if r[12] == '' else round(r[12], 3)}"
                if r[13] != "": extra += f" pi0 {r[13]:.3f}"
                if r[14] != "": extra += f" pi1 {r[14]:.3f}"
                if r[15] != "": extra += f" trend b {r[15]:+.3f}"
                print(f"     {r[4]:<26} dAIC {r[7] - best_aic:>8.2f}  KS p {r[10]:.3f}  model K=0 {r[17]:.3f} K=n {r[19]:.3f}  {extra}")


def main():
    rows = []
    run_engine(c.load_stockfish(), rows, 1)
    run_engine(c.load_lc0(), rows, 500000)
    c.write_csv(c.OUT / "count_fits.csv", rows, HEADER)
    summarise(rows)


if __name__ == "__main__":
    main()
