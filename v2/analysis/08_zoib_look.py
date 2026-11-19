"""Step 8 (first look): does a zero-one-inflated beta (ZOIB) describe the relative error?

ZOIB for the error e of a move: P(e = 0) = pi0, P(e = 1) = pi1, and otherwise e ~ Beta(a, b) on (0, 1).
Its cumulative share is F(t) = pi0 + (1 - pi0 - pi1) * I_t(a, b) for t < 1 (I = regularised incomplete beta).

Fitted at five levels per engine by maximum likelihood, with each move weighted 1 / (legal moves in its
position) so that every position counts equally; the weighted empirical distribution is then exactly the
average share curve of figures.share_by_tolerance, and the gap between the two measures the fit.
Two versions: all moves, and "other moves", leaving out one best move (e = 0) and one worst move (e = 1)
per position, because every position has those by construction.
This is a pooled fit: it describes the average curve, not how positions differ from one another.
Output: out/zoib_look.csv, printed summary.
"""
import numpy as np
from scipy import optimize
from scipy.special import betainc, betaln

import common as c

GRID = np.linspace(0.0, 0.999, 1000)


def fit_zoib(e, w):
    w = w / w.sum()
    pi0, pi1 = float(w[e == 0].sum()), float(w[e == 1].sum())
    inner = (e > 0) & (e < 1)
    x, wi = e[inner], w[inner] / w[inner].sum()
    lx, l1x = np.log(x), np.log1p(-x)

    def nll(th):
        a, b = np.exp(th)
        return -np.sum(wi * ((a - 1) * lx + (b - 1) * l1x - betaln(a, b)))

    best = min((optimize.minimize(nll, s, method="L-BFGS-B", bounds=[(-6, 6), (-6, 6)]) for s in
                ([0.0, 0.0], [-1.0, -1.0], [1.0, 1.0])), key=lambda r: r.fun)
    a, b = np.exp(best.x)
    return pi0, pi1, float(a), float(b)


def zoib_cdf(t, pi0, pi1, a, b):
    return pi0 + (1 - pi0 - pi1) * betainc(a, b, np.clip(t, 0, 1))


def weighted_cdf(e, w, t):
    w = w / w.sum()
    return np.array([w[e <= x + 1e-12].sum() for x in t])


def main():
    rows = []
    print("Zero-one-inflated beta fitted to the relative error (each position weighted equally)")
    print(f"  {'engine':<10}{'level':>7} {'version':<12}{'pi0':>7}{'pi1':>7}{'a':>7}{'b':>7}   largest gap   gap at t = 0.05 / 0.10 / 0.25")
    for run in (c.load_stockfish(), c.load_lc0()):
        xs = run.x()
        for li in (0, 6, 12, 18, 24):
            errs = [c.relative_errors(xs[p][li:li + 1])[0] for p in run.pos_ids]
            for version in ("all moves", "other moves"):
                e_list, w_list = [], []
                for e in errs:
                    if version == "other moves":
                        if len(e) < 3:
                            continue
                        e = np.sort(e)
                        e = e[1:-1] if e[-1] == 1.0 else e[1:]
                    e_list.append(e)
                    w_list.append(np.full(len(e), 1.0 / len(e)))
                e_all, w_all = np.concatenate(e_list), np.concatenate(w_list)
                pi0, pi1, a, b = fit_zoib(e_all, w_all)
                gap = np.abs(zoib_cdf(GRID, pi0, pi1, a, b) - weighted_cdf(e_all, w_all, GRID))
                at = [float(np.abs(zoib_cdf(np.array([t]), pi0, pi1, a, b) - weighted_cdf(e_all, w_all, np.array([t])))[0])
                      for t in c.THRESHOLDS]
                rows.append([run.engine, run.levels[li], version, round(pi0, 4), round(pi1, 4), round(a, 4), round(b, 4),
                             round(float(gap.max()), 4), round(float(GRID[gap.argmax()]), 3)] + [round(v, 4) for v in at])
                print(f"  {run.engine:<10}{run.levels[li]:>7} {version:<12}{pi0:>7.3f}{pi1:>7.3f}{a:>7.3f}{b:>7.3f}"
                      f"   {gap.max():.3f} at {GRID[gap.argmax()]:.2f}   " + " / ".join(f"{v:.3f}" for v in at))
    c.write_csv(c.OUT / "zoib_look.csv", rows, ["engine", "level", "version", "pi0", "pi1", "a", "b", "largest_gap",
                                                "t_of_largest_gap", "gap_t0.05", "gap_t0.10", "gap_t0.25"])


if __name__ == "__main__":
    main()
