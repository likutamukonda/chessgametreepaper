"""8. Are the parameters measuring different things? (Stockfish, near-ties counted as ties)

Three kinds of correlation, each answering a different question:
1. Across positions (depth 25, Spearman): each position's own p0, p1, middle share, m and s (m and s only for
   positions with at least three moves strictly between 0 and 1), its ties with the best and with the worst beyond
   its own best and worst move, its share within 0.10, its number of legal moves, its number of pieces, and how
   decided it is (the gap between its best and worst expected score). Do the parameters pick up distinct features
   of positions, or one common factor?
2. Between the estimates (depths 5 and 25, Pearson and Spearman over the bootstrap resamples of positions, with the
   model share within 0.10 they imply): when a resample gives a higher p0, does it also give a different p1, m or s?
   This is the statistical entanglement of the estimates themselves. p0 + p1 + middle = 1 by construction, so the
   middle share is not free.
3. Between the infinite-depth levels (Pearson and Spearman over the resamples in which every depth function has a
   level; gamma kernel for s): do the extrapolated levels trade off against each other?
Output: out/parameter_correlations.csv
"""
import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr

import common as c
import model as mdl

ORDER = ("p0", "p1", "middle", "m", "s")


def matrix_rows(block, names, mat, note=""):
    return [[block, note, a, b, round(float(mat[i, j]), 4)] for i, a in enumerate(names) for j, b in enumerate(names) if j > i]


def show(title, names, mat, width=8):
    print(title)
    print(" " * 14 + "".join(f"{n:>{width}}" for n in names))
    for i, a in enumerate(names):
        print(f"  {a:<12}" + "".join(f"{mat[i, j]:>{width}.2f}" if j != i else f"{'':>{width}}" for j in range(len(names))))


def main():
    S = mdl.limits()
    run, xs, d = S["run"], S["xs"], S["d"]
    rows = []

    # ---- 1. across positions, depth 25
    li = list(d).index(25)
    feats = {k: [] for k in ("p0", "p1", "middle", "m", "s", "ties best", "ties worst", "within 0.10", "legal moves",
                             "pieces", "decided")}
    for p, e in zip(run.pos_ids, S["per_depth"][li]):
        x, n = xs[p][li], len(e)
        same = np.all(e == 0)
        inner = e[(e > 0) & (e < 1)]
        z = logit(inner) if len(inner) >= 3 else None
        feats["p0"].append(np.mean(e == 0))
        feats["p1"].append(np.mean(e == 1))
        feats["middle"].append(np.mean((e > 0) & (e < 1)))
        feats["m"].append(np.mean(z) if z is not None else np.nan)
        feats["s"].append(np.std(z) if z is not None else np.nan)
        feats["ties best"].append(0.0 if same else (np.sum(e == 0) - 1) / n)
        feats["ties worst"].append(0.0 if same else (np.sum(e == 1) - 1) / n)
        feats["within 0.10"].append(np.mean(e <= 0.10 + 1e-12))
        feats["legal moves"].append(n)
        feats["pieces"].append(c.piece_count(run.fens[p]))
        feats["decided"].append(float(x.max() - x.min()))
    names = list(feats)
    X = np.array([feats[k] for k in names], float).T
    ok = ~np.isnan(X).any(axis=1)
    rho = spearmanr(X[ok]).correlation
    rows += matrix_rows("positions, depth 25 (Spearman)", names, rho, f"{ok.sum()} positions")
    show(f"1. Across positions at depth 25 (Spearman; {ok.sum()} positions with at least three moves strictly between)",
         names, rho, width=12)

    # ---- 2. between the estimates, over bootstrap resamples (with the model share within 0.10 they imply)
    names = ORDER + ("within 0.10",)
    for dep in (5, 25):
        lj = list(d).index(dep)
        R = np.array([[r[q][lj] for q in ORDER] + [mdl.model_curve(mdl.at(r, lj), np.array([0.10]))[0]] for r in S["reps"]])
        cor, rho = np.corrcoef(R, rowvar=False), spearmanr(R).correlation
        rows += matrix_rows(f"estimates, depth {dep} (Pearson over resamples)", names, cor, f"{len(R)} resamples")
        rows += matrix_rows(f"estimates, depth {dep} (Spearman over resamples)", names, rho, f"{len(R)} resamples")
        show(f"\n2. Between the estimates at depth {dep} (Pearson over {len(R)} bootstrap resamples)", names, cor, width=12)
        show("   Spearman", names, rho, width=12)

    # ---- 3. between the infinite-depth levels
    bl = mdl.boot_levels(S, "s")
    Lm = np.array([[L[q] for q in ORDER] + [mdl.model_curve(L, np.array([0.10]))[0]] for L in bl])
    cor3, rho3 = np.corrcoef(Lm, rowvar=False), spearmanr(Lm).correlation
    rows += matrix_rows("infinite-depth levels (Pearson over resamples)", names, cor3, f"{len(bl)} resamples")
    rows += matrix_rows("infinite-depth levels (Spearman over resamples)", names, rho3, f"{len(bl)} resamples")
    show(f"\n3. Between the infinite-depth levels (Pearson over the {len(bl)} resamples with every level; gamma kernel for s)",
         names, cor3, width=12)
    show("   Spearman", names, rho3, width=12)
    c.write_csv(c.OUT / "parameter_correlations.csv", rows, ["block", "note", "a", "b", "correlation"])


if __name__ == "__main__":
    main()
