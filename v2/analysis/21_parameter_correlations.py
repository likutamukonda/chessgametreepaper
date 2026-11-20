"""Step 21: are the parameters measuring different things? (Stockfish, near-ties counted as ties)

Three kinds of correlation, each answering a different question:
1. Across positions (depth 25, Spearman): for each position its own p0, p1, middle share, m and s (m and s
   only for positions with at least three moves strictly between 0 and 1), its ties with the best and with the
   worst beyond its own best and worst move, its share within 0.10, its number of legal moves, its number of
   pieces, and how decided it is (the gap between its best and worst expected score). Do the parameters pick
   up distinct features of positions, or one common factor?
2. Between the estimates (depths 5 and 25, Pearson over the 200 bootstrap resamples of positions): when a
   resample gives a higher p0, does it also give a different p1, m or s? This is the statistical entanglement
   of the estimates themselves. p0 + p1 + middle = 1 by construction, so middle is not free.
3. Between the infinite-depth levels (Pearson over the resamples in which every depth function has a level;
   gamma kernel for s): do the extrapolated levels trade off against each other?
Output: out/parameter_correlations.csv, printed summary.
"""
import importlib

import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr

import common as c

f15 = importlib.import_module("15_infinite_depth")
f16 = importlib.import_module("16_adjusted_infinite_depth")
ALL = f15.ALL


def matrix_rows(block, names, mat, extra=""):
    return [[block, extra, a, b, round(float(mat[i, j]), 4)] for i, a in enumerate(names) for j, b in enumerate(names) if j > i]


def show(title, names, mat, width=8):
    print(title)
    print(" " * 14 + "".join(f"{n:>{width}}" for n in names))
    for i, a in enumerate(names):
        print(f"  {a:<12}" + "".join(f"{mat[i, j]:>{width}.2f}" if j != i else f"{'':>{width}}" for j in range(len(names))))


def main():
    S = f16.stockfish_limits()
    run, xs, d = S["run"], S["xs"], S["d"]
    rows = []

    # ---- 1. across positions, depth 25
    li = list(d).index(25)
    per = S["per_depth"][li]
    feats = {k: [] for k in ("p0", "p1", "middle", "m", "s", "ties best", "ties worst", "within 0.10", "legal moves",
                             "pieces", "decided")}
    for p, e in zip(run.pos_ids, per):
        x = xs[p][li]
        n = len(e)
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
        feats["decided"].append(float(np.nanmax(x) - np.nanmin(x)))
    names = list(feats)
    X = np.array([feats[k] for k in names], float).T
    ok = ~np.isnan(X).any(axis=1)
    rho = spearmanr(X[ok]).correlation
    rows += matrix_rows("positions, depth 25 (Spearman)", names, rho, f"{ok.sum()} positions")
    show(f"1. Across positions at depth 25 (Spearman; {ok.sum()} positions with at least three moves strictly between)",
         names, rho, width=12)

    # ---- 2. between the estimates, over bootstrap resamples
    for dep in (5, 25):
        lj = list(d).index(dep)
        R = np.array([[r[q][lj] for q in ALL] for r in S["reps"]])
        cor = np.corrcoef(R, rowvar=False)
        rows += matrix_rows(f"estimates, depth {dep} (Pearson over resamples)", ALL, cor)
        show(f"\n2. Between the estimates at depth {dep} (Pearson over {len(R)} bootstrap resamples)", ALL, cor)

    # ---- 3. between the infinite-depth levels
    bl = [L for L in (f16.levels(bf, "s") for bf in S["boot_fits"]) if L is not None]
    Lm = np.array([[L[q] for q in ALL] + [f15.share_curve(L["p0"], L["middle"], L["m"], L["s"], np.array([0.10]))[0]]
                   for L in bl])
    names3 = ALL + ("within 0.10",)
    cor3 = np.corrcoef(Lm, rowvar=False)
    rho3 = spearmanr(Lm).correlation
    rows += matrix_rows("infinite-depth levels (Pearson over resamples)", names3, cor3, f"{len(bl)} resamples")
    rows += matrix_rows("infinite-depth levels (Spearman over resamples)", names3, rho3, f"{len(bl)} resamples")
    show(f"\n3. Between the infinite-depth levels (Spearman over the {len(bl)} resamples with every level; "
         "gamma kernel for s)", names3, rho3, width=12)
    c.write_csv(c.OUT / "parameter_correlations.csv", rows, ["block", "note", "a", "b", "correlation"])


if __name__ == "__main__":
    main()
