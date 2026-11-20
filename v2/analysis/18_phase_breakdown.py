"""Step 18: the error distribution by game phase (Stockfish, near-ties counted as ties).

Positions are grouped by the number of pieces on the board, kings and pawns included: 12 or fewer
(endgames), 13-22 (middlegames) and 23 or more (openings and early middlegames, after the book).
For each group, at depths 5 and 25: number of positions, mean number of legal moves, the five
parameters of the zero-one-inflated logit-normal (closed form, each position weighted equally, as in
steps 12-16) and the observed share of moves within 0.05, 0.10, 0.25 and 0.5, with bootstrap standard
errors (200 resamples of the positions within each group).
Output: out/phase_breakdown.csv
"""
import importlib

import numpy as np

import common as c

f15 = importlib.import_module("15_infinite_depth")
B, SEED, SNAP = 200, 20261120, 1e-3
GROUPS = (("12 or fewer", 0, 12), ("13-22", 13, 22), ("23 or more", 23, 32))
DEPTHS = (5, 25)
TOL = (0.05, 0.10, 0.25, 0.5)


def summary(per_pos, n_legal):
    """per_pos: list of error arrays (one depth) -> dict of the quantities reported."""
    ser = f15.series([per_pos], np.arange(len(per_pos)))
    out = {q: float(ser[q][0]) for q in f15.ALL}
    for t in TOL:
        out[f"within {t}"] = float(np.mean([np.mean(e <= t + 1e-12) for e in per_pos]))
    out["legal moves"] = float(np.mean(n_legal))
    return out


def main():
    run = c.load_stockfish()
    xs = run.x()
    snap = lambda e: np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e))
    pieces = np.array([c.piece_count(run.fens[p]) for p in run.pos_ids])
    n_legal = np.array([xs[p].shape[1] for p in run.pos_ids])
    rng = np.random.default_rng(SEED)
    rows = []
    for name, lo, hi in GROUPS + (("all", 0, 32),):
        idx = np.flatnonzero((pieces >= lo) & (pieces <= hi))
        boots = [rng.choice(idx, len(idx), replace=True) for _ in range(B)]
        for dep in DEPTHS:
            li = run.levels.index(dep)
            errs = {i: snap(c.relative_errors(xs[run.pos_ids[i]][li:li + 1])[0]) for i in idx}
            full = summary([errs[i] for i in idx], n_legal[idx])
            reps = [summary([errs[i] for i in b], n_legal[b]) for b in boots]
            for q, v in full.items():
                se = float(np.std([r[q] for r in reps], ddof=1))
                rows.append([name, len(idx), dep, q, round(v, 4), round(se, 4)])
    c.write_csv(c.OUT / "phase_breakdown.csv", rows, ["pieces", "positions", "depth", "quantity", "value", "se"])
    quants = list(dict.fromkeys(r[3] for r in rows))
    for dep in DEPTHS:
        print(f"\nDepth {dep}: value (bootstrap se)")
        print(f"  {'':<14}" + "".join(f"{g[0] + ' (' + str(next(r[1] for r in rows if r[0] == g[0])) + ')':>22}" for g in GROUPS + (('all', 0, 32),)))
        for q in quants:
            cells = []
            for g in GROUPS + (("all", 0, 32),):
                r = next(r for r in rows if r[0] == g[0] and r[2] == dep and r[3] == q)
                cells.append(f"{r[4]:>8.3f} ({r[5]:.3f})")
            print(f"  {q:<14}" + "".join(f"{x:>22}" for x in cells))


if __name__ == "__main__":
    main()
