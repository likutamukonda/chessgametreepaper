"""6. The sample, and the distribution by game phase (Stockfish, near-ties counted as ties).

The sample: number of positions and moves, pieces on the board and legal moves per position.
Positions are grouped by the number of pieces on the board, kings and pawns included: 12 or fewer (endgames),
13-22 (middlegames) and 23 or more (openings and early middlegames, after the book). For each group, at depths 5 and
25: number of positions, mean number of legal moves, the five parameters of the zero-one-inflated logit-normal and
the observed share of moves within 0.05, 0.10, 0.25 and 0.5, with bootstrap standard errors (200 resamples of the
positions within each group; for all positions together, the resamples used everywhere else).
Outputs: out/sample.csv, out/phase_breakdown.csv
"""
import numpy as np

import common as c
import model as mdl

DEPTHS = (5, 25)


def main():
    S = mdl.setup()
    run, xs, per_depth = S["run"], S["xs"], S["per_depth"]
    pieces = np.array([c.piece_count(run.fens[p]) for p in run.pos_ids])
    n_legal = np.array([xs[p].shape[1] for p in run.pos_ids])
    sample = [["positions", len(pieces)], ["legal moves in all", int(n_legal.sum())],
              ["pieces, median", float(np.median(pieces))], ["pieces, fewest", int(pieces.min())],
              ["pieces, most", int(pieces.max())], ["legal moves, mean", round(float(n_legal.mean()), 3)],
              ["legal moves, fewest", int(n_legal.min())], ["legal moves, most", int(n_legal.max())]]
    c.write_csv(c.OUT / "sample.csv", sample, ["quantity", "value"])
    print("Sample: " + ", ".join(f"{k} {v}" for k, v in sample))
    rng = np.random.default_rng(mdl.SEED)
    rows = []
    for name, lo, hi in mdl.PHASES + (("all", 0, 32),):
        idx = np.flatnonzero((pieces >= lo) & (pieces <= hi))
        boots = S["boots"] if name == "all" else [rng.choice(idx, len(idx), replace=True) for _ in range(mdl.B)]
        for dep in DEPTHS:
            li = run.levels.index(dep)
            full = mdl.phase_summary([per_depth[li][i] for i in idx], n_legal[idx])
            reps = [mdl.phase_summary([per_depth[li][i] for i in b], n_legal[b]) for b in boots]
            for q, v in full.items():
                rows.append([name, len(idx), dep, q, round(v, 6), round(float(np.std([r[q] for r in reps], ddof=1)), 6)])
    c.write_csv(c.OUT / "phase_breakdown.csv", rows, ["pieces", "positions", "depth", "quantity", "value", "se"])
    quants = list(dict.fromkeys(r[3] for r in rows))
    groups = [g[0] for g in mdl.PHASES] + ["all"]
    for dep in DEPTHS:
        print(f"\nDepth {dep}: value (bootstrap se)")
        print(f"  {'':<14}" + "".join(f"{g + ' (' + str(next(r[1] for r in rows if r[0] == g)) + ')':>22}" for g in groups))
        for q in quants:
            cells = [next(f"{r[4]:>8.3f} ({r[5]:.3f})" for r in rows if r[0] == g and r[2] == dep and r[3] == q) for g in groups]
            print(f"  {q:<14}" + "".join(f"{x:>22}" for x in cells))


if __name__ == "__main__":
    main()
