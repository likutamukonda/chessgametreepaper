"""Step 7 (first look): the distribution of the relative error itself, not the counts under a tolerance.

For every move in every position (positions with one legal move left out), the relative error e on the
common scale, pooled over positions, binned at every Stockfish depth and Lc0 node level. By construction
each position has at least one move at e = 0 (the best) and one at e = 1 (the worst), unless all moves are
equal, when all are at 0. With perfect play every e would be exactly 0, 1/2 or 1.
Output: out/error_distribution_look.csv, printed summary for selected levels.
"""
import numpy as np

import common as c

EDGES = [0.0, 0.05, 0.10, 0.25, 0.45, 0.55, 0.75, 0.95, 1.0]
LABELS = ["e = 0", "0-0.05", "0.05-0.10", "0.10-0.25", "0.25-0.45", "0.45-0.55", "0.55-0.75", "0.75-0.95", "0.95-1", "e = 1"]


def binned(errors):
    e = np.concatenate(errors)
    out = [np.mean(e == 0.0)]
    inner = e[(e > 0) & (e < 1)]
    counts, _ = np.histogram(inner, bins=EDGES)
    out += list(counts / len(e))
    out.append(np.mean(e == 1.0))
    return out


def main():
    rows = []
    for run, picks in ((c.load_stockfish(), (0, 4, 9, 14, 19, 24)), (c.load_lc0(), (0, 6, 12, 18, 24))):
        xs = run.x()
        keep = [p for p in run.pos_ids if len(run.moves[p]) >= 2]
        for li, level in enumerate(run.levels):
            errs = [c.relative_errors(xs[p][li:li + 1])[0] for p in keep]
            per_pos_zero = np.mean([np.sum(e == 0) for e in errs])
            rows.append([run.engine, li, level] + [round(float(v), 4) for v in binned(errs)] + [round(float(per_pos_zero), 3)])
        print(f"\n{run.engine}: share of all moves in each error band (pooled over {len(keep)} positions)")
        print(f"  {'level':>7} " + " ".join(f"{l:>9}" for l in LABELS) + "   moves at e=0 per position")
        for r in rows:
            if r[0] == run.engine and r[1] in picks:
                print(f"  {r[2]:>7} " + " ".join(f"{v:>9.3f}" for v in r[3:13]) + f"   {r[13]:>6.2f}")
    c.write_csv(c.OUT / "error_distribution_look.csv", rows, ["engine", "level_index", "level"] + LABELS + ["mean_moves_at_zero"])


if __name__ == "__main__":
    main()
