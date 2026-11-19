"""Step 9 (check): is about half of the legal moves within relative error 1/2, and is that robust?

e <= 1/2 means a move's value is at least as close to the best move's value as to the worst move's.
Reports the average share with e <= 0.5 (and its spread across positions) for several value scales and
for selected Stockfish depths and Lc0 node levels.
Output: out/half_tolerance_check.csv, printed summary.
"""
import numpy as np

import common as c

SCALES = [("common, Lichess steepness", 1.0, False), ("common, half as steep", 0.5, False),
          ("common, twice as steep", 2.0, False), ("engine's own win/draw/loss scale", None, True)]


def share_half(run, li, steepness, own):
    xs = run.x(steepness or 1.0, own)
    return np.array([np.mean(c.relative_errors(xs[p][li:li + 1])[0] <= 0.5 + 1e-12) for p in run.pos_ids])


def main():
    rows = []
    for run, picks in ((c.load_stockfish(), (0, 12, 24)), (c.load_lc0(), (0, 12, 24))):
        for name, steep, own in SCALES:
            for li in picks:
                s = share_half(run, li, steep, own)
                m, se = c.mean_se(s)
                q1, med, q3 = np.percentile(s, [25, 50, 75])
                rows.append([run.engine, run.levels[li], name, round(m, 4), round(se, 4), round(med, 4), round(q1, 4), round(q3, 4)])
    # centimetre-style raw centipawns as a further scale (clipped at +-1000, mates at the clip)
    for run in (c.load_stockfish(), c.load_lc0()):
        for li in (0, 12, 24):
            vals = []
            for p in run.pos_ids:
                cp = np.clip(run.cp[p][li:li + 1], -1000, 1000)
                vals.append(np.mean(c.relative_errors(cp)[0] <= 0.5 + 1e-12))
            m, se = c.mean_se(vals)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append([run.engine, run.levels[li], "centipawns clipped at +-1000", round(m, 4), round(se, 4),
                         round(med, 4), round(q1, 4), round(q3, 4)])
    c.write_csv(c.OUT / "half_tolerance_check.csv", rows,
                ["engine", "level", "scale", "mean_share_e_le_half", "se", "median", "q25", "q75"])
    print("Share of legal moves with e <= 0.5: average ± s.e. [middle half of positions]")
    for r in rows:
        print(f"  {r[0]:<10}{r[1]:>6}  {r[2]:<34} {r[3]:.3f} ± {r[4]:.3f}  [{r[6]:.2f}, {r[7]:.2f}]")


if __name__ == "__main__":
    main()
