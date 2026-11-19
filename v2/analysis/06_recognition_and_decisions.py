"""Step 6 (checks for the modelling discussion): do the engines change their decisions, and are the
near-best moves at finite depth a subset of the moves that keep the result?

1. Recognition. Reference set R = moves that keep the best result according to Stockfish's
   win/draw/loss verdicts at depth 25 (step 5). For each Stockfish depth and Lc0 node level and each
   tolerance: precision = share of near-best moves that are in R; recognition = share of R that is
   near-best. If precision stays high and recognition climbs with depth, the near-best set at depth d
   is a growing subset of R, which must reach all of R at infinite depth.
2. Decisions. How often the best move (any move tied for best) at a level is also best at the top level,
   the overlap of the e <= 0.10 sets with the top level, and the correlation of the counts.
3. Optimism. The average expected score of the best move (common scale) and how often the verdict for
   the side to move (win / draw / loss) differs from the top level.
Output: out/recognition_and_decisions.csv, printed summary.
"""
import importlib

import numpy as np

import common as c

rk = importlib.import_module("05_result_keeping")


def main():
    sf, lc = c.load_stockfish(), c.load_lc0()
    pids = sf.pos_ids
    wdl_sf, wdl_lc = rk.load_wdl(c.SF_RUN, "depth", 1), rk.load_wdl(c.LC0_RUN, "level", 0)
    ref = {}
    for p in pids:
        res = rk.results(wdl_sf[p])[-1]
        ref[p] = {m for m, v in zip(sf.moves[p], res) if v == res.max()}
    rows = []
    print("1. Near-best moves against the result-keeping moves R (Stockfish depth 25 verdicts)")
    print(f"   {'level':<22}" + "".join(f"{'e<=' + str(t) + ' prec / recog':>24}" for t in c.THRESHOLDS))
    for run, wdl, picks in ((sf, wdl_sf, (4, 9, 14, 19, 24)), (lc, wdl_lc, (0, 12, 18, 24))):
        top = len(run.levels) - 1
        x = run.x()
        top_sets = c.near_best_sets(run, top, 0.10)
        top_best = c.near_best_sets(run, top, 0.0)
        for li in range(len(run.levels)):
            cells = []
            for t in c.THRESHOLDS:
                sets = c.near_best_sets(run, li, t)
                prec = np.mean([len(sets[p] & ref[p]) / len(sets[p]) for p in pids])
                recog = np.mean([len(sets[p] & ref[p]) / len(ref[p]) for p in pids])
                cells.append((prec, recog))
            best = c.near_best_sets(run, li, 0.0)
            same_best = np.mean([bool(best[p] & top_best[p]) for p in pids])
            s10 = c.near_best_sets(run, li, 0.10)
            jac = np.mean([len(s10[p] & top_sets[p]) / len(s10[p] | top_sets[p]) for p in pids])
            cnt = np.array([len(s10[p]) for p in pids]); cnt_top = np.array([len(top_sets[p]) for p in pids])
            corr = np.corrcoef(cnt, cnt_top)[0, 1] if li != top else 1.0
            best_x = np.mean([x[p][li].max() for p in pids])
            verdict = np.mean([rk.results(wdl[p])[li].max() != rk.results(wdl[p])[top].max() for p in pids])
            rows.append([run.engine, li, run.levels[li]] + [round(v, 4) for pr in cells for v in pr]
                        + [round(same_best, 4), round(jac, 4), round(corr, 4), round(best_x, 4), round(verdict, 4)])
            if li in picks:
                name = f"{run.engine} {'depth ' if run.engine == 'Stockfish' else ''}{run.levels[li]}{'' if run.engine == 'Stockfish' else ' nodes'}"
                print(f"   {name:<22}" + "".join(f"{pr:>15.2f} / {rc:.2f}" for pr, rc in cells))
    c.write_csv(c.OUT / "recognition_and_decisions.csv", rows,
                ["engine", "level_index", "level"] + [f"{k}_e{t}" for t in c.THRESHOLDS for k in ("precision", "recognition")]
                + ["best_move_same_as_top", "jaccard_e0.10_with_top", "corr_counts_with_top", "mean_best_expected_score",
                   "verdict_differs_from_top"])
    print("\n2-3. Decisions and optimism, compared with each engine's own top level")
    print(f"   {'level':<22}{'best move same':>15}{'overlap e<=0.10':>17}{'corr counts':>13}{'mean best x':>13}{'verdict differs':>17}")
    for r in rows:
        if (r[0] == "Stockfish" and r[1] in (0, 1, 2, 4, 9, 14, 19, 24)) or (r[0] == "Lc0" and r[1] in (0, 6, 12, 18, 24)):
            name = f"{r[0]} {'depth ' if r[0] == 'Stockfish' else ''}{r[2]}{'' if r[0] == 'Stockfish' else ' nodes'}"
            print(f"   {name:<22}{r[9]:>15.2f}{r[10]:>17.2f}{r[11]:>13.2f}{r[12]:>13.3f}{r[13]:>17.2f}")


if __name__ == "__main__":
    main()
