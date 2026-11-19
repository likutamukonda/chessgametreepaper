"""Step 5: the infinite-depth limit read directly, as the share of moves that keep the result.

With perfect play every move's value is exactly a win, draw or loss, so the relative error of every
move is 0, 1/2 or 1 (or 0 for all when all moves give the same result). For every tolerance below
1/2 the near-best moves are therefore the same set: the moves that keep the position's best result.
The smooth depth trends of step 3 cannot reach this limit for Stockfish (its share is still rising
at depth 25), so here the limit is estimated directly: each move's result is read from the engine's
own win/draw/loss verdict (the most likely of the three; a majority rule is the robustness check),
and a position's share is the fraction of moves whose result equals the best one.

Reported for every Stockfish depth and Lc0 node level, by the verdict for the side to move, and as
agreement between the engines at the top level. The tablebase step (04) checks the verdicts against
perfect play where that is known.
Output: out/result_keeping.csv
"""
import collections
import csv
import io
import zipfile

import numpy as np

import common as c

RESULT = {0: 1.0, 1: 0.5, 2: 0.0}            # index of the largest of (w, d, l) -> result for the mover
NAME = {1.0: "win", 0.5: "draw", 0.0: "loss"}


def load_wdl(zip_path, level_col, offset):
    """pos_id -> array (levels, moves, 3) of (w, d, l) in thousandths, moves sorted as in common.Run."""
    raw = collections.defaultdict(lambda: collections.defaultdict(dict))
    with zipfile.ZipFile(zip_path) as zf:
        for r in csv.DictReader(io.TextIOWrapper(zf.open("lc0_evals.csv" if "lc0" in zip_path.name else "stockfish_evals.csv"),
                                                 encoding="utf-8", newline="")):
            raw[int(r["pos_id"])][int(r[level_col]) - offset][r["move"]] = (int(r["w"]), int(r["d"]), int(r["l"]))
    out = {}
    for p, levels in raw.items():
        moves = sorted(levels[0])
        out[p] = np.array([[levels[k][m] for m in moves] for k in range(len(levels))], dtype=float)
    return out


def results(wdl, rule="argmax"):
    """array (levels, moves, 3) -> array (levels, moves) of 1 / 0.5 / 0 for the player who moved."""
    if rule == "argmax":
        return np.vectorize(RESULT.get)(np.argmax(wdl, axis=2))
    return np.where(wdl[:, :, 0] >= 500, 1.0, np.where(wdl[:, :, 2] >= 500, 0.0, 0.5))


def keep_counts(res):
    """array (levels, moves) of results -> number of moves keeping the best result, and that result."""
    best = res.max(axis=1)
    return (res == best[:, None]).sum(axis=1), best


def main():
    engines = {"Stockfish": (load_wdl(c.SF_RUN, "depth", 1), c.load_stockfish()),
               "Lc0": (load_wdl(c.LC0_RUN, "level", 0), c.load_lc0())}
    rows, top = [], {}
    for eng, (wdl, run) in engines.items():
        n = np.array([len(run.moves[p]) for p in run.pos_ids])
        for rule in ("argmax", "majority"):
            kc = np.array([keep_counts(results(wdl[p], rule))[0] for p in run.pos_ids])      # (positions, levels)
            verdict = np.array([keep_counts(results(wdl[p], rule))[1] for p in run.pos_ids])
            for li, level in enumerate(run.levels):
                share = kc[:, li] / n
                m, se = c.mean_se(share)
                by = {NAME[v]: (int((verdict[:, li] == v).sum()),
                                float(share[verdict[:, li] == v].mean()) if (verdict[:, li] == v).any() else float("nan"))
                      for v in (1.0, 0.5, 0.0)}
                rows.append([eng, rule, li, level, round(m, 4), round(se, 4), round(float(kc[:, li].mean()), 3)]
                            + [x for v in ("win", "draw", "loss") for x in (by[v][0], round(by[v][1], 4))])
            if rule == "argmax":
                top[eng] = (kc[:, -1], verdict[:, -1])
    c.write_csv(c.OUT / "result_keeping.csv", rows,
                ["engine", "rule", "level_index", "level", "share_keeping", "se", "mean_count_keeping",
                 "positions_won", "share_won", "positions_drawn", "share_drawn", "positions_lost", "share_lost"])

    print("Share of legal moves that keep the best result (results from each engine's own win/draw/loss)")
    for eng in engines:
        sel = [r for r in rows if r[0] == eng and r[1] == "argmax"]
        picks = sel[::4] + ([sel[-1]] if (len(sel) - 1) % 4 else [])
        print(f"  {eng:<9} " + "  ".join(f"{r[3]}: {r[4]:.3f}" for r in picks))
    for eng in engines:
        r = [x for x in rows if x[0] == eng and x[1] == "argmax"][-1]
        rm = [x for x in rows if x[0] == eng and x[1] == "majority"][-1]
        print(f"  {eng} top level: {r[4]:.3f} ± {r[5]:.3f} (majority rule {rm[4]:.3f}); mean count {r[6]:.1f}; "
              f"won {r[7]} positions share {r[8]:.2f}, drawn {r[9]} share {r[10]:.2f}, lost {r[11]} share {r[12]:.2f}")
    (ks, vs), (kl, vl) = top["Stockfish"], top["Lc0"]
    print(f"  Stockfish d25 vs Lc0 10k: same verdict for the side to move in {np.mean(vs == vl):.2f} of positions; "
          f"correlation of counts {np.corrcoef(ks, kl)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
