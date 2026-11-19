"""Step 4: compare the engines with perfect play, in positions with <= 7 pieces.

Reads tablebase_results.csv (made by tablebase_query.py on a computer with internet access, from
tablebase_positions.txt: the 78 positions of the first Lc0 run with at most 7 pieces, 16 of which are
among the 264 paired positions):
    python 04_tablebase_check.py [path to tablebase_results.csv]
Default path: ../results/tablebase_results.csv.

With perfect play every move's value is exactly a win (1), draw (1/2) or loss (0) for the player who
makes it, so for any tolerance below 1/2 the near-best moves are exactly the moves that keep the best
result (all moves, when every move gives the same result: 0/0 := 0). The tablebase reports each move's
result for the opponent (the side to move afterwards); a win or loss that only holds beyond the
50-move rule ("cursed win", "blessed loss") counts as a draw, as in a real game.

Compares with perfect play:
  - on the 16 paired positions: both engines' near-best sets at every level, and their result-keeping
    sets read from their win/draw/loss verdicts (step 5);
  - on all 78 positions: Lc0's result-keeping sets at 1,000 nodes (first run);
Output: out/tablebase_check.csv, printed summary.
"""
import collections
import csv
import importlib
import sys

import numpy as np

import common as c

rk = importlib.import_module("05_result_keeping")

MOVER_VALUE = {"loss": 1.0, "maybe-loss": 1.0, "win": 0.0, "maybe-win": 0.0, "draw": 0.5,
               "cursed-win": 0.5, "blessed-loss": 0.5}
POSITION_VALUE = {"win": 1.0, "maybe-win": 1.0, "loss": 0.0, "maybe-loss": 0.0, "draw": 0.5,
                  "cursed-win": 0.5, "blessed-loss": 0.5}


def load_tablebase(path):
    values, verdict = collections.defaultdict(dict), {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = int(r["pos_id"])
            verdict[p] = r["position_category"]
            values[p][r["move"]] = MOVER_VALUE.get(r["category_after_move"], np.nan)
    return values, verdict


def compare_sets(sets, exact, moves, pids):
    share = np.mean([len(sets[p]) / len(moves[p]) for p in pids])
    prec = np.mean([len(sets[p] & exact[p]) / len(sets[p]) for p in pids])
    rec = np.mean([len(sets[p] & exact[p]) / len(exact[p]) for p in pids])
    return share, prec, rec


def keeping_sets(wdl, run, li, pids):
    out, verdict = {}, {}
    for p in pids:
        res = rk.results(wdl[p])[li]
        out[p] = {m for m, v in zip(run.moves[p], res) if v == res.max()}
        verdict[p] = res.max()
    return out, verdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(c.RESULTS / "tablebase_results.csv")
    tb, verdict = load_tablebase(path)
    sf, lc, lc1 = c.load_stockfish(), c.load_lc0(), c.load_lc0(c.LC0_RUN_1000)
    known, paired = set(lc1.pos_ids), set(sf.pos_ids)
    problems, usable = [], []
    for p, vals in sorted(tb.items()):
        if p not in known:
            problems.append(f"position {p} is not in the runs")
        elif set(vals) != set(lc1.moves[p]):
            problems.append(f"position {p}: tablebase moves differ from the legal moves")
        elif any(np.isnan(v) for v in vals.values()):
            problems.append(f"position {p}: a move has an unknown result")
        elif POSITION_VALUE.get(verdict[p]) != max(vals.values()):
            problems.append(f"position {p}: position verdict '{verdict[p]}' does not match its best move")
        else:
            usable.append(p)
    if problems:
        print("Problems:\n  " + "\n  ".join(problems))
    exact = {p: {m for m, v in tb[p].items() if v == max(tb[p].values())} for p in usable}
    pids = [p for p in usable if p in paired]
    ex_share = lambda ps: np.mean([len(exact[p]) / len(lc1.moves[p]) for p in ps])
    print(f"{len(usable)} positions checked ({len(pids)} among the 264 paired positions); perfect-play verdicts: "
          f"{dict(collections.Counter(verdict[p] for p in usable))}")
    print(f"exact share of result-keeping moves: {ex_share(usable):.3f} (all), {ex_share(pids):.3f} (paired)")

    rows = []
    print(f"\nPaired positions: near-best sets (e <= 0.10) and result-keeping sets against perfect play")
    print(f"{'engine and rule':<34}{'level':>7} {'share':>7} {'exact':>7} {'precision':>10} {'recall':>8} {'verdict right':>14}")
    for run in (sf, lc):
        sub = run.subset(pids)
        for li, level in enumerate(run.levels):
            for t in c.THRESHOLDS:
                sh, pr, re = compare_sets(c.near_best_sets(sub, li, t), exact, sub.moves, pids)
                rows.append([run.engine, f"e<={t}", li, level, len(pids), round(sh, 4), round(ex_share(pids), 4),
                             round(pr, 4), round(re, 4), ""])
                if t == 0.10 and li in (4, 11, 18, 24):
                    print(f"{run.engine + ' near-best e<=0.10':<34}{level:>7} {sh:>7.3f} {ex_share(pids):>7.3f} {pr:>10.2f} {re:>8.2f} {'':>14}")
    for name, zip_path, col, off, run, levels in (("Stockfish", c.SF_RUN, "depth", 1, sf, (7, 11, 15, 19, 24)),
                                                    ("Lc0", c.LC0_RUN, "level", 0, lc, (0, 12, 18, 24))):
        wdl = rk.load_wdl(zip_path, col, off)
        for li in levels:
            sets, ver = keeping_sets(wdl, run, li, pids)
            sh, pr, re = compare_sets(sets, exact, run.moves, pids)
            right = np.mean([ver[p] == max(tb[p].values()) for p in pids])
            rows.append([run.engine, "keeps result", li, run.levels[li], len(pids), round(sh, 4), round(ex_share(pids), 4),
                         round(pr, 4), round(re, 4), round(right, 4)])
            print(f"{name + ' keeps result (win/draw/loss)':<34}{run.levels[li]:>7} {sh:>7.3f} {ex_share(pids):>7.3f} {pr:>10.2f} {re:>8.2f} {right:>14.2f}")

    wdl1 = rk.load_wdl(c.LC0_RUN_1000, "level", 0)
    top1 = len(lc1.levels) - 1
    sets, ver = keeping_sets(wdl1, lc1, top1, usable)
    sh, pr, re = compare_sets(sets, exact, lc1.moves, usable)
    right = np.mean([ver[p] == max(tb[p].values()) for p in usable])
    rows.append(["Lc0 (first run)", "keeps result", top1, lc1.levels[top1], len(usable), round(sh, 4),
                 round(ex_share(usable), 4), round(pr, 4), round(re, 4), round(right, 4)])
    print(f"\nAll {len(usable)} positions, Lc0 1,000 nodes (first run), result-keeping moves: share {sh:.3f} against exact "
          f"{ex_share(usable):.3f}; precision {pr:.2f}, recall {re:.2f}; verdict right in {right:.2f} of positions")
    c.write_csv(c.OUT / "tablebase_check.csv", rows,
                ["engine", "rule", "level_index", "level", "positions", "share_engine", "share_exact", "precision",
                 "recall", "verdict_right"])


if __name__ == "__main__":
    main()
