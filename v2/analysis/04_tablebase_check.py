"""Step 4: check the infinite-depth picture against perfect play, in the positions with <= 7 pieces.

Reads tablebase_results.csv (made by tablebase_query.py on a computer with internet access):
    python 04_tablebase_check.py [path to tablebase_results.csv]
Default path: ../results/tablebase_results.csv.

With perfect play every move's value is exactly a win (1), draw (1/2) or loss (0) for the player
who makes it, so for any tolerance below 1/2 the near-best moves are exactly the moves that keep
the best result (all moves, when every move gives the same result: 0/0 := 0). The tablebase reports
each move's result for the opponent (the side to move afterwards); a win or loss that only holds
beyond the 50-move rule ("cursed win", "blessed loss") counts as a draw, as in a real game.

Compares, on these positions, the share of near-best moves at every Stockfish depth and Lc0 node
level with the exact share, the overlap of the near-best sets with the exact set, and whether the
depth trend fitted to these positions points at the exact value.
Output: out/tablebase_check.csv (per level), printed summary.
"""
import collections
import csv
import importlib
import sys

import numpy as np

import common as c

trends = importlib.import_module("03_depth_trends")

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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(c.RESULTS / "tablebase_results.csv")
    tb, verdict = load_tablebase(path)
    sf, lc = c.load_stockfish(), c.load_lc0()
    problems = []
    for p, vals in tb.items():
        if set(vals) != set(sf.moves[p]):
            problems.append(f"position {p}: tablebase moves differ from the legal moves")
        if any(np.isnan(v) for v in vals.values()):
            problems.append(f"position {p}: a move has an unknown result")
        elif POSITION_VALUE.get(verdict[p]) != max(vals.values()):
            problems.append(f"position {p}: position verdict '{verdict[p]}' does not match its best move")
    if problems:
        print("Problems:\n  " + "\n  ".join(problems))
    pids = [p for p in sorted(tb) if not any(np.isnan(v) for v in tb[p].values())]
    exact = {}
    for p in pids:
        best = max(tb[p].values())
        exact[p] = {m for m, v in tb[p].items() if v == best}       # also covers "all moves equal"
    exact_share = np.array([len(exact[p]) / len(sf.moves[p]) for p in pids])
    counts = collections.Counter(verdict[p] for p in pids)
    print(f"{len(pids)} positions; perfect-play verdicts for the side to move: {dict(counts)}")
    print(f"exact share of near-best moves (= result-keeping moves): {exact_share.mean():.3f}")

    rows = []
    print("\nNear-best at e <= 0.10 compared with perfect play, on these positions")
    print(f"{'engine':<10}{'level':>7} {'share':>7} {'exact':>7} {'best move keeps result':>23} {'precision':>10} {'recall':>8}")
    for run in (sf, lc):
        sub = run.subset(pids)
        for li, level in enumerate(run.levels):
            for t in c.THRESHOLDS:
                sets = c.near_best_sets(sub, li, t)
                tops = c.near_best_sets(sub, li, 0.0)
                share = np.mean([len(sets[p]) / len(sub.moves[p]) for p in pids])
                keeps = np.mean([bool(tops[p] & exact[p]) for p in pids])
                prec = np.mean([len(sets[p] & exact[p]) / len(sets[p]) for p in pids])
                rec = np.mean([len(sets[p] & exact[p]) / len(exact[p]) for p in pids])
                rows.append([run.engine, li, level, t, round(share, 4), round(float(exact_share.mean()), 4),
                             round(keeps, 4), round(prec, 4), round(rec, 4)])
                if t == 0.10 and (li in (0, 4, 9, 14, 19, 24)):
                    print(f"{run.engine:<10}{level:>7} {share:>7.3f} {exact_share.mean():>7.3f} {keeps:>23.2f} {prec:>10.2f} {rec:>8.2f}")
    c.write_csv(c.OUT / "tablebase_check.csv", rows,
                ["engine", "level_index", "level", "threshold", "share_engine", "share_exact",
                 "best_move_keeps_result", "precision", "recall"])

    # Does the Stockfish depth trend, fitted on these positions only, point at the exact share?
    print("\nStockfish trend fitted on these positions (depths 5-25), extrapolated, against the exact share")
    eng = trends.Engine(sf.subset(pids))
    q = eng.quantities(np.arange(len(pids)), ("mean_count",))
    for ti, t in enumerate(c.THRESHOLDS):
        y = q["mean_share"][:, ti]
        out = []
        for form in ("constant", "linear", "exponential", "power"):
            params, lim, ident = trends.fit_form(form, eng.x[eng.use], y[eng.use])
            out.append(f"{form} {'-' if lim is None or not ident else f'{lim:.3f}'}")
        print(f"  e<={t}: depth 25 {y[-1]:.3f} | limits: " + ", ".join(out) + f" | exact {exact_share.mean():.3f}")


if __name__ == "__main__":
    main()
