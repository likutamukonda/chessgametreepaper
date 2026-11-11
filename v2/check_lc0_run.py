"""Check the output folder of v2/lc0_kaggle.ipynb for internal consistency.

Usage:
    python check_lc0_run.py <lc0_run folder> <optimal_positions.txt>

Needs the `chess` package (pip install chess). Prints a short summary and any
problems it finds; exits with an error if the files do not fit together.
"""
import collections
import csv
import json
import statistics
import sys

import chess


def main(out, positions_file):
    info = json.load(open(f"{out}/run_info.json"))
    ladder = info["ladder"]

    done = {}
    with open(f"{out}/positions_analyzed.txt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        assert header == ["pos_id", "file_index", "legal_moves", "fen"], header
        for line in fh:
            pos_id, file_index, n_legal, fen = line.rstrip("\n").split("\t")
            assert int(pos_id) not in done, f"position {pos_id} is listed twice"
            done[int(pos_id)] = (int(file_index), int(n_legal), fen)

    fens = [line.strip() for line in open(positions_file) if line.strip()]

    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    n_rows = 0
    with open(f"{out}/lc0_evals.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            pos_id, level = int(row["pos_id"]), int(row["level"])
            assert level not in rows[pos_id][row["move"]], f"duplicate row: {pos_id} {row['move']} level {level}"
            rows[pos_id][row["move"]][level] = row
    assert set(rows) == set(done), "the CSV and positions_analyzed.txt list different positions"

    problems, overshoot, mates, game_ending, searches = [], [], 0, 0, 0
    for pos_id, (file_index, n_legal, fen) in done.items():
        assert fens[file_index] == fen, f"position {pos_id}: FEN differs from line {file_index} of the positions file"
        board = chess.Board(fen)
        legal = {move.uci() for move in board.legal_moves}
        if set(rows[pos_id]) != legal or len(legal) != n_legal:
            problems.append(f"position {pos_id}: the moves are not exactly the legal moves")
        for move, levels in rows[pos_id].items():
            if sorted(levels) != list(range(len(ladder))):
                problems.append(f"position {pos_id} {move}: levels {sorted(levels)}")
                continue
            child = board.copy(stack=False)
            child.push_uci(move)
            expected = ("checkmate" if child.is_checkmate() else "stalemate" if child.is_stalemate()
                        else "insufficient_material" if child.is_insufficient_material() else "")
            terminal = levels[0]["terminal"]
            if terminal != expected:
                problems.append(f"position {pos_id} {move}: terminal '{terminal}', expected '{expected}'")
            if terminal:
                game_ending += 1
                if terminal == "checkmate" and float(levels[0]["wl"]) != 1.0:
                    problems.append(f"position {pos_id} {move}: checkmate without wl = 1")
                continue
            searches += 1
            previous = 0
            for level in range(len(ladder)):
                row = levels[level]
                assert int(row["target_nodes"]) == ladder[level]
                nodes, wl = int(row["nodes"]), float(row["wl"])
                if nodes < previous:
                    problems.append(f"position {pos_id} {move}: node count falls at level {level}")
                if not -1.0 <= wl <= 1.0:
                    problems.append(f"position {pos_id} {move}: wl = {wl}")
                if row["w"] != "":
                    w, d, l = int(row["w"]), int(row["d"]), int(row["l"])
                    if w + d + l != 1000:
                        problems.append(f"position {pos_id} {move}: w + d + l = {w + d + l}")
                    if row["mate"] == "" and abs((w - l) / 1000 - wl) > 0.0021:
                        problems.append(f"position {pos_id} {move} level {level}: wl {wl} but w - l = {(w - l) / 1000}")
                if row["mate"] != "":
                    mates += 1
                previous = nodes
            overshoot.append(int(levels[len(ladder) - 1]["nodes"]) / ladder[-1])

    expected_rows = sum(n for _, n, _ in done.values()) * len(ladder)
    print(f"positions finished: {len(done)}")
    print(f"rows: {n_rows:,} (expected {expected_rows:,})")
    print(f"move searches: {searches:,}; game-ending moves: {game_ending}; rows with a proven mate: {mates}")
    print(f"ladder: {len(ladder)} levels up to {ladder[-1]:,} nodes")
    if overshoot:
        print(f"nodes reached at the top level, relative to the target: median {statistics.median(overshoot):.2f}, "
              f"largest {max(overshoot):.2f}")

    order = list(csv.DictReader(open(f"{out}/positions_order.csv")))
    assert all(fens[int(r["file_index"])] == r["fen"] for r in order[:300]), "positions_order.csv does not match the positions file"
    assert [int(r["pos_id"]) for r in order[:50]] == list(range(min(50, len(order))))
    if n_rows != expected_rows:
        problems.append(f"the CSV has {n_rows:,} rows but {expected_rows:,} were expected")

    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems[:20]:
            print("  " + problem)
    else:
        print("no problems found")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
