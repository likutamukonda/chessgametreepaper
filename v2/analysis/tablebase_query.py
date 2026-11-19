"""Look up the exact result of every legal move in the small positions, using the Lichess tablebase.

Run this on a computer with internet access (it only needs Python; no extra packages):

    python tablebase_query.py

It looks for positions_analyzed.txt next to this file (or in a folder below it, such as lc0_run), or
you can name the file:  python tablebase_query.py --positions "C:\\path\\to\\positions_analyzed.txt"

For each position with at most 7 pieces it asks https://tablebase.lichess.ovh for the perfect-play
result of every legal move (about one request per second) and writes, next to this file:

    tablebase_results.csv   one row per position and move
    tablebase_raw.jsonl     the server's full answers, one line per position

Send tablebase_results.csv back for the comparison step (04_tablebase_check.py).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://tablebase.lichess.ovh/standard?fen="
MAX_PIECES = 7


def find_positions(given):
    if given:
        if not os.path.isfile(given):
            sys.exit(f"The position list was not found at {given}")
        return given
    for folder in (HERE, os.getcwd()):
        for pattern in ("positions_analyzed.txt", os.path.join("*", "positions_analyzed.txt"),
                        os.path.join("*", "*", "positions_analyzed.txt")):
            hits = sorted(glob.glob(os.path.join(folder, pattern)))
            if hits:
                return hits[0]
    sys.exit("positions_analyzed.txt was not found. Put the lc0_run folder next to this file, or use --positions")


def small_positions(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            fen = parts[3]
            board, castling = fen.split()[0], fen.split()[2]
            if sum(ch.isalpha() for ch in board) <= MAX_PIECES and castling == "-":
                out.append((int(parts[0]), fen))
    return out


def ask(fen):
    if len(fen.split()) == 4:                        # the position lists leave out the move counters
        fen += " 0 1"
    url = API + urllib.parse.quote(fen.replace(" ", "_"), safe="/_-")
    request = urllib.request.Request(url, headers={"User-Agent": "chess-game-tree-research (tablebase_query.py)"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 429:                      # asked to slow down
                print("  the server asked us to slow down; waiting 60 s")
                time.sleep(60)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    sys.exit(f"No answer from the tablebase server for {fen}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", default="", help="path to positions_analyzed.txt")
    args = parser.parse_args()
    positions = small_positions(find_positions(args.positions))
    print(f"{len(positions)} positions with at most {MAX_PIECES} pieces")
    rows = []
    with open(os.path.join(HERE, "tablebase_raw.jsonl"), "w", encoding="utf-8") as raw:
        for i, (pos_id, fen) in enumerate(positions, 1):
            answer = ask(fen)
            raw.write(json.dumps({"pos_id": pos_id, "fen": fen, "answer": answer}) + "\n")
            for m in answer.get("moves", []):
                rows.append([pos_id, fen, answer.get("category"), m.get("uci"), m.get("category"), m.get("dtz"),
                             m.get("dtm"), m.get("checkmate"), m.get("stalemate"), m.get("insufficient_material")])
            print(f"  [{i}/{len(positions)}] position {pos_id}: {answer.get('category')}, {len(answer.get('moves', []))} moves")
            time.sleep(1.0)
    out = os.path.join(HERE, "tablebase_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pos_id", "fen", "position_category", "move", "category_after_move", "dtz", "dtm",
                    "checkmate", "stalemate", "insufficient_material"])
        w.writerows(rows)
    print(f"Done. Send this file back: {out}")


if __name__ == "__main__":
    main()
