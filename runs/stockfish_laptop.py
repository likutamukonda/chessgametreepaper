"""Stockfish evaluation of every legal move, for the positions Lc0 finished.

Reads    positions_analyzed.txt   (from the lc0_run folder made on Kaggle)
Writes   stockfish_run/stockfish_evals.csv    one row per position, move and depth
         stockfish_run/positions_done.txt     the positions that are finished
         stockfish_run/run_info.json          settings and timings
         stockfish_run.zip                    the folder, zipped, when the run stops

How it works
  * One search per position, to DEPTH, with every legal move scored
    (MultiPV = number of legal moves).  Stockfish deepens one step at a time,
    so the scores at depths 1, 2, ... DEPTH all come from that one search.
  * Each engine uses one thread and starts every position with an empty hash
    table, so a position gives the same numbers on any computer and in any
    order.  Several engines run side by side to use the processor's cores.
  * Results are saved after every position.  Stop with Ctrl+C at any time and
    run the same command again later: finished positions are skipped.

Columns of stockfish_evals.csv
  pos_id   the position's number, the same as in the Lc0 files
  move     the legal move, in UCI notation
  depth    search depth (1 ... DEPTH)
  cp       score in centipawns for the side to move, after that move
  mate     filled in instead of cp when a forced mate is seen
           (positive: the side to move mates in that many moves)
  w, d, l  Stockfish's win, draw and loss chances in thousandths, for the side to move
  nodes, secs   the engine's node count and clock when that depth was finished

Set-up (once)
  1. pip install chess
  2. Download Stockfish 19 for Windows and unzip it next to this file:
     https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip
  3. Put the lc0_run folder (or just positions_analyzed.txt) next to this file.

Run
  python stockfish_laptop.py --max-positions 10     a short trial with timings
  python stockfish_laptop.py                        the full run (continues a stopped one)
"""
import argparse
import collections
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

try:
    import chess
    import chess.engine
except ImportError:
    sys.exit("The 'chess' package is missing. Install it with:  pip install chess")

# ---- Settings (each can also be given on the command line) ---------------------
DEPTH = 25                    # deepest search depth; every depth up to it is recorded
WORKERS = 0                   # engines running side by side; 0 = logical processors minus 4
HASH_MB = 256                 # hash memory per engine, in MB (keep the same for a whole run)
MAX_MINUTES_PER_POSITION = 0  # 0 = no limit; otherwise a position stops early and keeps the depths it finished
OUT_DIR = "stockfish_run"

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_HEADER = ["pos_id", "move", "depth", "cp", "mate", "w", "d", "l", "nodes", "secs"]
DONE_HEADER = "pos_id\tlegal_moves\tdepth_reached\tsecs\tfen\n"


def say(*parts):
    print(*parts, flush=True)


def clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ---- Finding the engine and the position list ---------------------------------------
def find_stockfish(given):
    if given:
        if not os.path.isfile(given):
            sys.exit(f"Stockfish was not found at {given}")
        return given
    found = []
    for folder in (HERE, os.getcwd()):
        for pattern in ("stockfish*", os.path.join("*", "stockfish*"), os.path.join("*", "*", "stockfish*")):
            for path in glob.glob(os.path.join(folder, pattern)):
                name = os.path.basename(path).lower()
                if not os.path.isfile(path) or name.endswith((".py", ".zip", ".txt", ".csv", ".json", ".md", ".gz", ".tar")):
                    continue
                if os.name == "nt" and not name.endswith(".exe"):
                    continue
                if os.name != "nt" and not os.access(path, os.X_OK):
                    continue
                found.append(path)
    if found:
        return sorted(set(found))[0]
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    sys.exit("Stockfish was not found. Unzip it next to this file, or give its location with\n"
             "  --stockfish \"C:\\path\\to\\stockfish-windows-x86-64-universal.exe\"")


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
    sys.exit("positions_analyzed.txt was not found. Put the lc0_run folder next to this file, or give the file with\n"
             "  --positions \"C:\\path\\to\\positions_analyzed.txt\"")


def read_positions(path):
    positions = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or not parts[0].isdigit():
                continue                       # header or blank line
            positions.append({"pos_id": int(parts[0]), "n_legal": int(parts[2]), "fen": parts[3]})
    positions.sort(key=lambda p: p["pos_id"])
    if not positions:
        sys.exit(f"No positions were found in {path}")
    return positions


# ---- Engine -------------------------------------------------------------------------------
class Engine:
    def __init__(self, path, hash_mb):
        self.path, self.hash_mb = path, hash_mb
        self.engine, self.name = None, ""
        self.start()

    def start(self):
        extra = {}
        if os.name == "nt":                    # lower priority keeps the computer usable meanwhile
            extra["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        # setpgrp: Ctrl+C does not reach the engines directly, so this script
        # can shut them down in an orderly way.
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=120, setpgrp=True, **extra)
        options = {"Threads": 1, "Hash": self.hash_mb, "UCI_ShowWDL": True}
        if "NumaPolicy" in self.engine.options:
            options["NumaPolicy"] = "none"     # many one-thread engines: never pin them to a core
        self.engine.configure(options)
        self.name = self.engine.id.get("name", "")

    def restart(self):
        self.close()
        self.start()

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass


class Stopped(Exception):
    """Raised inside a worker when the run is being stopped."""


def analyse_position(eng, pos, depth, max_seconds, stop):
    """One search to `depth`; returns the rows of every completed depth."""
    board = chess.Board(pos["fen"])
    mover = board.turn
    n = board.legal_moves.count()
    if n != pos["n_legal"]:
        raise RuntimeError(f"position {pos['pos_id']}: {n} legal moves here but {pos['n_legal']} in the list")
    limit = chess.engine.Limit(depth=depth, time=max_seconds or None)
    table = collections.defaultdict(dict)      # depth -> move -> (cp, mate, w, d, l)
    clock_at = {}                              # depth -> (nodes, secs) when that depth was finished
    deepest = 0
    started = time.perf_counter()
    # A new `game` makes Stockfish clear its hash table, so the result does not
    # depend on what this engine searched before.
    with eng.engine.analysis(board, limit, multipv=n, game=("position", pos["pos_id"], started)) as analysis:
        for info in analysis:
            if stop.is_set():
                raise Stopped()
            if "pv" not in info or "score" not in info or "depth" not in info:
                continue
            if info.get("lowerbound") or info.get("upperbound"):
                continue                       # not an exact score
            d = info["depth"]
            if d > depth:
                continue
            score = info["score"].pov(mover)   # from the side to move's point of view
            cp, mate = ("", score.mate()) if score.is_mate() else (score.score(), "")
            w = dr = l = ""
            wdl = info.get("wdl")
            if wdl is not None:
                p = wdl.pov(mover)
                w, dr, l = p.wins, p.draws, p.losses
            # While Stockfish works through the moves of a depth it can show a
            # provisional score for a move before its final one, so the LAST
            # report for a (depth, move) is the one that is kept.
            table[d][info["pv"][0].uci()] = (cp, mate, w, dr, l)
            # Lines of the previous depth are repeated while the next depth is
            # searched; only lines of the depth in progress move that depth's clock.
            if d >= deepest:
                deepest = d
                secs = info.get("time")
                clock_at[d] = (info.get("nodes", ""), "" if secs is None else round(secs, 3))
    elapsed = time.perf_counter() - started
    if stop.is_set():
        raise Stopped()
    # A time limit can cut the search off inside a depth, leaving provisional
    # scores there.  That depth is dropped unless the search clearly ended by itself.
    last_ok = deepest
    if max_seconds and not (deepest == depth and elapsed < max_seconds - 1.0):
        last_ok = deepest - 1
    rows, reached, depth_secs = [], 0, {}
    for d in range(1, last_ok + 1):
        if len(table.get(d, {})) != n:
            break                              # this depth was not finished
        reached = d
        nodes, secs = clock_at.get(d, ("", ""))
        for move in sorted(table[d]):
            rows.append([pos["pos_id"], move, d, *table[d][move], nodes, secs])
        depth_secs[d] = secs if secs != "" else 0.0
    if reached == 0:
        raise RuntimeError(f"position {pos['pos_id']}: Stockfish returned no complete depth")
    return rows, reached, depth_secs, elapsed


# ---- Keeping Windows awake -------------------------------------------------------------------
def keep_awake(on):
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


# ---- The run --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stockfish evaluation of every legal move, at every depth.")
    parser.add_argument("--stockfish", default="", help="path to the Stockfish program (default: look next to this file)")
    parser.add_argument("--positions", default="", help="path to positions_analyzed.txt (default: look next to this file)")
    parser.add_argument("--out", default=OUT_DIR, help="output folder (default: stockfish_run)")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--workers", type=int, default=WORKERS, help="engines side by side (default: logical processors minus 4)")
    parser.add_argument("--hash", type=int, default=HASH_MB, help="hash memory per engine in MB")
    parser.add_argument("--max-positions", type=int, default=0, help="stop once this many positions are finished in total")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES_PER_POSITION,
                        help="time limit per position in minutes (0 = none)")
    args = parser.parse_args()

    stockfish = find_stockfish(args.stockfish)
    positions_file = find_positions(args.positions)
    positions = read_positions(positions_file)
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "stockfish_evals.csv")
    done_path = os.path.join(out, "positions_done.txt")
    info_path = os.path.join(out, "run_info.json")
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 4)
    max_seconds = args.max_minutes * 60 if args.max_minutes > 0 else 0

    try:
        first = Engine(stockfish, args.hash)
    except Exception as exc:
        sys.exit(f"Stockfish could not be started from {stockfish}\n  {exc!r}\n"
                 "Check that this is the Stockfish program itself, or point to it with --stockfish.")
    say(f"Engine:    {first.name}  ({stockfish})")
    say(f"Positions: {len(positions)} in {positions_file}")
    say(f"Settings:  depth {args.depth}, {workers} engine(s) with 1 thread and {args.hash} MB hash each"
        + (f", at most {args.max_minutes:g} min per position" if max_seconds else ""))

    # ---- Continue an earlier run (results already in the folder are never overwritten) ----
    done = {}
    old = {}
    if os.path.exists(info_path):
        with open(info_path) as fh:
            old = json.load(fh)
        for key, now in (("engine", first.name), ("hash_mb", args.hash)):
            if key in old and old[key] != now:
                first.close()
                sys.exit(f"The folder {out} holds results made with {key} = {old[key]!r}, but this run has {now!r}.\n"
                         "Use the same setting, or a new folder with --out.")
    if os.path.exists(done_path) != os.path.exists(csv_path):
        first.close()
        sys.exit(f"The folder {out} holds only one of stockfish_evals.csv and positions_done.txt.\n"
                 "Move it aside or use a new folder with --out.")
    if os.path.exists(done_path):
        good_lines = []
        with open(done_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                # A line cut off by a crash is ignored; its position is simply done again.
                if len(parts) == 5 and line.endswith("\n") and parts[0].isdigit() and parts[2].isdigit():
                    done[int(parts[0])] = int(parts[2])
                    good_lines.append(line)
        tmp = done_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(DONE_HEADER)
            fh.writelines(good_lines)
        os.replace(tmp, done_path)
        kept = dropped = 0
        tmp = csv_path + ".tmp"
        with open(csv_path, newline="") as src, open(tmp, "w", newline="") as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            next(reader, None)
            writer.writerow(CSV_HEADER)
            for row in reader:
                if len(row) == len(CSV_HEADER) and row[0].isdigit() and int(row[0]) in done:
                    writer.writerow(row)
                    kept += 1
                else:
                    dropped += 1
        os.replace(tmp, csv_path)
        say(f"Continuing: {len(done)} positions already finished ({kept:,} rows kept, {dropped:,} unfinished rows dropped).")
    else:
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_HEADER)
        with open(done_path, "w", encoding="utf-8") as fh:
            fh.write(DONE_HEADER)

    todo = [p for p in positions if p["pos_id"] not in done]
    if args.max_positions > 0:
        todo = todo[:max(0, args.max_positions - len(done))]
    if not todo:
        first.close()
        say("Nothing to do: every requested position is already finished.")
        return
    say(f"To do now: {len(todo)} positions ({sum(p['n_legal'] for p in todo):,} moves). Stop with Ctrl+C; run again to continue.")
    say("")

    info = {"engine": first.name, "hash_mb": args.hash, "threads_per_engine": 1,
            "positions_file": os.path.abspath(positions_file), "positions_listed": len(positions),
            "python_chess": chess.__version__, "positions_finished": len(done),
            "sessions": old.get("sessions", [])}
    with open(info_path, "w") as fh:           # written now, so a later run can check the settings
        json.dump(info, fh, indent=1)

    lock = threading.Lock()
    stop = threading.Event()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    done_file = open(done_path, "a", encoding="utf-8")
    state = {"next": 0, "new": 0, "moves": 0, "failed": [], "depth_sums": collections.Counter(),
             "depth_counts": collections.Counter(), "short": 0}
    moves_total = sum(p["n_legal"] for p in todo)
    t_start = time.time()

    def take():
        with lock:
            if stop.is_set() or state["next"] >= len(todo):
                return None
            p = todo[state["next"]]
            state["next"] += 1
            return p

    def commit(pos, rows, reached, depth_secs, secs):
        with lock:
            csv_writer.writerows(rows)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            done_file.write(f"{pos['pos_id']}\t{pos['n_legal']}\t{reached}\t{secs:.1f}\t{pos['fen']}\n")
            done_file.flush()
            os.fsync(done_file.fileno())
            done[pos["pos_id"]] = reached
            state["new"] += 1
            state["moves"] += pos["n_legal"]
            state["short"] += reached < args.depth
            for d, s in depth_secs.items():
                state["depth_sums"][d] += s
                state["depth_counts"][d] += 1
            elapsed = time.time() - t_start
            left = (moves_total - state["moves"]) * elapsed / state["moves"]
            say(f"[{state['new']:4d}/{len(todo)}] position {pos['pos_id']:5d}  {pos['n_legal']:2d} moves  "
                f"depth {reached:2d}  {secs:7.1f} s | running {clock(elapsed)} | about {clock(left)} left")

    def work(eng):
        while True:
            p = take()
            if p is None:
                return
            for attempt in (1, 2):
                try:
                    commit(p, *analyse_position(eng, p, args.depth, max_seconds, stop))
                    break
                except Stopped:
                    return
                except Exception as exc:
                    if stop.wait(0.5):         # the run is being stopped: not a real failure
                        return
                    say(f"Position {p['pos_id']} failed (attempt {attempt}): {exc!r}")
                    try:
                        eng.restart()
                    except Exception as exc2:
                        say(f"An engine could not be restarted: {exc2!r}")
                        with lock:
                            state["failed"].append(p["pos_id"])
                        return
            else:
                with lock:
                    state["failed"].append(p["pos_id"])

    engines = [first] + [Engine(stockfish, args.hash) for _ in range(workers - 1)]
    threads = [threading.Thread(target=work, args=(eng,), daemon=True) for eng in engines]
    keep_awake(True)
    interrupted = False
    try:
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        say("\nStopping ... (positions in progress are dropped; finished ones are safe)")
        stop.set()
    finally:
        stop.set()
        for eng in engines:
            eng.close()
        for t in threads:
            t.join(timeout=20)
        keep_awake(False)
        with lock:                             # never close in the middle of a save
            csv_file.close()
            done_file.close()

    # ---- Summary ----
    elapsed = time.time() - t_start
    depth_table = {d: round(state["depth_sums"][d] / state["depth_counts"][d], 2) for d in sorted(state["depth_counts"])}
    info["sessions"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                             "depth": args.depth, "workers": workers, "max_minutes_per_position": args.max_minutes,
                             "positions_finished": state["new"], "positions_failed": state["failed"],
                             "elapsed_seconds": round(elapsed), "mean_secs_to_depth": depth_table})
    info["positions_finished"] = len(done)
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=1)
    say("")
    say(f"{'Stopped' if interrupted else 'Finished'}: {state['new']} positions in this session ({clock(elapsed)}), "
        f"{len(done)} of {len(positions)} in total.")
    if depth_table:
        shown = [d for d in (10, 15, 18, 20, 22, 24, 25, 28, 30) if d in depth_table]
        if args.depth in depth_table and args.depth not in shown:
            shown.append(args.depth)
        say("Mean engine time per position to finish depth: "
            + ", ".join(f"{d}: {depth_table[d]:.1f} s" for d in shown))
    if state["short"]:
        say(f"{state['short']} position(s) hit the time limit before depth {args.depth}; their finished depths are kept.")
    if state["failed"]:
        say(f"Positions that failed and will be tried again next time: {state['failed']}")
    archive = shutil.make_archive(out.rstrip("/\\"), "zip", out)
    say(f"Results: {out}")
    say(f"Zipped:  {archive}")
    if len(done) < len(positions) and not args.max_positions:
        say("Run the same command again to continue.")


if __name__ == "__main__":
    main()
