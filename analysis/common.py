"""Loading the engine runs, the value scale and the relative error.

Value scale
-----------
x is the expected score of the player who made the move:

    x = 1 / (1 + exp(-k * cp)),   k = 0.00368208 * steepness

With steepness 1 this is the curve Lichess uses for its win probability. The centipawn value cp comes from:

    Stockfish  the centipawns it reports;
    Lc0        its win-minus-loss value converted with Lc0's own formula,
               cp = 90 * tan(1.5637541897 * wl)   (Lc0 0.32, src/search/classic/search.cc).

A forced mate for the player who moved counts as x = 1, being mated as x = 0, whatever the distance to mate.
Lc0 rows with a win-minus-loss value of +-1 (proven results) are treated the same way. Each engine's own
scale is kept as an alternative: Stockfish (w + d/2) / 1000 from its win/draw/loss model, Lc0 (1 + wl) / 2.

Relative error
--------------
    e(m) = (x_best - x(m)) / (x_best - x_worst),   with e = 0 for every move when all moves are worth the same.
"""
import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "out"

SF_RUN = RESULTS / "stockfish_run_depth25.zip"
LC0_RUN = RESULTS / "lc0_run_10000_nodes.zip"

LICHESS_K = 0.00368208     # Lichess win% curve: 50 + 50 * (2 / (1 + exp(-k cp)) - 1)


@dataclass
class Run:
    """One engine's evaluations of every legal move, at every level (Stockfish depth or Lc0 node level)."""
    engine: str
    level_name: str                 # "depth" or "nodes"
    levels: list                    # depth numbers, or the node count of each ladder level
    pos_ids: list
    fens: dict
    moves: dict                     # pos_id -> list of legal moves (UCI), sorted
    cp: dict                        # pos_id -> array (levels, moves): centipawns, +-inf for a decided result
    own: dict                       # pos_id -> array (levels, moves): expected score on the engine's own scale
    info: dict = field(default_factory=dict)

    def x(self, steepness=1.0, own_scale=False):
        """pos_id -> array (levels, moves) of expected scores."""
        if own_scale:
            return self.own
        k = LICHESS_K * steepness
        return {p: expit(k * v) for p, v in self.cp.items()}


def _csv_rows(zip_path, name, delimiter=","):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as fh:
            yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline=""), delimiter=delimiter)


def _json(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(name))


def lc0_cp(wl):
    """Lc0's own W-L -> centipawn conversion; proven results (|wl| = 1) become +-inf."""
    if abs(wl) >= 0.99995:
        return math.copysign(math.inf, wl)
    return 90.0 * math.tan(1.5637541897 * wl)


def _assemble(raw, n_levels):
    """raw: pos_id -> {level_index: {move: (cp, own)}} -> sorted moves and (levels, moves) arrays."""
    moves, cp, own = {}, {}, {}
    for p, by_level in raw.items():
        mv = sorted(by_level[0])
        moves[p] = mv
        cp[p] = np.array([[by_level[k][m][0] for m in mv] for k in range(n_levels)], dtype=float)
        own[p] = np.array([[by_level[k][m][1] for m in mv] for k in range(n_levels)], dtype=float)
    return moves, cp, own


def load_stockfish(zip_path=SF_RUN):
    info = _json(zip_path, "run_info.json")
    fens = {int(r["pos_id"]): r["fen"] for r in _csv_rows(zip_path, "positions_done.txt", "\t")}
    depth = max(int(r["depth_reached"]) for r in _csv_rows(zip_path, "positions_done.txt", "\t"))
    raw = {p: {k: {} for k in range(depth)} for p in fens}
    for r in _csv_rows(zip_path, "stockfish_evals.csv"):
        if r["mate"]:
            cpv = math.copysign(math.inf, int(r["mate"]))
        else:
            cpv = float(r["cp"])
        own = (int(r["w"]) + 0.5 * int(r["d"])) / 1000.0
        raw[int(r["pos_id"])][int(r["depth"]) - 1][r["move"]] = (cpv, own)
    moves, cp, own = _assemble(raw, depth)
    return Run("Stockfish", "depth", list(range(1, depth + 1)), sorted(fens), fens, moves, cp, own, info)


def load_lc0(zip_path=LC0_RUN):
    info = _json(zip_path, "run_info.json")
    ladder = info["ladder"]
    fens = {int(r["pos_id"]): r["fen"] for r in _csv_rows(zip_path, "positions_analyzed.txt", "\t")}
    raw = {p: {k: {} for k in range(len(ladder))} for p in fens}
    for r in _csv_rows(zip_path, "lc0_evals.csv"):
        wl = float(r["wl"])
        if r["mate"]:
            wl = math.copysign(1.0, int(r["mate"]))
        raw[int(r["pos_id"])][int(r["level"])][r["move"]] = (lc0_cp(wl), (1.0 + wl) / 2.0)
    moves, cp, own = _assemble(raw, len(ladder))
    return Run("Lc0", "nodes", list(ladder), sorted(fens), fens, moves, cp, own, info)


def relative_errors(x):
    """x: array (levels, moves) of expected scores -> relative errors of the same shape (0/0 := 0)."""
    best = x.max(axis=1, keepdims=True)
    den = best - x.min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (best - x) / den, 0.0)


def piece_count(fen):
    """Pieces on the board, kings and pawns included."""
    return sum(ch.isalpha() for ch in fen.split()[0])


def write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
