"""Shared definitions for the v2 analysis: loading the runs and the common value scale.

The value scale (fixed for the paper)
-------------------------------------
x is the expected score of the player who made the move, put on one curve for both engines:

    x = 1 / (1 + exp(-k * cp)),   k = 0.00368208 * STEEPNESS

With STEEPNESS = 1 this is the Lichess win% curve. The centipawn value cp comes from:

    Stockfish  the centipawns it reports (Stockfish 19 normalises them so that +100 cp is
               about a 50% chance of winning in the middlegame);
    Lc0        its W-L value converted with Lc0's own "centipawn" formula,
               cp = 90 * tan(1.5637541897 * wl)   (Lc0 v0.32.1, src/search/classic/search.cc).

A forced mate for the player who moved counts as x = 1, being mated as x = 0, whatever the
distance to mate. Lc0 rows with W-L of +-1 (proven results) are treated the same way.

Relative error of move m in a position:

    e(m) = (x_best - x(m)) / max(x_best - x_worst, DELTA)

The main results use DELTA = 0: the original definition, with 0/0 := 0 when every move is worth
the same. DELTA > 0 is a floor that grades moves against DELTA when their spread is smaller. It was
tested (01_scale_sensitivity.py) and rejected: Lc0 becomes certain of a result earlier than
Stockfish, so a floor of 0.01 bites in 24 Lc0 positions but 12 Stockfish ones and pulls the two
engines apart (share at e <= 0.10: 0.338 vs 0.382, against 0.334 vs 0.322 without it). Plain
relative error normalises within each position, which is what keeps the engines comparable.

Each engine's own scale is kept for comparison: Lc0 (1 + wl) / 2 and Stockfish (w + d/2) / 1000
from its WDL model.
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

LC0_RUN = RESULTS / "lc0_run_10000_nodes.zip"
LC0_RUN_1000 = RESULTS / "lc0_run_1000_nodes.zip"
SF_RUN = RESULTS / "stockfish_run_depth25.zip"

LICHESS_K = 0.00368208     # Lichess win% curve: 50 + 50 * (2 / (1 + exp(-k cp)) - 1)
STEEPNESS = 1.0            # multiple of LICHESS_K used for the main results
DELTA = 0.0                # floor on the spread, in expected score (0 = plain 0/0 := 0; see above)
THRESHOLDS = (0.05, 0.10, 0.25)


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

    def x(self, steepness=STEEPNESS, own_scale=False):
        """pos_id -> array (levels, moves) of expected scores."""
        if own_scale:
            return self.own
        k = LICHESS_K * steepness
        return {p: expit(k * v) for p, v in self.cp.items()}

    def subset(self, pos_ids):
        keep = [p for p in self.pos_ids if p in set(pos_ids)]
        return Run(self.engine, self.level_name, self.levels, keep, {p: self.fens[p] for p in keep},
                   {p: self.moves[p] for p in keep}, {p: self.cp[p] for p in keep},
                   {p: self.own[p] for p in keep}, self.info)


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


def relative_errors(x, delta=DELTA):
    """x: array (levels, moves) of expected scores -> relative errors of the same shape."""
    best = x.max(axis=1, keepdims=True)
    den = np.maximum(best - x.min(axis=1, keepdims=True), delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (best - x) / den, 0.0)


def near_best_counts(run, thresholds=THRESHOLDS, steepness=STEEPNESS, delta=DELTA, own_scale=False):
    """Array (positions, levels, thresholds) of the number of moves with e <= threshold, and the
    array (positions,) of legal-move counts. Positions follow run.pos_ids."""
    xs = run.x(steepness, own_scale)
    counts = np.empty((len(run.pos_ids), len(run.levels), len(thresholds)), dtype=int)
    for i, p in enumerate(run.pos_ids):
        e = relative_errors(xs[p], delta)
        for j, t in enumerate(thresholds):
            counts[i, :, j] = (e <= t + 1e-12).sum(axis=1)
    n_legal = np.array([len(run.moves[p]) for p in run.pos_ids])
    return counts, n_legal


def near_best_sets(run, level_index, threshold, steepness=STEEPNESS, delta=DELTA, own_scale=False):
    """pos_id -> set of moves with e <= threshold at one level."""
    xs = run.x(steepness, own_scale)
    out = {}
    for p in run.pos_ids:
        e = relative_errors(xs[p][level_index:level_index + 1], delta)[0]
        out[p] = {m for m, v in zip(run.moves[p], e) if v <= threshold + 1e-12}
    return out


def piece_count(fen):
    return sum(ch.isalpha() for ch in fen.split()[0])


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
