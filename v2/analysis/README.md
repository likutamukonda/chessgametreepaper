# v2 analysis

Analysis of the paired runs on the same 264 positions: Stockfish 19 to depth 25 (`../results/stockfish_run_depth25.zip`) and Lc0 BT4 to 10,000 nodes per move (`../results/lc0_run_10000_nodes.zip`). The earlier Lc0 run to 1,000 nodes on 1,365 positions (`../results/lc0_run_1000_nodes.zip`) is kept as a cross-check. The scripts read the zip files directly.

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib` and `chess`. Run the scripts from this folder, in order; each writes its tables to `out/`.

| Script | What it does | Run time |
|---|---|---|
| `common.py` | Loads the runs and defines the value scale and the relative error (imported by the others). | |
| `01_scale_sensitivity.py` | How much the scale choices move the near-best share at the top level. | seconds |
| `02_count_distributions.py` | Fits five count models to the number of near-best moves at every level and tolerance. | ~20 s |
| `03_depth_trends.py` | Fits trend shapes against depth, tests them on held-out depths, and extrapolates to infinite depth with bootstrap error bars. | ~25 min |
| `tablebase_query.py` | Asks the Lichess tablebase for the perfect-play result of every move in the positions with at most 7 pieces. Needs internet access; run it on any computer. | ~1 min |
| `04_tablebase_check.py` | Compares the engines and the extrapolation with perfect play in those positions (reads `../results/tablebase_results.csv`). | ~10 s |
| `figures.py` | Draws the figures in `out/`. | ~10 s |

## Definitions fixed by these scripts

**Value of a move.** The expected score of the player who made it, from one curve for both engines: `x = 1 / (1 + exp(-k * cp))` with the Lichess constant `k = 0.00368208`. Stockfish's centipawns are used as reported. Lc0's W-L value is converted with Lc0's own formula, `cp = 90 * tan(1.5637541897 * wl)`. A forced mate (or a W-L of +-1) counts as 1 for the winner and 0 for the loser, whatever the distance to mate.

**Relative error.** `e = (x_best - x) / (x_best - x_worst)`, with `0/0 := 0` (every move counts as perfect when all are worth the same). A floor on the denominator was tested and rejected: Lc0 becomes certain of a result earlier than Stockfish, so a floor bites in different positions for the two engines and pulls them apart (`01_scale_sensitivity.py`).

**Tolerances.** e <= 0.05, 0.10 and 0.25.

**Count model.** In each position, `K` = the number of moves other than the best with e <= tolerance, out of `n` = legal moves - 1. This respects that the best move is always near-best and that positions have different numbers of legal moves.

**Trends.** Stockfish: train on depths 5-20, hold out 21-25 (depths 1-4 are mostly quiescence search). Lc0: train on 1-1,000 nodes, hold out 1,468-10,000. Shapes: constant, linear (no limit), exponential approach and power approach. The shape with the smallest held-out error is refitted on all levels; its value at infinite depth is the limit. Error bars come from refitting on positions resampled with replacement.
