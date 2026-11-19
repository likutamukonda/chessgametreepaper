# v2 analysis

Analysis of the paired runs on the same 264 positions: Stockfish 19 to depth 25 (`../results/stockfish_run_depth25.zip`) and Lc0 BT4 to 10,000 nodes per move (`../results/lc0_run_10000_nodes.zip`). The earlier Lc0 run to 1,000 nodes on 1,365 positions (`../results/lc0_run_1000_nodes.zip`) is kept as a cross-check. The scripts read the zip files directly.

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib` and `chess`. Run the scripts from this folder, in order; each writes its tables to `out/`.

| Script | What it does | Run time |
|---|---|---|
| `common.py` | Loads the runs and defines the value scale and the relative error (imported by the others). | |
| `01_scale_sensitivity.py` | How much the scale choices move the near-best share at the top level. | seconds |
| `02_count_distributions.py` | Fits five count models to the number of near-best moves at every level and tolerance. | ~20 s |
| `03_depth_trends.py` | Fits trend shapes against depth, tests them on held-out depths, and extrapolates to infinite depth with bootstrap error bars. | ~5 min |
| `05_result_keeping.py` | Reads the infinite-depth limit directly: the share of moves that keep the best result, from each engine's win/draw/loss verdicts. | seconds |
| `tablebase_query.py` | Asks the Lichess tablebase for the perfect-play result of every move in the positions with at most 7 pieces. Needs internet access; run it on any computer. | ~1 min |
| `04_tablebase_check.py` | Compares the engines and the extrapolation with perfect play in those positions (reads `../results/tablebase_results.csv`). | ~10 s |
| `figures.py` | Draws the figures in `out/`. | ~10 s |

## Definitions fixed by these scripts

**Value of a move.** The expected score of the player who made it, from one curve for both engines: `x = 1 / (1 + exp(-k * cp))` with the Lichess constant `k = 0.00368208`. Stockfish's centipawns are used as reported. Lc0's W-L value is converted with Lc0's own formula, `cp = 90 * tan(1.5637541897 * wl)`. A forced mate (or a W-L of +-1) counts as 1 for the winner and 0 for the loser, whatever the distance to mate.

**Relative error.** `e = (x_best - x) / (x_best - x_worst)`, with `0/0 := 0` (every move counts as perfect when all are worth the same). A floor on the denominator was tested and rejected: Lc0 becomes certain of a result earlier than Stockfish, so a floor bites in different positions for the two engines and pulls them apart (`01_scale_sensitivity.py`).

**Tolerances.** e <= 0.05, 0.10 and 0.25.

**Count model.** In each position, `K` = the number of moves other than the best with e <= tolerance, out of `n` = legal moves - 1. This respects that the best move is always near-best and that positions have different numbers of legal moves.

**Trends.** Stockfish: train on depths 5-20, hold out 21-25 (depths 1-4 are mostly quiescence search). Lc0: train on 1-1,000 nodes, hold out 1,468-10,000. Shapes: constant, linear (no limit), exponential approach and power approach. The shape with the smallest held-out error is refitted on all levels; its value at infinite depth is the limit. Error bars come from refitting on positions resampled with replacement.

**The limit.** With perfect play every move's value is exactly a win, draw or loss, so every relative error is 0, 1/2 or 1. For any tolerance below 1/2 the near-best moves at infinite depth are therefore the same set: the moves that keep the position's best result (all moves, when every move gives the same result). Step 5 estimates this set from the engines' own win/draw/loss verdicts; step 4 checks it against tablebases.

## Results so far (264 positions, Stockfish depth 25 and Lc0 10,000 nodes)

1. **Scale.** On the common curve the two engines agree: the share of moves with e <= 0.10 is 0.334 for Stockfish and 0.322 for Lc0, and the per-position counts correlate at 0.91. Making the curve twice as gentle or twice as steep moves the share between about 0.36 and 0.30, but the engines stay within 0.02 of each other. On each engine's own win/draw/loss scale they disagree (0.435 against 0.244).
2. **Distribution.** The binomial is rejected at every level. The beta-binomial passes the goodness-of-fit test at all 25 levels for e <= 0.05 and 0.10, for both engines. At e <= 0.25 there are more positions with exactly one acceptable move than it predicts, and a zero-inflated beta-binomial (extra share 0.04-0.10 of such positions) fits better. At the top level, e <= 0.10: Stockfish mu = 0.285, phi = 1.72; Lc0 mu = 0.271, phi = 1.75. Stockfish's parameters move with depth until they meet Lc0's, which hardly move at all.
3. **Depth trends.** Lc0 is flat from 1 to 10,000 nodes, so its limits are close to its current values. Stockfish at e <= 0.05 and 0.10 is still rising at depth 25 (about 0.4 and 0.3 points of share per depth), and the held-out test cannot tell a levelling-off curve from a straight line: the limit is not identified from depths 1-25. At e <= 0.25 both engines are flat (0.425 and 0.403).
4. **Result-keeping moves (the limit).** 0.562 of the legal moves for Stockfish at depth 25 and 0.545 for Lc0 at 10,000 nodes, about 14 moves per position. The share barely depends on search effort (Stockfish 0.547 at depth 5, Lc0 0.529 at 1 node), the engines give the same verdict for the side to move in 95% of positions, and by verdict the share is 0.42 in won positions, 0.44 in drawn positions and 1 in lost positions (every move loses). Stockfish at depth 12 gives 96.6% of moves the same result as depth 25.

Still to do: the tablebase check (16 positions), and walks down the tree to estimate its growth per move.
