# v2 analysis

Analysis of the paired runs on the same 264 positions: Stockfish 19 to depth 25 (`../results/stockfish_run_depth25.zip`) and Lc0 BT4 to 10,000 nodes per move (`../results/lc0_run_10000_nodes.zip`). The earlier Lc0 run to 1,000 nodes on 1,365 positions (`../results/lc0_run_1000_nodes.zip`) is kept as a cross-check. The scripts read the zip files directly.

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib` and `chess`. Run the scripts from this folder, in order; each writes its tables to `out/`.

| Script | What it does | Run time |
|---|---|---|
| `common.py` | Loads the runs and defines the value scale and the relative error (imported by the others). | |
| `01_scale_sensitivity.py` | How much the scale choices move the near-best share at the top level. | seconds |
| `02_count_distributions.py` | Fits five count models to the number of near-best moves at every level and tolerance. | ~20 s |
| `03_depth_trends.py` | Fits trend shapes against depth, tests them on held-out depths, and extrapolates to infinite depth with bootstrap error bars. | ~5 min |
| `05_result_keeping.py` | The share of moves that keep the best result according to each engine's own win/draw/loss verdict, at every depth and node level. | seconds |
| `06_recognition_and_decisions.py` | How far the near-best moves overlap the result-keeping moves, how often the best move changes with depth, and whether the engines grow more or less optimistic. | seconds |
| `07_error_distribution_look.py` | The relative error itself, binned, by depth and node level. | seconds |
| `08_zoib_look.py` | A first, pooled fit of the zero-one-inflated beta to the relative error at five levels per engine. | seconds |
| `09_half_tolerance_check.py` | The share of moves with e <= 1/2 across scales, depths and engines. | seconds |
| `10_error_distribution_fits.py` | Fits candidate distributions to the relative error at every level separately (pooled over positions, each position weighted equally) and compares them; also a sensitivity version with errors within 0.001 of 0 or 1 counted as 0 or 1. | ~15 s |
| `11_stockfish_parameters.py` | Stockfish only: every parameter of the zero-one-inflated beta and logit-normal against depth, with 95% bootstrap bands; the logit-normal also with near-ties (within 0.001 of 0 or 1) counted as ties. | ~10 s |
| `12_logitnormal_both_versions.py` | Stockfish only: the zero-one-inflated logit-normal (p0, p1, m, s) by depth in both versions (errors as measured; near-ties within 0.001 counted as ties), with two five-parameter extensions of the middle (skewness; heavier tails). | ~15 s |
| `13_depth_functions.py` | Stockfish only: functions of depth for p0, p1, m and s (saturating exponential, power approach, logistic S-curve; linear and log-linear benchmarks), with held-out tests on depths 21-25 and bootstrap intervals for the asymptotes. | ~25 s |
| `tablebase_query.py` | Asks the Lichess tablebase for the perfect-play result of every move in `tablebase_positions.txt` (the 78 positions of the first Lc0 run with at most 7 pieces; 16 are among the 264 paired positions). Needs internet access: `python tablebase_query.py --positions tablebase_positions.txt`, then put `tablebase_results.csv` in `../results/`. | ~2 min |
| `04_tablebase_check.py` | Compares the engines and the extrapolation with perfect play in those positions (reads `../results/tablebase_results.csv`). | ~10 s |
| `figures.py` | Draws the figures in `out/`. | ~10 s |

## Definitions fixed by these scripts

**Value of a move.** The expected score of the player who made it, from one curve for both engines: `x = 1 / (1 + exp(-k * cp))` with the Lichess constant `k = 0.00368208`. Stockfish's centipawns are used as reported. Lc0's W-L value is converted with Lc0's own formula, `cp = 90 * tan(1.5637541897 * wl)`. A forced mate (or a W-L of +-1) counts as 1 for the winner and 0 for the loser, whatever the distance to mate.

**Relative error.** `e = (x_best - x) / (x_best - x_worst)`, with `0/0 := 0` (every move counts as perfect when all are worth the same). A floor on the denominator was tested and rejected: Lc0 becomes certain of a result earlier than Stockfish, so a floor bites in different positions for the two engines and pulls them apart (`01_scale_sensitivity.py`).

**Tolerances.** e <= 0.05, 0.10 and 0.25.

**Count model.** In each position, `K` = the number of moves other than the best with e <= tolerance, out of `n` = legal moves - 1. This respects that the best move is always near-best and that positions have different numbers of legal moves.

**Trends.** Stockfish: train on depths 5-20, hold out 21-25 (depths 1-4 are mostly quiescence search). Lc0: train on 1-1,000 nodes, hold out 1,468-10,000. Shapes: constant, linear (no limit), exponential approach and power approach. The shape with the smallest held-out error is refitted on all levels; its value at infinite depth is the limit. Error bars come from refitting on positions resampled with replacement.

**Infinite depth.** Deferred: the analysis so far describes the measured depths and node levels only. Step 5 measures, at each level, which moves keep the result according to the engine's own verdict; step 4 compares the engines with perfect play in the positions with at most 7 pieces.

## Results so far (264 positions, Stockfish depth 25 and Lc0 10,000 nodes)

1. **Scale.** On the common curve the two engines agree: the share of moves with e <= 0.10 is 0.334 for Stockfish and 0.322 for Lc0, and the per-position counts correlate at 0.91. Making the curve twice as gentle or twice as steep moves the share from about 0.37 to about 0.28, but the engines stay within 0.02 of each other. On each engine's own win/draw/loss scale they disagree (0.435 against 0.244).
2. **Distribution.** The binomial is rejected at every level. The beta-binomial passes the goodness-of-fit test at all 25 levels for e <= 0.05 and 0.10, for both engines. At e <= 0.25 there are more positions with exactly one acceptable move than it predicts, and a zero-inflated beta-binomial (an extra 3-10% of such positions) fits better. At the top level, e <= 0.10: Stockfish mu = 0.285, phi = 1.72; Lc0 mu = 0.271, phi = 1.75. Stockfish's parameters move with depth until they meet Lc0's, which hardly move at all.
3. **Depth trends.** Lc0 is flat from 1 to 10,000 nodes, so its limits are close to its current values. Stockfish at e <= 0.05 and 0.10 is still rising at depth 25 (about 0.4 and 0.3 points of share per depth), and the held-out test cannot tell a levelling-off curve from a straight line: the limit is not identified from depths 1-25. At e <= 0.25 both engines are flat (0.425 and 0.403).
4. **Result-keeping moves (by the engines' own verdicts).** 0.562 of the legal moves for Stockfish at depth 25 and 0.545 for Lc0 at 10,000 nodes, about 14 moves per position. The share barely depends on search effort (Stockfish 0.547 at depth 5, Lc0 0.529 at 1 node), the engines give the same verdict for the side to move in 95% of positions, and by verdict the share is 0.42 in won positions, 0.44 in drawn positions and 1 in lost positions (every move loses). Stockfish at depth 12 gives 96.6% of moves the same result as depth 25.

5. **Distribution of the error itself (step 10).** Fitted level by level. For Stockfish the zero-one-inflated logit-normal fits best (lowest AIC at 21 of 25 depths; largest gap to the observed curve about 5 percentage points), the zero-one-inflated beta next (about 7). For Lc0 the zero-one-inflated beta has the lowest AIC at all 25 levels but no candidate follows the curve (largest gaps 15-20 points): Lc0 rates about a quarter of all moves within 0.0001 of its worst move. Counting errors within 0.001 of 0 or 1 as 0 or 1, the logit-normal version wins at all 25 levels for both engines (largest gaps about 4 points for Lc0, 6 for Stockfish). The ordered beta coincides with the zero-one-inflated beta when each level is fitted on its own; censored models fit poorly. Every single-family middle part still misses some shape that a two-beta benchmark captures.

Still to do: the tablebase check (78 positions, 16 of them paired), and walks down the tree to estimate its growth per move.
