# Chess game-tree paper: code and data

Code and data for *Rational Pruning and Structural Decomposition of the Chess Game Tree* by Shikukumwa Likuta Mukonda. The paper is being reworked, so this repository holds two versions side by side.

## Layout

| Folder | What it holds |
|---|---|
| `data/` | `optimal_positions.zip`: 1,000,000 positions sampled from TCEC engine games after the opening book, one FEN per line (board, side to move, castling rights, en passant square). |
| `v1_first_draft/` | The notebooks, scripts and results behind the October 2026 draft, unchanged. |
| `v2/` | The reworked pipeline. So far: the Lc0 notebook for Kaggle and a checker for its output. |

About 1.4% of the positions carry castling rights that only exist in Chess960. The v2 pipeline leaves those positions out.

## v2: Lc0 evaluation on Kaggle

`v2/lc0_kaggle.ipynb` asks Leela Chess Zero how good the position is after each legal move, at a ladder of increasing node counts.

- **Sample.** Positions are taken in a fixed, seeded order that keeps the piece-count mix of the whole file, so wherever a run stops, the finished positions are a fair sample.
- **Ladder.** 25 node counts, evenly spaced on a log scale from 1 node (the network alone) to the top level: 1, 2, 3 ... 8, 10, 13, 18 ... 750, 1000 (eight per factor of ten). Lc0 keeps its search tree between the readings, so the whole ladder costs about as much as its top level plus roughly 0.02 s per reading. Lc0 does not stop on exactly the node count asked for (it overshoots by a few per cent, and by one or two nodes at the very bottom), so use the `nodes` column, not `target_nodes`, as the x-axis.
- **Top level.** `TOP_NODES = 1000` by default. On the two T4s of the first run this leaves room for more than 1,000 positions in 8 hours (about 1,200, estimated). Setting `TOP_NODES = None` lets a short calibration pick the highest top level that fits `TARGET_POSITIONS`.
- **Engine settings.** Lc0 0.32 with the BT4 network, score type W-L, smart pruning off (a search is never stopped before its node target).
- **Safety.** Results are written after every position, and the run stops by itself before the time budget ends. A later session can continue an earlier run.

### Running it

1. Import the notebook into Kaggle. Choose the accelerator *GPU T4 x2*, turn internet on, and attach the positions dataset. If no dataset is attached, the notebook fetches the positions from this repository.
2. For a trial, set `TOTAL_BUDGET_HOURS = 0.5` in the first code cell and run all cells. The log shows the seconds each position takes. (With `TOP_NODES = None` it also prints how many hours `TARGET_POSITIONS` positions would take at each top level.)
3. For the full run, set `TOTAL_BUDGET_HOURS = 8.0` and use *Save Version → Save & Run All*.
4. Download `lc0_run.zip` from the notebook's output.

### Output

| File | Contents |
|---|---|
| `lc0_evals.csv` | One row per position, move and ladder level: nodes reached, wins minus losses for the player who made the move (`wl`), win/draw/loss chances, proven mates, and game-ending moves. |
| `positions_analyzed.txt` | The finished positions, tab-separated: place in the seeded order, line number in `optimal_positions.txt`, number of legal moves, FEN. The Stockfish step reads this file. |
| `positions_order.csv` | The seeded order (first 5,000 positions), with the reason for any position that is left out. |
| `run_info.json` | Settings, ladder, calibration table and timings. |

To check a finished run:

```
pip install chess
python v2/check_lc0_run.py lc0_run optimal_positions.txt
```

## v1: first draft

Run the scripts from inside `v1_first_draft/`; they read and write files by name in that folder.

1. `stockfish-analysis.ipynb` and `leela-analysis.ipynb` (Kaggle) produce `continuous_pilot_results_d20.csv` and `lc0_erel_qvalues_nodes.csv`.
2. `aggregate_lc0.py` turns the Lc0 file into `lc0_aggregated_nodes.csv`.
3. `fit_zib_distribution.py` fits the distribution at each depth or node count and writes `stockfish_zib_params.csv` and `lc0_zib_params.csv`.
4. `trends.py` and `alpha_and_beta.py` fit the parameters against depth and write their tables to `trend_fitting_results/`.
5. `test_and_graphs.py` reads those tables and draws the figures.
