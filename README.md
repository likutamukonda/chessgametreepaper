# Rational pruning in chess: code and data

Code and data for *Rational Pruning in Chess: How the Distribution of Relative Move Error Changes with Search Depth* by Shikukumwa Likuta Mukonda (University of Namibia; draft, November 2026).

The study takes 264 positions from Top Chess Engine Championship (TCEC) games, scores every legal move of each position with Stockfish 19 at every search depth from 1 to 25, and places each move on a relative scale from the best move of its position (0) to the worst (1). It then asks which distribution describes this relative error, how its parameters change with depth, and where they settle as depth goes to infinity.

## Layout

| Folder | What it holds |
|---|---|
| `data/` | `optimal_positions.zip`: 1,000,000 positions from TCEC games after the opening book, one FEN per line. |
| `runs/` | The two engine runs. `lc0_kaggle.ipynb` (Kaggle, 2 × T4 GPU) takes the positions in a fixed, seeded order that keeps the piece-count mix of the whole file and scores every legal move with Lc0 0.32 (BT4 network) at 25 node counts from 1 to 10,000; the 264 positions it finished in its time budget are the sample. `stockfish_laptop.py` (a Windows laptop) scores every legal move of the same positions with Stockfish 19 at every depth from 1 to 25. Instructions are at the top of each file. |
| `results/` | The outputs of the two runs, zipped, exactly as the analysis reads them: `stockfish_run_depth25.zip` (4.6 hours with 10 engines side by side) and `lc0_run_10000_nodes.zip` (8.4 hours). |
| `analysis/` | The analysis: `common.py`, `model.py`, the scripts `01`–`10` and their outputs in `out/`. |

Positions whose castling rights exist only in Chess960 (about 1.4% of the file) are left out by the Lc0 notebook.

## The analysis

Every script reads the zipped runs from `results/` and writes to `analysis/out/`. The main measure is Stockfish's; Lc0 appears only in a side check (script 09).

| File | What it does | Main outputs |
|---|---|---|
| `common.py` | Loads the runs; turns centipawns into expected scores with the Lichess curve, x = 1 / (1 + exp(−0.00368208 · cp)); computes the relative error e = (x_best − x) / (x_best − x_worst). | |
| `model.py` | The model shared by the scripts: the zero-one-inflated logit-normal (p0 = share at e = 0, p1 = share at e = 1, logit e ~ Normal(m, s) in between), its closed-form estimates with every position weighted equally, the depth functions (S-curves; gamma kernel for s), and the bootstrap over positions (200 resamples, seed 20261120). | |
| `01_candidate_distributions.py` | Seven candidate distributions fitted at every depth, as measured and with near-ties (within 0.001 of 0 or 1) counted as ties. | `candidate_distributions.csv` |
| `02_normal_extensions.py` | Does the logit part need skew or heavier tails than the normal? | `normal_extensions.csv` |
| `03_depth_shapes.py` | Exponential, power, S-curve, gamma kernel and straight line as functions of depth, with a held-out test on depths 21–25. | `depth_shapes.csv` |
| `04_spike_parameterisation.py` | Which two of p0, p1 and the middle share should carry depth functions. | `spike_parameterisation.csv` |
| `05_infinite_depth.py` | Why depths 1–4 are left out (horizon effect), the depth functions on depths 5–25, and the distribution at infinite depth with 95% intervals. | `shallow_depths.csv`, `depth_functions.csv`, `infinite_depth.csv` |
| `06_game_phase.py` | The distribution by number of pieces on the board. | `phase_breakdown.csv` |
| `07_goodness_of_fit.py` | How well the model fits at each depth (largest and average gap, share of moves misplaced, R² over bands, a Wald test that resamples positions), the shape of the fitted middle at depth 25, how well the depth functions fit (R² against a straight line, change from depth 20 to 25), and what p0 and p1 are made of. | `fit_by_depth.csv`, `band_residuals_depth25.csv`, `depth_function_checks.csv`, `p0_p1_split.csv` |
| `08_parameter_correlations.py` | Correlations between the parameters: across positions, between the estimates, and between the infinite-depth levels. | `parameter_correlations.csv` |
| `09_robustness.py` | Scale steepness (and the check that the two ends do not depend on it), near-ties, the first depth fitted, Lc0 on the same positions, and weighting by moves instead of positions. | `robustness_*.csv` |
| `10_paper_figures.py` | The figures and LaTeX tables of the paper (reads the outputs of 01–09). | `out/paper/` |

### Running it

Python 3.10 or later with NumPy, SciPy and Matplotlib (tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11):

```
pip install numpy scipy matplotlib
cd analysis
python run_all.py
```

`run_all.py` runs the scripts in order; the whole analysis takes about three minutes. Each script can also be run on its own (`python 05_infinite_depth.py`) once the scripts before it have run. Results are the same on every run: the bootstrap uses a fixed seed.

## Earlier work

The first draft (October 2026) and the exploratory analysis that led to this version (count models, other distributions, a closer comparison with Lc0, the tablebase check) are kept in the history of this repository: see commit `87e0ce5` on the `v2-analysis` branch.
