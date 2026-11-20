"""The model shared by the analysis scripts (Stockfish).

Distribution at one depth: the zero-one-inflated logit-normal for the relative error e of the legal moves,
    P(e = 0) = p0,   P(e = 1) = p1,   and for 0 < e < 1   logit(e) ~ Normal(m, s),
where logit(e) = log(distance to the best move / distance to the worst move). The middle share is
middle = 1 - p0 - p1, and the share of moves with e <= t is
    F(t) = p0 + middle * Phi((logit t - m) / s)      for 0 < t < 1.
The parameters are estimated in closed form from all moves of all positions, each move weighted
1 / (legal moves in its position) so that every position counts equally: p0 and p1 are weighted shares, m and s
the weighted mean and standard deviation of logit(e) over the moves strictly between 0 and 1.
Near-ties: errors within SNAP = 0.001 of 0 or 1 count as exactly 0 or 1.

Depth functions, fitted to the parameters at depths FIRST..25 by weighted least squares (weights 1 / se^2, se the
bootstrap standard error of the parameter at that depth):
    logistic S-curve   y(d) = A + (L - A) / (1 + exp(-k (d - d0)))      for p0, middle and m; level L
    gamma kernel       s(d) = c + H (d / d_p)^a exp(a (1 - d / d_p))     for s; level c (the S-curve as alternative)
    p1 = 1 - p0 - middle.
Uncertainty: B bootstrap resamples of the positions (seed SEED), the same resample for every depth and parameter.
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import logit, ndtr

import common as c

warnings.simplefilter("ignore", OptimizeWarning)

SNAP, B, SEED, FIRST = 1e-3, 200, 20261120, 5
QUANTS = ("p0", "middle", "p1", "m", "s")
GRID = np.concatenate([[0.0], np.linspace(1e-4, 0.9999, 2000)])
BINS = ([("exactly 0", None)]
        + [(f"{a:g}-{round(a + 0.1, 1):g}", (a, round(a + 0.1, 1))) for a in np.round(np.arange(0, 1, 0.1), 1)]
        + [("exactly 1", None)])
PHASES = (("12 or fewer", 0, 12), ("13-22", 13, 22), ("23 or more", 23, 32))


# ---------------------------------------------------------------- data
def snap(e):
    return np.where(e < SNAP, 0.0, np.where(e > 1 - SNAP, 1.0, e))


def errors_by_depth(run, xs=None, near_ties=True):
    """List over depths of lists over positions of relative-error arrays."""
    xs = run.x() if xs is None else xs
    f = snap if near_ties else (lambda e: e)
    return [[f(c.relative_errors(xs[p][li:li + 1])[0]) for p in run.pos_ids] for li in range(len(run.levels))]


def parameters(per_depth, idx):
    """per_depth: list over depths of per-position error arrays; idx: positions to use -> dict of arrays over depths."""
    out = {k: [] for k in QUANTS}
    for per_pos in per_depth:
        e = np.concatenate([per_pos[i] for i in idx])
        w = np.concatenate([np.full(len(per_pos[i]), 1.0 / len(per_pos[i])) for i in idx])
        W = w.sum()
        p0, p1 = w[e == 0].sum() / W, w[e == 1].sum() / W
        inner = (e > 0) & (e < 1)
        z, wz = logit(e[inner]), w[inner] / w[inner].sum()
        m = float(np.sum(wz * z))
        out["p0"].append(p0)
        out["p1"].append(p1)
        out["middle"].append(1 - p0 - p1)
        out["m"].append(m)
        out["s"].append(float(np.sqrt(np.sum(wz * (z - m) ** 2))))
    return {k: np.array(v) for k, v in out.items()}


def at(params, li):
    """The parameters of one depth as a dict of floats."""
    return {q: float(params[q][li]) for q in QUANTS}


# ---------------------------------------------------------------- the distribution
def share_curve(p0, middle, m, s, t=GRID):
    """F(t): share of moves with e <= t."""
    t = np.asarray(t, float)
    with np.errstate(divide="ignore"):
        inner = ndtr((logit(np.clip(t, 1e-300, 1 - 1e-16)) - m) / s)
    return np.where(t >= 1, 1.0, np.where(t <= 0, p0, p0 + middle * inner))


def model_curve(par, t=GRID):
    return share_curve(par["p0"], par["middle"], par["m"], par["s"], t)


def observed_curve(per_pos, t=GRID):
    """Observed share of moves with e <= t, each position weighted equally."""
    t = np.asarray(t, float)
    return np.mean([(e[None, :] <= t[:, None] + 1e-12).mean(axis=1) for e in per_pos], axis=0)


def band_probs(p0, middle, m, s):
    """Model share of moves in each band of BINS: exactly 0, 0-0.1, ..., 0.9-1, exactly 1."""
    out = []
    for name, edges in BINS:
        if edges is None:
            out.append(p0 if name == "exactly 0" else 1 - p0 - middle)
            continue
        a, b = edges
        upper = 1.0 if b >= 1 else ndtr((logit(b) - m) / s)
        lower = 0.0 if a <= 0 else ndtr((logit(a) - m) / s)
        out.append(middle * (upper - lower))
    return np.array(out)


def observed_bands(per_pos):
    """Observed share of moves in each band of BINS, each position weighted equally."""
    e = np.concatenate(per_pos)
    w = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in per_pos])
    inner = (e > 0) & (e < 1)
    out = np.empty(len(BINS))
    for i, (name, edges) in enumerate(BINS):
        if edges is None:
            out[i] = w[e == (0.0 if name == "exactly 0" else 1.0)].sum()
        else:
            out[i] = w[(e > edges[0]) & (e <= edges[1]) & inner].sum()
    return out / len(per_pos)


def band_diff(per_pos):
    """Observed minus fitted share of moves in each band of BINS (zero at exactly 0 and 1 by construction)."""
    par = at(parameters([per_pos], np.arange(len(per_pos))), 0)
    return observed_bands(per_pos) - band_probs(par["p0"], par["middle"], par["m"], par["s"])


# ---------------------------------------------------------------- functions of depth
RANGE = {"p0": (0.0, 1.0), "p1": (0.0, 1.0), "middle": (0.0, 1.0), "m": (-10.0, 10.0), "s": (0.0, 20.0)}
GK_LB, GK_UB = [0.0, -20.0, 1.0, 0.01], [20.0, 20.0, 500.0, 50.0]


def gamma_kernel(d, cc, H, dp, a):
    return cc + H * (d / dp) ** a * np.exp(a * (1 - d / dp))


SHAPE_F = {"exponential": lambda d, L, cc, k: L - cc * np.exp(-k * d),
           "power": lambda d, L, cc, g: L - cc * np.power(d, -g),
           "logistic": lambda d, A, L, k, d0: A + (L - A) / (1 + np.exp(-k * (d - d0))),
           "gamma kernel": gamma_kernel,
           "linear": lambda d, a, b: a + b * d}
RATE_INDEX = {"exponential": 2, "power": 2, "logistic": 2}


def fit_gamma(d, y, se, start=None):
    """Gamma kernel by weighted least squares -> (params, level c, has a level)."""
    starts = [start] if start is not None else [[cc, H, dp, a] for cc in (1.0, 2.0, 2.4) for H in (0.5, 1.0)
                                                for dp in (20.0, 40.0, 100.0) for a in (0.2, 1.0, 3.0)]
    best = None
    for s0 in starts:
        s0 = [min(max(v, lo + 1e-9), hi - 1e-9) for v, lo, hi in zip(s0, GK_LB, GK_UB)]
        try:
            p, _ = curve_fit(gamma_kernel, d, y, p0=s0, sigma=se, bounds=(GK_LB, GK_UB), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(((gamma_kernel(d, *p) - y) / se) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p)
    if best is None:
        return None, None, False
    p = best[1]
    ok = GK_LB[0] + 1e-6 < p[0] < GK_UB[0] - 1e-6 and p[2] < GK_UB[2] - 1e-6
    return [float(v) for v in p], float(p[0]), bool(ok)


def fit_shape(shape, d, y, se, par, start=None):
    """Weighted least squares for one shape -> (params, level or None, levels off)."""
    if shape == "gamma kernel":
        return fit_gamma(d, y, se, start)
    lo_y, hi_y = RANGE[par]
    if shape == "linear":
        X = np.column_stack([np.ones_like(d), d])
        Wt = 1 / se ** 2
        coef = np.linalg.solve(X.T @ (X * Wt[:, None]), X.T @ (Wt * y))
        return [float(v) for v in coef], None, False
    span = float(y[-1] - y[0]) or 1e-3
    if shape == "exponential":
        bounds = ([lo_y, -50, 1e-3], [hi_y, 50, 3.0])
        starts = [start] if start is not None else [[y[-1] + span * 0.2, span, r] for r in (0.03, 0.1, 0.3)]
    elif shape == "power":
        bounds = ([lo_y, -50, 1e-3], [hi_y, 50, 5.0])
        starts = [start] if start is not None else [[y[-1] + span * 0.2, span, g] for g in (0.2, 0.6, 1.5)]
    else:
        bounds = ([lo_y, lo_y, 1e-3, -20.0], [hi_y, hi_y, 5.0, 60.0])
        starts = [start] if start is not None else [[y[0], y[-1] + span * 0.1, k, d0] for k in (0.2, 0.5) for d0 in (10, 18, 25)]
    best = None
    for s0 in starts:
        s0 = [min(max(v, lb + 1e-9), ub - 1e-9) for v, lb, ub in zip(s0, *bounds)]
        try:
            p, _ = curve_fit(SHAPE_F[shape], d, y, p0=s0, sigma=se, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum(((SHAPE_F[shape](d, *p) - y) / se) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p)
    if best is None:
        return None, None, False
    p = best[1]
    rate = p[RATE_INDEX[shape]]
    level = p[1] if shape == "logistic" else p[0]
    at_bound = rate <= 1.5e-3 or abs(level - lo_y) < 1e-6 or abs(level - hi_y) < 1e-6
    return [float(v) for v in p], float(level), not at_bound


# The depth functions of the paper. "s_logistic" is the S-curve alternative for s.
SHAPES = {"p0": "logistic", "middle": "logistic", "m": "logistic", "s": "gamma kernel", "s_logistic": "logistic"}


def data_key(q):
    return "s" if q == "s_logistic" else q


def fit_all(d, params, se, first=FIRST, starts=None):
    """Fit every depth function of SHAPES on depths first..25 -> {quantity: (params, level, has a level)}."""
    k = d >= first
    return {q: fit_shape(SHAPES[q], d[k], params[data_key(q)][k], se[data_key(q)][k], data_key(q),
                         None if starts is None else starts[q][0]) for q in SHAPES}


def levels(fits, s_key="s"):
    """Levels at infinite depth, or None if any curve has no level."""
    need = ("p0", "middle", "m", s_key)
    if any(fits[q][0] is None or not fits[q][2] for q in need):
        return None
    out = {"p0": fits["p0"][1], "middle": fits["middle"][1], "m": fits["m"][1], "s": fits[s_key][1]}
    out["p1"] = 1 - out["p0"] - out["middle"]
    return out


def curve_values(q, fits, dd, s_key="s"):
    """The fitted depth function of quantity q at depths dd (p1 from p0 and middle)."""
    if q == "p1":
        return 1 - SHAPE_F["logistic"](dd, *fits["p0"][0]) - SHAPE_F["logistic"](dd, *fits["middle"][0])
    key = s_key if q == "s" else q
    return SHAPE_F[SHAPES[key]](dd, *fits[key][0])


def setup(near_ties=True):
    """Stockfish run, errors, per-depth parameters, bootstrap resamples and standard errors."""
    run = c.load_stockfish()
    xs = run.x()
    d = np.array(run.levels, float)
    n_pos = len(run.pos_ids)
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n_pos, n_pos) for _ in range(B)]
    per_depth = errors_by_depth(run, xs, near_ties)
    full = parameters(per_depth, np.arange(n_pos))
    reps = [parameters(per_depth, idx) for idx in boots]
    se = {q: np.array([r[q] for r in reps]).std(axis=0, ddof=1) for q in QUANTS}
    return {"run": run, "xs": xs, "d": d, "boots": boots, "per_depth": per_depth, "full": full, "reps": reps, "se": se}


def limits(first=FIRST, near_ties=True):
    """setup() plus the depth functions fitted from depth `first`, on the full data and on every resample."""
    S = setup(near_ties)
    S["fits"] = fit_all(S["d"], S["full"], S["se"], first)
    S["boot_fits"] = [fit_all(S["d"], r, S["se"], first, starts=S["fits"]) for r in S["reps"]]
    S["first"] = first
    return S


def boot_levels(S, s_key="s"):
    """Infinite-depth levels of every resample in which every curve has a level."""
    return [L for L in (levels(bf, s_key) for bf in S["boot_fits"]) if L is not None]


# ---------------------------------------------------------------- game phase
def phase_summary(per_pos, n_legal, tolerances=(0.05, 0.10, 0.25, 0.5)):
    """Parameters and observed shares within each tolerance for one group of positions at one depth."""
    q = parameters([per_pos], np.arange(len(per_pos)))
    out = {k: float(v[0]) for k, v in q.items()}
    for t in tolerances:
        out[f"within {t}"] = float(np.mean([np.mean(e <= t + 1e-12) for e in per_pos]))
    out["legal moves"] = float(np.mean(n_legal))
    return out
