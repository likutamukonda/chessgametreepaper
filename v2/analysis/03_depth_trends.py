"""Step 3: how do the near-best quantities change with depth, and where do they head at infinite depth?

Quantities per engine, tolerance and level:
  mean_count   average number of near-best moves per position (no model)
  mean_share   average share of legal moves that are near-best (no model)
  mu, phi      beta-binomial mean and precision from step 2

Trend shapes, fitted against Stockfish depth d, or against t = log10(nodes) for Lc0:
  constant      y = L
  linear        y = a + b x            (keeps going: no finite limit; a benchmark)
  exponential   y = L - c exp(-k x)    (levels off quickly; for Lc0 this is a power law in nodes)
  power         y = L - c x^(-g)       (levels off slowly; Stockfish only, x = depth >= 1)

Training and testing: Stockfish depths 5-20 train and 21-25 are held out (depths 1-4 are mostly
quiescence search and behave erratically); Lc0 1-1,000 nodes train and 1,468-10,000 are held out.
The shape with the smallest held-out error is selected and refitted on all levels; its value at
infinite depth is the limit. Error bars: the whole refit repeated on positions resampled with
replacement (bootstrap). A limit is flagged as not identified when the fitted curve has not started
to level off (rate at its lower bound), because it then just extends a straight line.

Outputs: out/depth_trend_fits.csv (every shape), out/limits.csv (selected shape, limit, interval).
"""
import importlib
import json
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

import common as c

cd = importlib.import_module("02_count_distributions")
warnings.simplefilter("ignore", OptimizeWarning)

RATE_MIN = 1e-3
FORMS = {
    "constant": lambda x, L: np.full(np.shape(x), L, dtype=float),
    "linear": lambda x, a, b: a + b * x,
    "exponential": lambda x, L, cc, k: L - cc * np.exp(-k * x),
    "power": lambda x, L, cc, g: L - cc * np.power(x, -g),
}
LIMITED = ("constant", "exponential", "power")
B_FAST, B_BB, SEED = 1000, 200, 20261119


def fit_form(form, x, y, p0=None):
    """Least-squares fit; returns (params, limit or None, identified flag).
    p0: a starting point (used for the bootstrap refits, which start from the full-data fit)."""
    if form == "constant":
        return [float(y.mean())], float(y.mean()), True
    if form == "linear":
        b, a = np.polyfit(x, y, 1)
        return [float(a), float(b)], None, False
    f = FORMS[form]
    span = float(y[-1] - y[0]) or 1e-3
    if p0 is not None:
        starts = [list(p0)]
    else:
        starts = [[float(y[-1]), span, rate] for rate in ((0.05, 0.2, 1.0) if form == "exponential" else (0.2, 1.0, 3.0))]
    best = None
    for start in starts:
        start = [start[0], start[1], min(max(start[2], RATE_MIN * 1.01), 9.99)]
        try:
            p, _ = curve_fit(f, x, y, p0=start, bounds=([-1e3, -1e3, RATE_MIN], [1e3, 1e3, 10.0]), maxfev=5000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((f(x, *p) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, p)
    if best is None:
        return None, None, False
    p = best[1]
    identified = p[2] > RATE_MIN * 1.5 and abs(p[0]) < 999
    return [float(v) for v in p], float(p[0]), bool(identified)


def predict(form, params, x):
    return FORMS[form](x, *params)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


class Engine:
    """Per-level quantities for one engine, recomputable on resampled positions."""

    def __init__(self, run):
        self.run = run
        self.counts, self.n_legal = c.near_best_counts(run)
        if run.level_name == "depth":
            self.x = np.array(run.levels, float)
            self.use = self.x >= 5
            self.train = self.use & (self.x <= 20)
        else:
            self.x = np.log10(np.array(run.levels, float))
            self.use = np.ones(len(self.x), bool)
            self.train = np.array(run.levels) <= 1000
        self.valid = self.use & ~self.train

    def quantities(self, idx, which, start=None):
        """idx: indices of positions (a bootstrap sample or all) -> {quantity: array over levels}"""
        cnt, n = self.counts[idx], self.n_legal[idx]
        out = {}
        if "mean_count" in which:
            out["mean_count"] = cnt.mean(axis=0)                        # (levels, thresholds)
            out["mean_share"] = (cnt / n[:, None, None]).mean(axis=0)
        if "mu" in which:
            keep = n >= 2
            nn = n[keep] - 1
            z = np.log(n[keep]) - np.log(n[keep]).mean()
            mu = np.empty(cnt.shape[1:]); phi = np.empty(cnt.shape[1:])
            for li in range(cnt.shape[1]):
                if not self.use[li]:
                    mu[li] = phi[li] = np.nan
                    continue
                for ti in range(cnt.shape[2]):
                    k = cnt[keep, li, ti] - 1
                    th = cd.fit("beta-binomial", k, nn, z)[0] if start is None else self._fit_from(k, nn, z, start[li, ti])
                    mu[li, ti], phi[li, ti] = 1 / (1 + np.exp(-th[0])), np.exp(th[1])
            out["mu"], out["phi"] = mu, phi
        return out

    @staticmethod
    def _fit_from(k, n, z, theta0):
        from scipy import optimize
        res = optimize.minimize(lambda th: -cd.logpmf("beta-binomial", th, k, n, z).sum(), theta0,
                                method="L-BFGS-B", bounds=cd.BOUNDS["beta-binomial"])
        return res.x


def analyse(eng, qty_all, qname, ti, boot_series):
    x, y = eng.x, qty_all[qname][:, ti]
    forms = [f for f in FORMS if not (f == "power" and eng.run.level_name != "depth")]
    fits = {}
    for f in forms:
        params, lim_train, ident = fit_form(f, x[eng.train], y[eng.train])
        if params is None:
            continue
        fits[f] = {"params": params, "rmse_train": rmse(predict(f, params, x[eng.train]), y[eng.train]),
                   "rmse_valid": rmse(predict(f, params, x[eng.valid]), y[eng.valid]),
                   "limit_train": lim_train, "identified_train": ident}
        p_all, lim_all, ident_all = fit_form(f, x[eng.use], y[eng.use])
        fits[f].update({"params_all": p_all, "limit_all": lim_all, "identified_all": ident_all})
    selected = min(fits, key=lambda f: fits[f]["rmse_valid"])
    # bootstrap limits for every shape with a limit, refitted on all levels
    for f in fits:
        if f not in LIMITED:
            continue
        lims = []
        for ys in boot_series:
            p, lim, ident = fit_form(f, x[eng.use], ys[eng.use], p0=fits[f]["params_all"] if f != "constant" else None)
            if lim is not None and ident:
                lims.append(lim)
        fits[f]["boot"] = np.array(lims)
    return fits, selected


def main():
    rng = np.random.default_rng(SEED)
    fit_rows, limit_rows = [], []
    for run in (c.load_stockfish(), c.load_lc0()):
        eng = Engine(run)
        all_idx = np.arange(len(run.pos_ids))
        full = eng.quantities(all_idx, ("mean_count", "mu"))
        theta0 = np.stack([np.log(full["mu"] / (1 - full["mu"])), np.log(full["phi"])], axis=-1)
        boots_fast, boots_bb = [], []
        for b in range(B_FAST):
            idx = rng.integers(0, len(all_idx), len(all_idx))
            boots_fast.append(eng.quantities(idx, ("mean_count",)))
            if b < B_BB:
                boots_bb.append(eng.quantities(idx, ("mu",), start=theta0))
        top = len(run.levels) - 1
        for qname in ("mean_count", "mean_share", "mu", "phi"):
            boots = boots_fast if qname in ("mean_count", "mean_share") else boots_bb
            for ti, t in enumerate(c.THRESHOLDS):
                series = [bq[qname][:, ti] for bq in boots]
                fits, selected = analyse(eng, full, qname, ti, series)
                se_top = float(np.std([s[top] for s in series], ddof=1))
                for f, d in fits.items():
                    fit_rows.append([run.engine, t, qname, f, f == selected, json.dumps([round(v, 6) for v in d["params"]]),
                                     round(d["rmse_train"], 5), round(d["rmse_valid"], 5),
                                     "" if d["limit_train"] is None else round(d["limit_train"], 5),
                                     "" if d["limit_all"] is None else round(d["limit_all"], 5), d["identified_all"]])
                sel = fits[selected]
                finite = {f: d for f, d in fits.items() if f in LIMITED and d["identified_all"] and d["limit_all"] is not None}
                lo = hi = ""
                if selected in finite and len(sel.get("boot", [])) > 20:
                    lo, hi = np.percentile(sel["boot"], [2.5, 97.5])
                limit_rows.append([run.engine, t, qname, round(float(full[qname][top, ti]), 5), round(se_top, 5), selected,
                                   round(sel["rmse_valid"], 5),
                                   "" if selected not in finite else round(sel["limit_all"], 5),
                                   "" if lo == "" else round(float(lo), 5), "" if hi == "" else round(float(hi), 5),
                                   len(sel.get("boot", [])),
                                   "" if not finite else round(min(d["limit_all"] for d in finite.values()), 5),
                                   "" if not finite else round(max(d["limit_all"] for d in finite.values()), 5),
                                   ";".join(f"{f}={d['limit_all']:.4f}" for f, d in finite.items())])
    c.write_csv(c.OUT / "depth_trend_fits.csv", fit_rows,
                ["engine", "threshold", "quantity", "shape", "selected", "params_train", "rmse_train", "rmse_heldout",
                 "limit_train_fit", "limit_all_levels_fit", "levels_off"])
    c.write_csv(c.OUT / "limits.csv", limit_rows,
                ["engine", "threshold", "quantity", "value_at_top", "se_at_top", "selected_shape", "rmse_heldout",
                 "limit", "limit_lo95", "limit_hi95", "bootstrap_fits_used", "limit_min_over_shapes",
                 "limit_max_over_shapes", "limits_by_shape"])

    print("Limits at infinite depth (selected shape refitted on all levels; 95% bootstrap interval)")
    print(f"{'engine':<10}{'e<=':>5} {'quantity':<11}{'at top':>16} {'shape':>12} {'held-out err':>13} {'limit':>9} {'95% interval':>19} {'range over shapes':>20}")
    for r in limit_rows:
        lim = "not reached" if r[7] == "" else f"{r[7]:.3f}"
        ci = "" if r[8] == "" else f"[{r[8]:.3f}, {r[9]:.3f}]"
        rng_s = "" if r[11] == "" else f"{r[11]:.3f} - {r[12]:.3f}"
        print(f"{r[0]:<10}{r[1]:>5} {r[2]:<11}{r[3]:>9.3f}±{r[4]:.3f} {r[5]:>12} {r[6]:>13.4f} {lim:>9} {ci:>19} {rng_s:>20}")


if __name__ == "__main__":
    main()
