"""Step 19: the figures and tables of the paper (Stockfish, near-ties counted as ties).

Everything is recomputed from the run and the functions of steps 10-18, so the paper's numbers match the
analysis: the same positions, scale, weights (each position counts equally), bootstrap resamples and depth
functions (S-curves fitted on depths 5-25; gamma kernel for s, with the S-curve for s as the alternative).
Outputs in out/paper/:
  fig_distribution.pdf    the distribution at depths 5, 15 and 25: share within a tolerance, the error by
                          band, and the log-ratio of in-between moves against its normal fit
  fig_depth.pdf           p0, p1, the middle share, m, s and the shares within 0.05, 0.10 and 0.25 against
                          depth, with the fitted depth functions and the levels at infinite depth
  fig_infinite.pdf        the distribution at infinite depth against depth 25
  tab_models.tex, tab_parameters.tex, tab_shares.tex, tab_phase.tex   LaTeX tables (booktabs)
  (PNG copies of the figures for a quick look)
"""
import csv
import importlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit, ndtr

import common as c

warnings.simplefilter("ignore")
f15 = importlib.import_module("15_infinite_depth")
f16 = importlib.import_module("16_adjusted_infinite_depth")
f18 = importlib.import_module("18_phase_breakdown")
OUT = c.OUT / "paper"
OUT.mkdir(exist_ok=True)
GRID = f15.GRID
ALL = f15.ALL
TOL = (0.05, 0.10, 0.25, 0.5)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID_C = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
DEPTH_SHADES = {5: "#9cc2ee", 15: "#4f8fdc", 25: "#16457f"}
TOL_SHADES = {0.05: "#16457f", 0.10: "#2a78d6", 0.25: "#8db8ec"}
INF_X, CURVE_END = 70.0, 60.0
WIDTH = 6.3                                    # inches: the text width of the paper


def style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.linewidth": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
                         "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def model_curve(par, t=GRID):
    return f15.share_curve(par["p0"], par["middle"], par["m"], par["s"], t)


def observed_curve(per_pos, t=GRID):
    return np.mean([(e[None, :] <= np.asarray(t)[:, None] + 1e-12).mean(axis=1) for e in per_pos], axis=0)


def at(ser, li):
    return {q: float(ser[q][li]) for q in ALL}


# ---------------------------------------------------------------- figure 1
def fig_distribution(S):
    style()
    d, full, per_depth = S["d"], S["full"], S["per_depth"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    ax = axes[0]
    for dep, col in DEPTH_SHADES.items():
        li = list(d).index(dep)
        ax.plot(GRID, observed_curve(per_depth[li]), color=col, linewidth=1.4, label=f"depth {dep}")
        ax.plot(GRID, model_curve(at(full, li)), color=col, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.plot([], [], color=INK2, linewidth=0.9, linestyle=(0, (3, 2)), label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tolerance $t$")
    ax.set_ylabel("share of legal moves with $e \\leq t$")
    ax.set_title("(a) Share within a tolerance", loc="left")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    li = list(d).index(25)
    names = ["0"] + [f"{a:g}–{b:g}" for a, b in [e for _, e in f15.BINS[1:-1]]] + ["1"]
    x = np.arange(len(names))
    obs = f15.observed_bands(per_depth[li])
    mod = f15.band_probs(full["p0"][li], full["middle"][li], full["m"][li], full["s"][li])
    ax.bar(x, mod, width=0.72, color=BLUE, alpha=0.85, label="model", linewidth=0)
    ax.plot(x, obs, "D", color=INK, markersize=2.8, linestyle="none", label="observed")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.2)
    ax.set_xlabel("relative error $e$")
    ax.set_ylabel("share of legal moves")
    ax.set_title("(b) Depth 25, by band", loc="left")
    ax.legend(loc="upper center", frameon=False)

    ax = axes[2]
    errs = per_depth[li]
    z = np.concatenate([logit(e[(e > 0) & (e < 1)]) for e in errs])
    w = np.concatenate([np.full(np.sum((e > 0) & (e < 1)), 1.0 / len(e)) for e in errs])
    w = w / w.sum()
    bins = np.arange(-7.0, 7.01, 0.5)
    ax.hist(z, bins=bins, weights=w / 0.5, color=BLUE, alpha=0.85, label="observed", linewidth=0)
    zz = np.linspace(-7, 7, 400)
    m_, s_ = full["m"][li], full["s"][li]
    ax.plot(zz, np.exp(-0.5 * ((zz - m_) / s_) ** 2) / (s_ * np.sqrt(2 * np.pi)), color=INK, linewidth=1.1,
            label="normal fit")
    ax.set_xlabel("logit $e$")
    ax.set_ylabel("density")
    ax.set_title("(c) Depth 25, logit of $e$", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(0, 0.29)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_distribution")


# ---------------------------------------------------------------- figure 2
def draw_limit(ax, x, level, lo, hi, color):
    ax.plot([x, x], [lo, hi], color=color, linewidth=1.1)
    ax.plot([x, x], [lo, lo], marker="_", color=color, markersize=5, markeredgewidth=1.1)
    ax.plot([x, x], [hi, hi], marker="_", color=color, markersize=5, markeredgewidth=1.1)
    ax.plot(x, level, "o", color=color, markersize=4.2, markeredgecolor="white", markeredgewidth=0.8, zorder=5)


def fig_depth(S, inf, bl):
    style()
    d, full, se, fits = S["d"], S["full"], S["se"], S["fits"]
    inf_alt = f16.levels(fits, "s_logistic")
    bl_alt = [L for L in (f16.levels(bf, "s_logistic") for bf in S["boot_fits"]) if L is not None]
    dd = np.linspace(f16.FIRST, CURVE_END, 300)
    keep = d >= f16.FIRST
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.1))
    titles = {"p0": "(a) $p_0$: as good as the best", "p1": "(b) $p_1$: as bad as the worst",
              "middle": "(c) strictly between", "m": "(d) $m$: centre of the log-ratio", "s": "(e) $s$: spread of the log-ratio"}
    clip = {"p1": 0.30, "m": 1.2}
    for ax, q in zip(axes.flat, ("p0", "p1", "middle", "m", "s")):
        y, s_ = full[q], se[q]
        ax.axvspan(25, 64, color=GRID_C, alpha=0.45, linewidth=0)
        ax.errorbar(d[~keep], y[~keep], yerr=1.96 * s_[~keep], fmt="o", color=MUTED, markersize=2.2, elinewidth=0.6,
                    capsize=0, markerfacecolor="white", markeredgewidth=0.7, label="depths 1–4 (left out)")
        ax.errorbar(d[keep], y[keep], yerr=1.96 * s_[keep], fmt="o", color=INK2, markersize=2.2, elinewidth=0.6, capsize=0,
                    label="depths 5–25 (95% band)")
        yy = f16.curve_values(q, fits, dd)
        ax.plot(dd[dd <= 25], yy[dd <= 25], color=BLUE, linewidth=1.3, label="depth function")
        ax.plot(dd[dd >= 25], yy[dd >= 25], color=BLUE, linewidth=0.9, linestyle=(0, (2, 1.5)))
        if q == "p0":
            ax.plot([], [], "o-", color=ORANGE, markersize=3.5, linewidth=1.0, label="infinite depth (95% interval)")
            ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
        lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
        top = clip.get(q)
        if q == "s":
            ya = f16.curve_values("s", fits, dd, "s_logistic")
            ax.plot(dd, ya, color=INK2, linewidth=0.9, linestyle=(0, (4, 2)))
            lo_a, hi_a = np.percentile([L["s"] for L in bl_alt], [2.5, 97.5])
            draw_limit(ax, INF_X - 3, inf_alt["s"], lo_a, hi_a, INK2)
            draw_limit(ax, INF_X + 1, inf[q], lo, hi, ORANGE)
            ax.text(INF_X - 4.5, inf_alt["s"], "S", ha="right", va="center", fontsize=6.3, color=INK2)
            ax.text(INF_X + 2.6, inf[q], "γ", ha="left", va="center", fontsize=6.8, color=ORANGE)
        else:
            shown_hi = min(hi, top) if top else hi
            draw_limit(ax, INF_X, inf[q], lo, shown_hi, ORANGE)
            if top and hi > top:
                ax.plot(INF_X, shown_hi, marker="^", color=ORANGE, markersize=4.2)
                ax.text(INF_X - 2.5, shown_hi, f"to {hi:.2f}", ha="right", va="top", fontsize=6.2, color=INK2)
        ax.set_xlim(0, INF_X + 5)
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60, INF_X])
        ax.set_xticklabels(["1", "10", "20", "30", "40", "50", "60", "∞"])
        ax.set_title(titles[q], loc="left")
        if top:
            ax.set_ylim(top=top * 1.04)
    axes[0, 1].set_ylim(0.04, 0.31)
    axes[1, 0].set_ylim(-0.55, 1.25)
    axes[0, 0].set_ylim(0.04, 0.30)
    ax = axes[1, 2]
    for k, (t, col) in enumerate(TOL_SHADES.items()):
        obs = np.array([np.mean([np.mean(e <= t + 1e-12) for e in S["per_depth"][li]]) for li in range(len(d))])
        ax.plot(d[~keep], obs[~keep], "o", color=col, markersize=2.2, markerfacecolor="white", markeredgewidth=0.7)
        ax.plot(d[keep], obs[keep], "o", color=col, markersize=2.2)
        cur = np.array([f15.share_curve(*(f16.curve_values(q, S["fits"], np.array([x]))[0] for q in ("p0", "middle", "m", "s")),
                                        t=np.array([t]))[0] for x in dd])
        ax.plot(dd[dd <= 25], cur[dd <= 25], color=col, linewidth=1.2, label=f"$t$ = {t:g}")
        ax.plot(dd[dd >= 25], cur[dd >= 25], color=col, linewidth=0.8, linestyle=(0, (2, 1.5)))
        vals = [f15.share_curve(L["p0"], L["middle"], L["m"], L["s"], np.array([t]))[0] for L in bl]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        draw_limit(ax, INF_X - 3 + 3 * k, model_curve(inf, np.array([t]))[0], lo, hi, col)
    ax.axvspan(25, 64, color=GRID_C, alpha=0.45, linewidth=0)
    ax.set_xlim(0, INF_X + 5)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, INF_X])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "50", "60", "∞"])
    ax.set_ylim(0.1, 0.5)
    ax.set_title("(f) share within $t$", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=1.2, columnspacing=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Stockfish depth")
    fig.tight_layout(h_pad=0.8, w_pad=0.9)
    save(fig, "fig_depth")


# ---------------------------------------------------------------- figure 3
def fig_infinite(S, inf, bl):
    style()
    d, full, per_depth = S["d"], S["full"], S["per_depth"]
    inf_alt = f16.levels(S["fits"], "s_logistic")
    li = list(d).index(25)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    Fb = np.array([model_curve(L) for L in bl])
    lo, hi = np.percentile(Fb, [2.5, 97.5], axis=0)
    ax.fill_between(GRID, lo, hi, color=ORANGE, alpha=0.16, linewidth=0)
    ax.plot(GRID, model_curve(inf), color=ORANGE, linewidth=1.6, label="infinite depth, gamma kernel for $s$")
    ax.plot(GRID, model_curve(inf_alt), color=ORANGE, linewidth=0.9, linestyle=(0, (4, 2)), label="infinite depth, S-curve for $s$")
    ax.plot(GRID, model_curve(at(full, li)), color=BLUE, linewidth=1.2, label="depth 25, model")
    ax.plot(GRID, observed_curve(per_depth[li]), color=INK, linewidth=0.8, linestyle=(0, (1, 1.2)), label="depth 25, observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tolerance $t$")
    ax.set_ylabel("share of legal moves with $e \\leq t$")
    ax.set_title("(a) Share within a tolerance", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax = axes[1]
    names = ["0"] + [f"{a:g}–{b:g}" for a, b in [e for _, e in f15.BINS[1:-1]]] + ["1"]
    x = np.arange(len(names))
    b25 = f15.band_probs(full["p0"][li], full["middle"][li], full["m"][li], full["s"][li])
    binf = f15.band_probs(inf["p0"], inf["middle"], inf["m"], inf["s"])
    bb = np.array([f15.band_probs(L["p0"], L["middle"], L["m"], L["s"]) for L in bl])
    blo, bhi = np.percentile(bb, [2.5, 97.5], axis=0)
    cap = 0.49
    ax.bar(x - 0.2, b25, width=0.38, color=BLUE, label="depth 25, model", linewidth=0)
    ax.bar(x + 0.2, binf, width=0.38, color=ORANGE, label="infinite depth (95% interval)", linewidth=0)
    ax.errorbar(x + 0.2, binf, yerr=[binf - blo, np.minimum(bhi, cap) - binf], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
    for xi, h in zip(x + 0.2, bhi):
        if h > cap:
            ax.plot(xi, cap, marker="^", color=INK2, markersize=3.5)
            ax.text(xi - 0.25, cap, f"to {h:.2f}", ha="right", va="center", fontsize=6.2, color=INK2)
    ax.plot(x - 0.2, f15.observed_bands(per_depth[li]), "D", color=INK, markersize=2.6, linestyle="none", label="depth 25, observed")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.2)
    ax.set_xlabel("relative error $e$")
    ax.set_ylabel("share of legal moves")
    ax.set_ylim(0, 0.52)
    ax.set_title("(b) By band", loc="left")
    ax.legend(loc="upper center", frameon=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_infinite")


# ---------------------------------------------------------------- tables
def fmt(v, nd=3):
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$") if v < 0 else s


def tab_models():
    rows = []
    for f in ("error_distribution_fits.csv", "error_distribution_fits_snapped.csv"):
        rs = [r for r in csv.DictReader(open(c.OUT / f)) if r["engine"] == "Stockfish"]
        single = [m for m in dict.fromkeys(r["model"] for r in rs) if "benchmark" not in m]
        wins = {m: 0 for m in dict.fromkeys(r["model"] for r in rs)}
        for lv in sorted({int(r["level"]) for r in rs}):
            best = min((r for r in rs if int(r["level"]) == lv and r["model"] in single), key=lambda r: float(r["aic"]))
            wins[best["model"]] += 1
        gaps = {m: float(np.median([float(r["largest_gap"]) for r in rs if r["model"] == m])) for m in wins}
        k = {m: int(next(r["n_params"] for r in rs if r["model"] == m)) for m in wins}
        rows.append((wins, gaps, k))
    order = ["ZOI logit-normal", "ZOI beta", "ordered beta", "ZOI Kumaraswamy", "censored normal", "censored logistic",
             "two-beta mixture (benchmark)"]
    label = {"ZOI logit-normal": "Zero-one-inflated logit-normal", "ZOI beta": "Zero-one-inflated beta",
             "ordered beta": "Ordered beta$^{a}$", "ZOI Kumaraswamy": "Zero-one-inflated Kumaraswamy",
             "censored normal": "Censored normal", "censored logistic": "Censored logistic",
             "two-beta mixture (benchmark)": "Two-beta mixture (benchmark)$^{b}$"}
    lines = [r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lcrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Lowest AIC (depths)} & \multicolumn{2}{c}{Median largest gap} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & $k$ & Measured & Near-ties & Measured & Near-ties \\", r"\midrule"]
    for m in order:
        (w0, g0, k0), (w1, g1, _) = rows
        wins0 = "--" if "benchmark" in m else str(w0[m])
        wins1 = "--" if "benchmark" in m else str(w1[m])
        if "benchmark" in m:
            lines.append(r"\midrule")
        lines.append(f"{label[m]} & {k0[m]} & {wins0} & {wins1} & {g0[m]:.3f} & {g1[m]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_models.tex").write_text("\n".join(lines) + "\n")
    return rows


def tab_parameters(S, inf, bl):
    d, full, se = S["d"], S["full"], S["se"]
    inf_alt = f16.levels(S["fits"], "s_logistic")
    bl_alt = [L for L in (f16.levels(bf, "s_logistic") for bf in S["boot_fits"]) if L is not None]
    i5, i25 = list(d).index(5), list(d).index(25)
    names = {"p0": r"$p_0$ (as good as the best)", "p1": r"$p_1$ (as bad as the worst)",
             "middle": r"$1-p_0-p_1$ (strictly between)", "m": r"$m$ (centre of the log-ratio)",
             "s": r"$s$ (spread of the log-ratio)"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Parameter & Depth 5 & Depth 25 & Infinite depth [95\% interval] \\", r"\midrule"]
    for q in ("p0", "p1", "middle", "m", "s"):
        lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
        tail = f"{fmt(inf[q])} [{fmt(lo)}, {fmt(hi)}]"
        if q == "s":
            tail += r"$^{a}$"
        lines.append(f"{names[q]} & {fmt(full[q][i5])} ({se[q][i5]:.3f}) & {fmt(full[q][i25])} ({se[q][i25]:.3f}) & {tail} \\\\")
    lo, hi = np.percentile([L["s"] for L in bl_alt], [2.5, 97.5])
    lines.append(rf"\quad $s$ with the S-curve instead & & & {fmt(inf_alt['s'])} [{fmt(lo)}, {fmt(hi)}] \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_parameters.tex").write_text("\n".join(lines) + "\n")


def tab_shares(S, inf, bl):
    d, per_depth, reps_idx = S["d"], S["per_depth"], S["boots"]
    inf_alt = f16.levels(S["fits"], "s_logistic")
    bl_alt = [L for L in (f16.levels(bf, "s_logistic") for bf in S["boot_fits"]) if L is not None]
    i5, i25 = list(d).index(5), list(d).index(25)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Observed} & Model & \multicolumn{2}{c}{Infinite depth [95\% interval]} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}",
             r"Tolerance $t$ & Depth 5 & Depth 25 & Depth 25 & Gamma kernel for $s$ & S-curve for $s$ \\", r"\midrule"]
    out = {}
    for t in TOL:
        obs = {}
        for li in (i5, i25):
            per = per_depth[li]
            sh = np.array([np.mean(e <= t + 1e-12) for e in per])
            obs[li] = (sh.mean(), np.std([sh[idx].mean() for idx in reps_idx], ddof=1))
        mod25 = model_curve(at(S["full"], i25), np.array([t]))[0]
        cells = []
        for L0, BL in ((inf, bl), (inf_alt, bl_alt)):
            v = model_curve(L0, np.array([t]))[0]
            lo, hi = np.percentile([model_curve(L, np.array([t]))[0] for L in BL], [2.5, 97.5])
            cells.append(f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
        out[t] = (obs[i5], obs[i25], mod25, cells)
        lines.append(f"{t:.2f} & {obs[i5][0]:.3f} ({obs[i5][1]:.3f}) & {obs[i25][0]:.3f} ({obs[i25][1]:.3f}) & {mod25:.3f} & "
                     f"{cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_shares.tex").write_text("\n".join(lines) + "\n")
    return out


def tab_phase(S):
    run, xs = S["run"], S["xs"]
    snap = lambda e: np.where(e < f16.SNAP, 0.0, np.where(e > 1 - f16.SNAP, 1.0, e))
    pieces = np.array([c.piece_count(run.fens[p]) for p in run.pos_ids])
    n_legal = np.array([xs[p].shape[1] for p in run.pos_ids])
    rng = np.random.default_rng(f18.SEED)
    li = run.levels.index(25)
    cols = []
    for name, lo, hi in f18.GROUPS + (("all", 0, 32),):
        idx = np.flatnonzero((pieces >= lo) & (pieces <= hi))
        # all positions: the resamples used everywhere else, so the standard errors match Tables 2 and 3
        boots = S["boots"] if name == "all" else [rng.choice(idx, len(idx), replace=True) for _ in range(f18.B)]
        errs = {i: snap(c.relative_errors(xs[run.pos_ids[i]][li:li + 1])[0]) for i in idx}
        full = f18.summary([errs[i] for i in idx], n_legal[idx])
        reps = [f18.summary([errs[i] for i in b], n_legal[b]) for b in boots]
        se = {q: float(np.std([r[q] for r in reps], ddof=1)) for q in full}
        cols.append((name, len(idx), full, se))
    rows = [("legal moves", "Legal moves (mean)", 1), ("p0", r"$p_0$", 3), ("p1", r"$p_1$", 3),
            ("middle", r"$1-p_0-p_1$", 3), ("m", r"$m$", 2), ("s", r"$s$", 2),
            ("within 0.05", r"Share within 0.05", 3), ("within 0.1", r"Share within 0.10", 3),
            ("within 0.25", r"Share within 0.25", 3), ("within 0.5", r"Share within 0.5", 3)]
    head = " & ".join([r"12 or fewer", r"13--22", r"23 or more", r"All"])
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", f"Pieces on the board & {head} \\\\", r"\midrule",
             "Positions & " + " & ".join(str(n) for _, n, _, _ in cols) + r" \\"]
    for key, label, nd in rows:
        cells = [f"{fmt(full[key], nd)} ({se[key]:.{nd}f})" for _, _, full, se in cols]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_phase.tex").write_text("\n".join(lines) + "\n")
    return cols


def main():
    S = f16.stockfish_limits()
    inf = f16.levels(S["fits"], "s")
    bl = [L for L in (f16.levels(bf, "s") for bf in S["boot_fits"]) if L is not None]
    fig_distribution(S)
    fig_depth(S, inf, bl)
    fig_infinite(S, inf, bl)
    models = tab_models()
    tab_parameters(S, inf, bl)
    shares = tab_shares(S, inf, bl)
    phase = tab_phase(S)
    print(f"resamples used for the infinite-depth intervals: {len(bl)} of {f16.B}")
    print("model comparison (wins as measured / near-ties; median largest gap):")
    for m in models[0][0]:
        print(f"  {m:<30} {models[0][0][m]:>3} {models[1][0][m]:>3}   {models[0][1][m]:.3f} {models[1][1][m]:.3f}")
    print("shares within t: observed d5 (se), observed d25 (se), model d25, infinite (gamma), infinite (S-curve)")
    for t, (o5, o25, m25, cells) in shares.items():
        print(f"  t={t}: {o5[0]:.3f} ({o5[1]:.3f})  {o25[0]:.3f} ({o25[1]:.3f})  {m25:.3f}  {cells[0]}  {cells[1]}")
    print("phase at depth 25:")
    for name, n, full, se in phase:
        print(f"  {name:<12} n={n:>3}  " + "  ".join(f"{k} {v:.3f}" for k, v in full.items()))
    print("files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
