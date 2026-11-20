"""10. The figures and tables of the paper (Stockfish, near-ties counted as ties).

Run scripts 01-09 first: the tables read their outputs (out/*.csv), so every number in the paper traces back to one
script. The figures and the remaining tables are computed here with model.py (same positions, scale, weights,
bootstrap resamples and depth functions as everywhere else).
Outputs in out/paper/:
  fig_distribution.pdf      (a) share of moves within a tolerance at depths 5, 15 and 25; (b) depth 25 by band;
                            (c) depth 25, logit e against its normal fit; (d) observed minus fitted share by band
  fig_depth.pdf             p0, p1, the middle share, m, s and the shares within 0.05, 0.10 and 0.25 against depth,
                            with the depth functions and the levels at infinite depth
  fig_infinite.pdf          the distribution at infinite depth against depth 25
  tab_models.tex            candidate distributions (from 01)
  tab_fit.tex               goodness of fit at selected depths (from 07)
  tab_parameters.tex        the parameters at depths 5 and 25 and at infinite depth
  tab_depth_functions.tex   the depth functions against a straight line, and the change from depth 20 to 25 (from 07)
  tab_correlations.tex      rank correlations between the estimates and between the infinite-depth levels (from 08)
  tab_shares.tex            share of moves within a tolerance
  tab_phase.tex             depth 25 by game phase, with p0 and p1 split into their parts (from 06 and 07)
  (PNG copies of the figures for a quick look)
"""
import csv
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import logit
from scipy.stats import norm

import common as c
import model as mdl

warnings.simplefilter("ignore")
OUT = c.OUT / "paper"
GRID = mdl.GRID
ORDER = ("p0", "p1", "middle", "m", "s")
TOL = (0.05, 0.10, 0.25, 0.5)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID_C, ERR_C = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df", "#b4b2ac"
DEPTH_SHADES = {5: "#9cc2ee", 15: "#4f8fdc", 25: "#16457f"}
TOL_SHADES = {0.05: "#16457f", 0.10: "#2a78d6", 0.25: "#8db8ec"}
DASH, DOT = (0, (3.5, 2)), (0, (1.5, 1.5))
INF_X, SHADE_END, CURVE_END = 70.0, 64.0, 60.0
WIDTH = 6.27                                   # inches: the text width of the paper (A4, 1 inch margins)


def style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.linewidth": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                         "axes.titlepad": 5, "pdf.fonttype": 42})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def idx(d, dep):
    return list(d).index(dep)


def band_names():
    return ["0"] + [f"{a:g}–{b:g}" for _, (a, b) in mdl.BINS[1:-1]] + ["1"]


def depth_axis(ax):
    ax.axvspan(25, SHADE_END, color=GRID_C, alpha=0.45, linewidth=0)
    ax.set_xlim(0, INF_X + 5)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, INF_X])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "50", "60", "∞"])


def draw_limit(ax, x, level, lo, hi, color, top=None):
    """The level at infinite depth with its 95% interval; an interval running past `top` is cut there and labelled."""
    shown = min(hi, top) if top is not None else hi
    ax.plot([x, x], [lo, shown], color=color, linewidth=1.2)
    ax.plot(x, lo, marker="_", color=color, markersize=6, markeredgewidth=1.2)
    if top is not None and hi > top:
        ax.plot(x, shown, marker="^", color=color, markersize=5)
        ax.text(x - 2.2, shown, f"to {hi:.2f}", ha="right", va="top", fontsize=7.4, color=INK2)
    else:
        ax.plot(x, hi, marker="_", color=color, markersize=6, markeredgewidth=1.2)
    ax.plot(x, level, "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=5)


# ---------------------------------------------------------------- figure 1
def fig_distribution(S):
    style()
    d, full, per_depth, boots = S["d"], S["full"], S["per_depth"], S["boots"]
    li25 = idx(d, 25)
    names = band_names()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.7))

    ax = axes[0, 0]
    for dep, col in DEPTH_SHADES.items():
        li = idx(d, dep)
        ax.plot(GRID, mdl.observed_curve(per_depth[li]), color=col, linewidth=1.7, label=f"depth {dep}")
        ax.plot(GRID, mdl.model_curve(mdl.at(full, li)), color=col, linewidth=1.1, linestyle=DASH)
    ax.plot([], [], color=INK2, linewidth=1.7, label="observed")
    ax.plot([], [], color=INK2, linewidth=1.1, linestyle=DASH, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tolerance $t$")
    ax.set_ylabel("share of legal moves with $e \\leq t$")
    ax.set_title("(a) Share of moves within a tolerance", loc="left")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[0, 1]
    x = np.arange(len(names))
    obs = mdl.observed_bands(per_depth[li25])
    mod = mdl.band_probs(full["p0"][li25], full["middle"][li25], full["m"][li25], full["s"][li25])
    ax.bar(x, mod, width=0.72, color=BLUE, alpha=0.85, label="model", linewidth=0)
    ax.plot(x, obs, "D", color=INK, markersize=3.8, linestyle="none", label="observed")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_xlabel("relative error $e$")
    ax.set_ylabel("share of legal moves")
    ax.set_title("(b) Depth 25, by band of $e$", loc="left")
    ax.legend(loc="upper center", frameon=False)

    ax = axes[1, 0]
    errs = per_depth[li25]
    z = np.concatenate([logit(e[(e > 0) & (e < 1)]) for e in errs])
    w = np.concatenate([np.full(np.sum((e > 0) & (e < 1)), 1.0 / len(e)) for e in errs])
    w = w / w.sum()
    bins = np.arange(-7.0, 7.01, 0.5)
    ax.hist(z, bins=bins, weights=w / 0.5, color=BLUE, alpha=0.85, label="observed", linewidth=0)
    zz = np.linspace(-7.5, 7.5, 400)
    m_, s_ = full["m"][li25], full["s"][li25]
    ax.plot(zz, norm.pdf(zz, m_, s_), color=INK, linewidth=1.3, label=f"normal ($m$ = {m_:.2f}, $s$ = {s_:.2f})")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(0, 0.275)
    ax.set_xlabel("logit $e$ = log(distance to best / to worst)")
    ax.set_ylabel("density")
    ax.set_title("(c) Depth 25, logit $e$ of in-between moves", loc="left")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1, 1]
    xm = np.arange(10)
    for k, (dep, col) in enumerate(DEPTH_SHADES.items()):
        li = idx(d, dep)
        diff = mdl.band_diff(per_depth[li])[1:11]
        se = np.array([mdl.band_diff([per_depth[li][i] for i in b])[1:11] for b in boots]).std(axis=0, ddof=1)
        ax.errorbar(xm + (k - 1) * 0.24, diff, yerr=1.96 * se, fmt="o", color=col, markersize=3.6, elinewidth=1.0,
                    capsize=0, markeredgecolor=INK2 if dep == 5 else col, markeredgewidth=0.5, label=f"depth {dep}")
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks(xm)
    ax.set_xticklabels(names[1:11], rotation=50, ha="right", rotation_mode="anchor")
    ax.set_xlabel("relative error $e$")
    ax.set_ylabel("observed $-$ model share")
    ax.set_title("(d) Observed minus model, by band", loc="left")
    ax.legend(loc="upper center", frameon=False, handletextpad=0.3)
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    save(fig, "fig_distribution")


# ---------------------------------------------------------------- figure 2
def fig_depth(S, inf, bl, inf_alt, bl_alt):
    style()
    d, full, se, fits, per_depth = S["d"], S["full"], S["se"], S["fits"], S["per_depth"]
    dd = np.linspace(mdl.FIRST, CURVE_END, 300)
    keep = d >= mdl.FIRST
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 8.0))
    panels = {"p0": ("(a) $p_0$: moves as good as the best", "share of legal moves", None),
              "p1": ("(b) $p_1$: moves as bad as the worst", "share of legal moves", 0.30),
              "middle": ("(c) $1-p_0-p_1$: moves strictly between", "share of legal moves", None),
              "m": ("(d) $m$: centre of logit $e$", "$m$", 1.2),
              "s": ("(e) $s$: spread of logit $e$", "$s$", None)}
    for ax, (q, (title, ylabel, top)) in zip(axes.flat, panels.items()):
        y, s_ = full[q], se[q]
        depth_axis(ax)
        ax.errorbar(d[~keep], y[~keep], yerr=1.96 * s_[~keep], fmt="o", color=MUTED, ecolor=ERR_C, markersize=3,
                    elinewidth=0.9, capsize=0, markerfacecolor="white", markeredgewidth=0.8)
        ax.errorbar(d[keep], y[keep], yerr=1.96 * s_[keep], fmt="o", color=INK2, ecolor=ERR_C, markersize=2.8,
                    elinewidth=0.9, capsize=0)
        yy = mdl.curve_values(q, fits, dd)
        ax.plot(dd[dd <= 25], yy[dd <= 25], color=BLUE, linewidth=1.8, zorder=4)
        ax.plot(dd[dd >= 25], yy[dd >= 25], color=BLUE, linewidth=1.2, linestyle=DOT, zorder=4)
        lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
        if q == "s":
            ya = mdl.curve_values("s", fits, dd, "s_logistic")
            ax.plot(dd[dd <= 25], ya[dd <= 25], color=INK, linewidth=1.2, linestyle=DASH, zorder=4)
            ax.plot(dd[dd >= 25], ya[dd >= 25], color=INK, linewidth=1.0, linestyle=DOT, zorder=4)
            lo_a, hi_a = np.percentile([L["s"] for L in bl_alt], [2.5, 97.5])
            draw_limit(ax, INF_X - 2.5, inf_alt["s"], lo_a, hi_a, INK2)
            draw_limit(ax, INF_X + 2.5, inf["s"], lo, hi, ORANGE)
            ax.legend(handles=[Line2D([], [], color=BLUE, linewidth=1.8, label="gamma kernel (orange level)"),
                               Line2D([], [], color=INK, linewidth=1.2, linestyle=DASH, label="S-curve (grey level)")],
                      loc="lower right", frameon=False)
        else:
            draw_limit(ax, INF_X, inf[q], lo, hi, ORANGE, top)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
    ax = axes[2, 1]
    depth_axis(ax)
    for k, (t, col) in enumerate(TOL_SHADES.items()):
        obs = np.array([np.mean([np.mean(e <= t + 1e-12) for e in per_depth[li]]) for li in range(len(d))])
        ax.plot(d[~keep], obs[~keep], "o", color=col, markersize=3, markerfacecolor="white", markeredgewidth=0.8)
        ax.plot(d[keep], obs[keep], "o", color=col, markersize=3)
        cur = np.array([mdl.share_curve(*(mdl.curve_values(q, fits, np.array([x]))[0] for q in ("p0", "middle", "m", "s")),
                                        t=np.array([t]))[0] for x in dd])
        ax.plot(dd[dd <= 25], cur[dd <= 25], color=col, linewidth=1.6, label=f"$t$ = {t:.2f}", zorder=4)
        ax.plot(dd[dd >= 25], cur[dd >= 25], color=col, linewidth=1.1, linestyle=DOT, zorder=4)
        lo, hi = np.percentile([mdl.model_curve(L, np.array([t]))[0] for L in bl], [2.5, 97.5])
        draw_limit(ax, INF_X - 3 + 3 * k, mdl.model_curve(inf, np.array([t]))[0], lo, hi, col)
    ax.set_title("(f) Share of moves within $t$", loc="left")
    ax.set_ylabel("share of legal moves with $e \\leq t$")
    ax.set_ylim(0.1, 0.5)
    ax.legend(loc="upper center", frameon=False, handlelength=1.6)
    for ax in axes[2]:
        ax.set_xlabel("Stockfish search depth")
    handles = [Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", markersize=4,
                      label="depths 1–4 (not fitted)"),
               Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=4, label="depths 5–25 (bars: 95% band)"),
               Line2D([], [], color=BLUE, linewidth=1.8, label="depth function"),
               Line2D([], [], color=BLUE, linewidth=1.2, linestyle=DOT, label="extrapolated (shaded)"),
               Line2D([], [], color=ORANGE, marker="o", linewidth=1.2, markersize=5, label="infinite depth, 95% interval")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6, handlelength=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.3, w_pad=1.6)
    save(fig, "fig_depth")


# ---------------------------------------------------------------- figure 3
def fig_infinite(S, inf, bl, inf_alt):
    style()
    d, full, per_depth = S["d"], S["full"], S["per_depth"]
    li25 = idx(d, 25)
    names = band_names()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4))
    ax = axes[0]
    Fb = np.array([mdl.model_curve(L) for L in bl])
    lo, hi = np.percentile(Fb, [2.5, 97.5], axis=0)
    ax.fill_between(GRID, lo, hi, color=ORANGE, alpha=0.16, linewidth=0)
    ax.plot(GRID, mdl.model_curve(inf), color=ORANGE, linewidth=1.8, label="infinite depth (95% band)")
    ax.plot(GRID, mdl.model_curve(inf_alt), color=ORANGE, linewidth=1.1, linestyle=DASH, label="infinite depth, S-curve for $s$")
    ax.plot(GRID, mdl.model_curve(mdl.at(full, li25)), color=BLUE, linewidth=1.4, label="depth 25, model")
    ax.plot(GRID, mdl.observed_curve(per_depth[li25]), color=INK, linewidth=1.0, linestyle=DOT, label="depth 25, observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tolerance $t$")
    ax.set_ylabel("share of legal moves with $e \\leq t$")
    ax.set_title("(a) Share of moves within a tolerance", loc="left")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    x = np.arange(len(names))
    b25 = mdl.band_probs(full["p0"][li25], full["middle"][li25], full["m"][li25], full["s"][li25])
    binf = mdl.band_probs(inf["p0"], inf["middle"], inf["m"], inf["s"])
    bb = np.array([mdl.band_probs(L["p0"], L["middle"], L["m"], L["s"]) for L in bl])
    blo, bhi = np.percentile(bb, [2.5, 97.5], axis=0)
    cap = 0.40
    ax.bar(x - 0.2, b25, width=0.38, color=BLUE, label="depth 25, model", linewidth=0)
    ax.plot(x - 0.2, mdl.observed_bands(per_depth[li25]), "D", color=INK, markersize=3.2, linestyle="none",
            label="depth 25, observed")
    ax.bar(x + 0.2, binf, width=0.38, color=ORANGE, label="infinite depth", linewidth=0)
    ax.errorbar(x + 0.2, binf, yerr=[binf - blo, np.minimum(bhi, cap) - binf], fmt="none", ecolor=INK2, elinewidth=0.8,
                capsize=1.8)
    for xi, h in zip(x + 0.2, bhi):
        if h > cap:
            ax.plot(xi, cap, marker="^", color=INK2, markersize=4)
            ax.text(xi - 0.14, cap - 0.004, f"to {h:.2f}", ha="right", va="top", rotation=90, fontsize=7.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_xlabel("relative error $e$")
    ax.set_ylabel("share of legal moves")
    ax.set_ylim(0, cap * 1.06)
    ax.set_title("(b) By band of $e$", loc="left")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_infinite")


# ---------------------------------------------------------------- tables
def fmt(v, nd=3):
    s = f"{v:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def read(name):
    with open(c.OUT / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n")


def tab_models():
    rs_all = read("candidate_distributions.csv")
    order = ["ZOI logit-normal", "ZOI beta", "ordered beta", "ZOI Kumaraswamy", "censored normal", "censored logistic",
             "two-beta mixture (benchmark)"]
    label = {"ZOI logit-normal": "Zero-one-inflated logit-normal", "ZOI beta": "Zero-one-inflated beta",
             "ordered beta": "Ordered beta$^{a}$", "ZOI Kumaraswamy": "Zero-one-inflated Kumaraswamy",
             "censored normal": "Censored normal", "censored logistic": "Censored logistic",
             "two-beta mixture (benchmark)": "Two-beta mixture (benchmark)$^{b}$"}
    summary = {}
    for version in ("measured", "near-ties"):
        rs = [r for r in rs_all if r["version"] == version]
        wins = {m: 0 for m in order}
        for dep in sorted({int(r["depth"]) for r in rs}):
            single = [r for r in rs if int(r["depth"]) == dep and "benchmark" not in r["model"]]
            wins[min(single, key=lambda r: float(r["aic"]))["model"]] += 1
        gaps = {m: float(np.median([float(r["largest_gap"]) for r in rs if r["model"] == m])) for m in order}
        k = {m: int(next(r["n_params"] for r in rs if r["model"] == m)) for m in order}
        summary[version] = (wins, gaps, k)
    lines = [r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lcrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Lowest AIC (depths)} & \multicolumn{2}{c}{Median largest gap} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", r"Model & $k$ & Measured & Near-ties & Measured & Near-ties \\",
             r"\midrule"]
    (w0, g0, k0), (w1, g1, _) = summary["measured"], summary["near-ties"]
    for m in order:
        bench = "benchmark" in m
        if bench:
            lines.append(r"\midrule")
        lines.append(f"{label[m]} & {k0[m]} & {'--' if bench else w0[m]} & {'--' if bench else w1[m]} & "
                     f"{g0[m]:.3f} & {g1[m]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_models.tex", lines)
    return summary


def tab_fit():
    rows = {int(r["depth"]): r for r in read("fit_by_depth.csv")}
    lines = [r"\begin{tabular}{rcccccc}", r"\toprule",
             r"Depth & Largest gap & Average gap & Share misplaced & Band $R^2$ & Wald $W$ & $p$ \\", r"\midrule"]
    for dep in (5, 10, 15, 20, 25):
        r = rows[dep]
        p = float(r["p_exact_fit"])
        mant, ex = f"{p:.1e}".split("e")
        lines.append(f"{dep} & {float(r['largest_gap']):.3f} & {float(r['average_gap']):.3f} & "
                     f"{float(r['share_misplaced']):.3f} & {float(r['band_r2']):.2f} & {float(r['wald_w']):.0f} & "
                     f"${mant}\\times 10^{{{int(ex)}}}$ \\\\")
    sel = [r for d_, r in rows.items() if d_ >= mdl.FIRST]
    rng = lambda key, nd: f"{min(float(r[key]) for r in sel):.{nd}f}--{max(float(r[key]) for r in sel):.{nd}f}"
    lines += [r"\midrule", f"5--25 & {rng('largest_gap', 3)} & {rng('average_gap', 3)} & {rng('share_misplaced', 3)} & "
              f"{rng('band_r2', 2)} & {rng('wald_w', 0)} & \\\\", r"\bottomrule", r"\end{tabular}"]
    write("tab_fit.tex", lines)
    return rows


def tab_parameters(S, inf, bl, inf_alt, bl_alt):
    d, full, se = S["d"], S["full"], S["se"]
    i5, i25 = idx(d, 5), idx(d, 25)
    names = {"p0": r"$p_0$ (as good as the best)", "p1": r"$p_1$ (as bad as the worst)",
             "middle": r"$1-p_0-p_1$ (strictly between)", "m": r"$m$ (centre of $\logit e$)",
             "s": r"$s$ (spread of $\logit e$)"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Parameter & Depth 5 & Depth 25 & Infinite depth [95\% interval] \\",
             r"\midrule"]
    for q in ORDER:
        lo, hi = np.percentile([L[q] for L in bl], [2.5, 97.5])
        tail = f"{fmt(inf[q])} [{fmt(lo)}, {fmt(hi)}]" + (r"$^{a}$" if q == "s" else "")
        lines.append(f"{names[q]} & {fmt(full[q][i5])} ({se[q][i5]:.3f}) & {fmt(full[q][i25])} ({se[q][i25]:.3f}) & {tail} \\\\")
    lo, hi = np.percentile([L["s"] for L in bl_alt], [2.5, 97.5])
    lines += [rf"\quad $s$ with the S-curve instead & & & {fmt(inf_alt['s'])} [{fmt(lo)}, {fmt(hi)}] \\", r"\bottomrule",
              r"\end{tabular}"]
    write("tab_parameters.tex", lines)


def tab_depth_functions():
    rows = read("depth_function_checks.csv")
    label = {("p0", "logistic"): r"$p_0$", ("middle", "logistic"): r"$1-p_0-p_1$",
             ("p1", "1 - p0 - middle"): r"$p_1$ (from the two above)", ("m", "logistic"): r"$m$",
             ("s", "gamma kernel"): r"$s$", ("s", "logistic"): r"$s$"}
    shape = {"logistic": "S-curve", "gamma kernel": "gamma kernel", "1 - p0 - middle": "--"}
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Weighted $R^2$} & \multicolumn{3}{c}{Change from depth 20 to 25} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
             r"Parameter & Depth function & Curve & Line & Change & 95\% interval & $p$ \\", r"\midrule"]
    for r in rows:
        if r["change_depths"] != "20-25":
            continue
        key = (r["parameter"], r["depth_function"])
        shown_above = key == ("s", "logistic")          # the change in s is the same data, shown once
        change = " & & " if shown_above else (
            f"{fmt(float(r['change']))} & [{fmt(float(r['change_lo95']))}, {fmt(float(r['change_hi95']))}] & "
            f"{float(r['p_no_change']):.3f}")
        lines.append(f"{label[key]} & {shape[r['depth_function']]} & {float(r['r2_weighted']):.3f} & "
                     f"{float(r['r2_line_weighted']):.3f} & {change} \\\\")
    r = next(r for r in rows if r["change_depths"] == "22-25")
    lines += [r"\midrule", f"$s$, depth 22 to 25 & & & & {fmt(float(r['change']))} & "
              f"[{fmt(float(r['change_lo95']))}, {fmt(float(r['change_hi95']))}] & {float(r['p_no_change']):.3f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_depth_functions.tex", lines)
    return rows


def tab_correlations():
    rows = read("parameter_correlations.csv")
    names = ["p0", "p1", "middle", "m", "s", "within 0.10"]
    tex = {"p0": "$p_0$", "p1": "$p_1$", "middle": "$1-p_0-p_1$", "m": "$m$", "s": "$s$", "within 0.10": "Within 0.10"}
    up = {(r["a"], r["b"]): float(r["correlation"]) for r in rows if r["block"] == "estimates, depth 25 (Spearman over resamples)"}
    low = {(r["a"], r["b"]): float(r["correlation"]) for r in rows if r["block"] == "infinite-depth levels (Spearman over resamples)"}
    get = lambda dct, a, b: dct.get((a, b), dct.get((b, a)))
    lines = [r"\begin{tabular}{l" + "c" * len(names) + "}", r"\toprule", " & " + " & ".join(tex[n] for n in names) + r" \\",
             r"\midrule"]
    for i, a in enumerate(names):
        cells = []
        for j, b in enumerate(names):
            if i == j:
                cells.append("")
            else:
                v = get(up, a, b) if j > i else get(low, a, b)
                cells.append(f"\\textbf{{{fmt(v, 2)}}}" if abs(v) >= 0.5 else fmt(v, 2))
        lines.append(f"{tex[a]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_correlations.tex", lines)
    return up, low


def tab_shares(S, inf, bl, inf_alt, bl_alt):
    d, per_depth, boots = S["d"], S["per_depth"], S["boots"]
    i5, i25 = idx(d, 5), idx(d, 25)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Observed} & Model & \multicolumn{2}{c}{Infinite depth [95\% interval]} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}",
             r"Tolerance $t$ & Depth 5 & Depth 25 & Depth 25 & Gamma kernel for $s$ & S-curve for $s$ \\", r"\midrule"]
    out = {}
    for t in TOL:
        obs = {}
        for li in (i5, i25):
            sh = np.array([np.mean(e <= t + 1e-12) for e in per_depth[li]])
            obs[li] = (sh.mean(), np.std([sh[b].mean() for b in boots], ddof=1))
        mod25 = mdl.model_curve(mdl.at(S["full"], i25), np.array([t]))[0]
        cells = []
        for L0, BL in ((inf, bl), (inf_alt, bl_alt)):
            v = mdl.model_curve(L0, np.array([t]))[0]
            lo, hi = np.percentile([mdl.model_curve(L, np.array([t]))[0] for L in BL], [2.5, 97.5])
            cells.append(f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
        out[t] = (obs[i5], obs[i25], mod25, cells)
        lines.append(f"{t:.2f} & {obs[i5][0]:.3f} ({obs[i5][1]:.3f}) & {obs[i25][0]:.3f} ({obs[i25][1]:.3f}) & {mod25:.3f} & "
                     f"{cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_shares.tex", lines)
    return out


def tab_phase():
    ph = [r for r in read("phase_breakdown.csv") if r["depth"] == "25"]
    split = [r for r in read("p0_p1_split.csv") if r["depth"] == "25"]
    groups = ["12 or fewer", "13-22", "23 or more", "all"]
    val = {(r["pieces"], r["quantity"]): (float(r["value"]), float(r["se"])) for r in ph}
    sp = {r["pieces"]: r for r in split}
    npos = {r["pieces"]: r["positions"] for r in ph}
    head = " & ".join([r"12 or fewer", r"13--22", r"23 or more", r"All"])
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", f"Pieces on the board & {head} \\\\", r"\midrule",
             "Positions & " + " & ".join(npos[g] for g in groups) + r" \\",
             "Legal moves (mean) & " + " & ".join(f"{val[(g, 'legal moves')][0]:.1f} ({val[(g, 'legal moves')][1]:.1f})"
                                                  for g in groups) + r" \\", r"\addlinespace"]
    cell = lambda g, q, nd: f"{fmt(val[(g, q)][0], nd)} ({val[(g, q)][1]:.{nd}f})"
    parts = lambda key: " & ".join(f"{float(sp[g][key]):.3f}" for g in groups)
    lines += [r"$p_0$ & " + " & ".join(cell(g, "p0", 3) for g in groups) + r" \\",
              r"\quad own best move ($1/n$) & " + parts("own_best_move") + r" \\",
              r"\quad all moves equal & " + parts("all_moves_equal") + r" \\",
              r"\quad other ties with the best & " + parts("ties_with_best") + r" \\",
              r"$p_1$ & " + " & ".join(cell(g, "p1", 3) for g in groups) + r" \\",
              r"\quad own worst move ($1/n$) & " + parts("own_worst_move") + r" \\",
              r"\quad other ties with the worst & " + parts("ties_with_worst") + r" \\",
              r"$1-p_0-p_1$ & " + " & ".join(cell(g, "middle", 3) for g in groups) + r" \\",
              r"$m$ & " + " & ".join(cell(g, "m", 2) for g in groups) + r" \\",
              r"$s$ & " + " & ".join(cell(g, "s", 2) for g in groups) + r" \\", r"\addlinespace"]
    for t in TOL:
        lines.append(f"Share within {t:.2f} & " + " & ".join(cell(g, f"within {t:g}", 3) for g in groups) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_phase.tex", lines)
    return val, sp


def main():
    S = mdl.limits()
    inf, inf_alt = mdl.levels(S["fits"], "s"), mdl.levels(S["fits"], "s_logistic")
    bl, bl_alt = mdl.boot_levels(S, "s"), mdl.boot_levels(S, "s_logistic")
    fig_distribution(S)
    fig_depth(S, inf, bl, inf_alt, bl_alt)
    fig_infinite(S, inf, bl, inf_alt)
    models = tab_models()
    tab_fit()
    tab_parameters(S, inf, bl, inf_alt, bl_alt)
    tab_depth_functions()
    tab_correlations()
    shares = tab_shares(S, inf, bl, inf_alt, bl_alt)
    tab_phase()
    print(f"resamples used for the infinite-depth intervals: {len(bl)} (gamma kernel), {len(bl_alt)} (S-curve) of {mdl.B}")
    print("model comparison (AIC wins as measured / near-ties; median largest gap):")
    for m, w in models["measured"][0].items():
        print(f"  {m:<30} {w:>3} {models['near-ties'][0][m]:>3}   {models['measured'][1][m]:.3f} {models['near-ties'][1][m]:.3f}")
    print("shares within t: observed d5 (se), observed d25 (se), model d25, infinite (gamma), infinite (S-curve)")
    for t, (o5, o25, m25, cells) in shares.items():
        print(f"  t={t}: {o5[0]:.3f} ({o5[1]:.3f})  {o25[0]:.3f} ({o25[1]:.3f})  {m25:.3f}  {cells[0]}  {cells[1]}")
    print("infinite depth (gamma kernel for s): " + "  ".join(f"{q} {inf[q]:.3f}" for q in ORDER))
    print("infinite depth (S-curve for s):      " + "  ".join(f"{q} {inf_alt[q]:.3f}" for q in ORDER))
    print("files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
