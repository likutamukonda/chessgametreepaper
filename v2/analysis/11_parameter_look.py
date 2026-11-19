"""Step 11 (first look): how the fitted parameters of each family move with depth and node count.

Reads out/error_fits_by_level.csv (step 10, pooled fits) and reports, for every level, the parameters
that can flatten the middle of the distribution and push errors towards 0 and 1:
  ZOIB         precision a + b (smaller = more mass near 0 and 1; as it shrinks, the beta part
               approaches two points at 0 and 1), and the masses pi0, pi1;
  ZOI-logitN   sigma (larger = more mass near 0 and 1);
  censored     the latent scale (larger = bigger masses at 0 and 1 and a flatter middle);
  ZOI-2logitN  the two clusters on the logit scale: centres mu1 < mu2, their separation mu2 - mu1,
               and the weight w of the cluster near 0.
Writes out/parameter_look.csv and out/parameter_look.png.
"""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

SF_COLOR, LC0_COLOR = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    rows = list(csv.DictReader(open(c.OUT / "error_fits_by_level.csv", newline="", encoding="utf-8")))
    get = lambda eng, fam: sorted([r for r in rows if r["engine"] == eng and r["family"] == fam], key=lambda r: int(r["level_index"]))
    out = []
    series = {}
    for eng in ("Stockfish", "Lc0"):
        zb, ln, cn, mx = get(eng, "ZOIB"), get(eng, "ZOI-logitN"), get(eng, "cens-normal"), get(eng, "ZOI-2logitN")
        s = {"level": [], "pi0": [], "pi1": [], "a+b": [], "sigma": [], "cens_scale": [], "w": [], "mu1": [], "mu2": [], "sep": []}
        for a, b, d, e in zip(zb, ln, cn, mx):
            pa, pb, pd, pe = (json.loads(r["params"]) for r in (a, b, d, e))
            vals = [int(a["level"]), pa["pi0"], pa["pi1"], pa["a"] + pa["b"], pb["sigma"], pd["scale"], pe["w"], pe["mu1"], pe["mu2"],
                    pe["mu2"] - pe["mu1"]]
            for k, v in zip(s, vals):
                s[k].append(v)
            out.append([eng] + [round(v, 4) if isinstance(v, float) else v for v in vals])
        series[eng] = s
    c.write_csv(c.OUT / "parameter_look.csv", out, ["engine", "level", "pi0", "pi1", "zoib_a_plus_b", "logitnormal_sigma",
                                                    "censored_normal_scale", "two_cluster_w", "two_cluster_mu1", "two_cluster_mu2",
                                                    "two_cluster_separation"])
    print("Parameters by level (pooled fits of the other moves)")
    print(f"  {'engine':<10}{'level':>7}{'pi0':>7}{'pi1':>7}{'a+b':>7}{'sigma':>7}{'cens s':>8}{'w':>7}{'mu1':>7}{'mu2':>7}{'sep':>7}")
    for r in out:
        if r[1] in (1, 7, 13, 19, 25, 10, 100, 1000, 10000) and not (r[0] == "Stockfish" and r[1] in (10, 100, 1000, 10000)):
            print(f"  {r[0]:<10}{r[1]:>7}" + "".join(f"{v:>7.3f}" for v in r[2:7]) + f"{r[7]:>8.3f}"[1:] + "".join(f"{v:>7.2f}" for v in r[8:]))
    for eng in ("Stockfish", "Lc0"):
        s = series[eng]
        x = np.arange(1, 26)
        print(f"  {eng}: correlation with level step (1-25): " + ", ".join(
            f"{k} {np.corrcoef(x, s[k])[0, 1]:+.2f}" for k in ("pi0", "pi1", "a+b", "sigma", "cens_scale", "w", "sep")))

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    panels = [("pi0", "mass at 0 (ties with the best)"), ("pi1", "mass at 1 (ties with the worst)"),
              ("a+b", "zero-one-inflated beta: a + b"), ("sigma", "logit-normal: sigma"),
              ("sep", "two clusters: separation mu2 - mu1"), ("w", "two clusters: weight of the cluster near 0")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, (key, title) in zip(axes.flat, panels):
        for eng, color in (("Stockfish", SF_COLOR), ("Lc0", LC0_COLOR)):
            ax.plot(range(1, 26), series[eng][key], "-o", color=color, markersize=2.5, linewidth=1.5,
                    label=f"{eng} ({'depth' if eng == 'Stockfish' else 'node level'} k)")
        ax.set_title(title, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("step k (Stockfish depth k; Lc0 level k = 1 ... 10,000 nodes)", fontsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Fitted parameters of the error distribution by search effort (pooled, other moves)", fontsize=10)
    fig.tight_layout()
    fig.savefig(c.OUT / "parameter_look.png", dpi=200)


if __name__ == "__main__":
    main()
