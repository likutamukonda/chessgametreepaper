"""Step 1: how much do the scale choices move the near-best share at the top level?

Compares Stockfish depth 25 and Lc0 10,000 nodes on the same 264 positions for:
  - the common curve at half, one and twice the Lichess steepness,
  - floors DELTA = 0 (plain 0/0 := 0), 0.01 and 0.02,
  - each engine's own expected-score scale.
Writes out/scale_sensitivity.csv and prints the e <= 0.10 rows.
"""
import numpy as np

import common as c


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    sf, lc = c.load_stockfish(), c.load_lc0()
    top_sf, top_lc = len(sf.levels) - 1, len(lc.levels) - 1
    # Fixed position groups for the whole table, from Stockfish's own verdict at depth 25.
    best_own = np.array([sf.own[p][top_sf].max() for p in sf.pos_ids])
    groups = {"all": np.ones(len(best_own), bool), "open": (best_own >= 0.05) & (best_own <= 0.95),
              "decided": (best_own < 0.05) | (best_own > 0.95)}

    variants = [("common", s, d) for s in (0.5, 1.0, 2.0) for d in (0.0, 0.01, 0.02)]
    variants += [("own", None, d) for d in (0.0, 0.01)]
    rows = []
    for scale, steep, delta in variants:
        own = scale == "own"
        cs, n = c.near_best_counts(sf, c.THRESHOLDS, steep or 1.0, delta, own)
        cl, _ = c.near_best_counts(lc, c.THRESHOLDS, steep or 1.0, delta, own)
        xs_sf, xs_lc = sf.x(steep or 1.0, own), lc.x(steep or 1.0, own)
        floor_sf = sum(1 for p in sf.pos_ids if np.ptp(xs_sf[p][top_sf]) < delta) if delta else 0
        floor_lc = sum(1 for p in lc.pos_ids if np.ptp(xs_lc[p][top_lc]) < delta) if delta else 0
        for j, t in enumerate(c.THRESHOLDS):
            s_sf, s_lc = cs[:, top_sf, j] / n, cl[:, top_lc, j] / n
            m_sf, se_sf = c.mean_se(s_sf)
            m_lc, se_lc = c.mean_se(s_lc)
            m_d, se_d = c.mean_se(s_sf - s_lc)
            corr = {g: correlation(cs[mask, top_sf, j], cl[mask, top_lc, j]) for g, mask in groups.items()}
            rows.append([scale, steep if steep else "", delta, t, round(m_sf, 4), round(se_sf, 4), round(m_lc, 4),
                         round(se_lc, 4), round(m_d, 4), round(se_d, 4), round(corr["all"], 3), round(corr["open"], 3),
                         round(corr["decided"], 3), floor_sf, floor_lc])
    header = ["scale", "steepness", "delta", "threshold", "share_sf_d25", "se_sf", "share_lc0_10k", "se_lc0",
              "difference_sf_minus_lc0", "se_difference", "corr_counts_all", "corr_counts_open",
              "corr_counts_decided", "positions_floor_bites_sf", "positions_floor_bites_lc0"]
    c.write_csv(c.OUT / "scale_sensitivity.csv", rows, header)

    print(f"{len(sf.pos_ids)} positions; groups from Stockfish's own verdict at depth 25: "
          f"open {groups['open'].sum()}, decided {groups['decided'].sum()}")
    print("\nShare of legal moves with e <= 0.10 at the top level (SF depth 25, Lc0 10,000 nodes)")
    print(f"{'scale':<7}{'steep':>6}{'delta':>7} {'SF':>14} {'Lc0':>14} {'SF - Lc0':>16} "
          f"{'corr all':>9} {'open':>6} {'decided':>8} {'floor SF/Lc0':>13}")
    for r in rows:
        if r[3] != 0.10:
            continue
        print(f"{r[0]:<7}{str(r[1]):>6}{r[2]:>7} {r[4]:>8.3f}±{r[5]:.3f} {r[6]:>8.3f}±{r[7]:.3f} "
              f"{r[8]:>+10.3f}±{r[9]:.3f} {r[10]:>9.2f} {r[11]:>6.2f} {r[12]:>8.2f} {r[13]:>7}/{r[14]}")


if __name__ == "__main__":
    main()
