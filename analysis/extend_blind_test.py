"""The calibrated blind test, with the three defects an audit found in the first version fixed.

WHAT THE CALIBRATION IS. Raw predictions are BULK kappa -- what a perfect infinite crystal would
show, because the training labels are first-principles BTE calculations. Nobody measures that. The
over-prediction is systematic in temperature (3.70x at 200 K falling to ~1.0x by 900 K), with the
sign and magnitude grain-boundary scattering demands, so it is removed with two constants:

    kappa_experimental = kappa_bulk * min(c * (T/300)^p, 1)

THREE FIXES OVER THE PREVIOUS VERSION, each of which was inflating the reported accuracy:

  1. THE HOLDOUT IS NOW BY CHEMISTRY CLUSTER, NOT BY COMPOUND. The base predictions come from one
     model per cluster (`run_target_blind_test.py` fits per cluster and applies to every member), so
     two compounds sharing a cluster share that model's bias and are not independent. Fitting the
     calibration leave-one-COMPOUND-out let a held-out compound's cluster-mates inform its own
     correction. The 64 compounds hold only ~27 clusters, so this matters.

  2. THE OBJECTIVE MATCHES THE SCORING. Everything is reported as a per-compound median, but the
     first version minimised the median over ROWS, which lets heavily-measured compounds dominate
     and optimises central accuracy at the cost of catastrophic misses. Measured cost of getting
     this wrong: full Heuslers 46.1% vs 36.7%, half-Heusler within-2x 70% vs 80%.

  3. THE HONEST ACCURACY IS THE PER-FOLD NUMBER, REPORTED SEPARATELY FROM THE PUBLISHED CONSTANTS.
     Taking the median of all fold fits and then scoring compounds with it is leaky -- every other
     fold saw the compound being scored. So two things are printed and they are NOT the same:
       - ACCURACY: each compound scored with the calibration from a fit that excluded its cluster.
       - PUBLISHED CONSTANTS: one (c, p) fitted on everything, for applying to NEW compounds.
     Quoting the second as if it produced the first is the mistake this version refuses to make.

The genuinely leakage-free headline is `nested_validation.py`, which also holds out the form,
objective and support threshold. This script is the descriptive layer beneath it.

RULES KEPT FROM THE FIRST VERSION:
  - scored PER COMPOUND, never per row;
  - half and full Heuslers never pooled;
  - low references are FLAGGED, never dropped (an earlier 0.5 W/m/K floor wrongly condemned
    MgAgSb, a real ultralow-kappa half Heusler).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from run_loco_chemistry import cluster_of

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
SUPPORT = "data/exports/kappa_v2/cluster_support_real.csv"
OUT_CSV = "data/exports/kappa_v2/blind_test_calibrated.csv"
OUT_JSON = "data/exports/kappa_v2/blind_test_calibrated.json"
REVIEW_FLOOR = 1.0          # W/m/K -- flagged for review only, never excluded
CS = np.linspace(0.10, 1.20, 111)
PS = np.linspace(-1.5, 1.5, 61)


def fit_cp(d: pd.DataFrame) -> tuple[float, float]:
    """(c, p) minimising the median of per-compound median absolute log errors.

    Per compound rather than per row: the result is reported per compound, so the fit must be too.
    A median (not a mean) so one digitisation artefact cannot steer the calibration for everything.
    """
    kb, kr, T = d.k_pred.values, d.k_ref.values, d["T"].values.astype(float)
    if len(d) < 4:
        return float("nan"), float("nan")
    codes = pd.Categorical(d.compound).codes
    masks = [codes == k for k in range(codes.max() + 1)]
    fac = np.minimum(CS[:, None, None] * (T[None, None, :] / 300.0) ** PS[None, :, None], 1.0)
    err = np.abs(np.log10(kb[None, None, :] * fac) - np.log10(kr[None, None, :]))
    score = np.empty((len(CS), len(PS)))
    for i in range(len(CS)):
        for j in range(len(PS)):
            e = err[i, j]
            score[i, j] = np.median([np.median(e[m]) for m in masks])
    i, j = np.unravel_index(np.argmin(score), score.shape)
    return float(CS[i]), float(PS[j])


def apply_cp(T, c, p):
    return np.minimum(c * (np.asarray(T, dtype=float) / 300.0) ** p, 1.0)


def score(g: pd.DataFrame) -> dict:
    return dict(n=int(len(g)), median_ape=float(g.ape.median()),
                n_within_30=int((g.ape < 30).sum()), within_30=float((g.ape < 30).mean()),
                n_within_2x=int(g.ratio.between(0.5, 2.0).sum()),
                within_2x=float(g.ratio.between(0.5, 2.0).mean()),
                bias=float(g.ratio.median()))


def main() -> int:
    D = pd.read_csv(BLIND)
    D["cluster"] = D.compound.map(cluster_of)
    sup = pd.read_csv(SUPPORT, index_col=0).n_compounds
    D["sib"] = D.cluster.map(sup).fillna(0).astype(int)
    print(f"blind-test conditions {len(D)}   compounds {D.compound.nunique()}"
          f"   chemistry clusters {D.cluster.nunique()}")
    print(f"  half {D[D.klass == 'half'].compound.nunique()}"
          f"   full {D[D.klass == 'full'].compound.nunique()}")

    # ---- 1. ACCURACY: leave-one-CLUSTER-out calibration --------------------------
    fac = np.full(len(D), np.nan)
    cs, ps = {}, {}
    for cl in sorted(D.cluster.unique()):
        te = (D.cluster == cl).values
        tr = ~te
        assert not (D.cluster.values[tr] == cl).any(), f"cluster {cl} leaked into its calibration"
        c, p = fit_cp(D[tr])
        if c != c:
            continue
        fac[te] = apply_cp(D["T"].values[te], c, p)
        cs[cl], ps[cl] = c, p
    D["fac_loco"] = fac
    D["k_loco"] = D.k_pred * fac

    # ---- 2. PUBLISHED CONSTANTS: one fit on everything, for NEW compounds --------
    c_pub, p_pub = fit_cp(D)
    D["fac_pub"] = apply_cp(D["T"].values, c_pub, p_pub)
    D["k_pub"] = D.k_pred * D.fac_pub
    D["k_none"] = D.k_pred
    D["k_glob"] = D.k_loco          # kept for downstream compatibility: the honest column
    D.to_csv(OUT_CSV, index=False)

    def per_compound(col):
        t = D.assign(_a=100 * (D[col] - D.k_ref).abs() / D.k_ref, _r=D[col] / D.k_ref)
        return (t.groupby(["compound", "klass", "cluster", "sib"], dropna=False)
                .agg(k_ref=("k_ref", "median"), k_pred=("k_pred", "median"),
                     kk=(col, "median"), ape=("_a", "median"), ratio=("_r", "median"),
                     n_cond=("T", "size")).reset_index())

    g = per_compound("k_loco")
    g["review"] = g.k_ref < REVIEW_FLOOR

    print("\n" + "=" * 96)
    print("BLIND TEST vs EXPERIMENT -- chemistry withheld from BOTH model and calibration")
    print("=" * 96)
    print(f"  calibration spread across the {len(cs)} cluster folds: "
          f"c {min(cs.values()):.2f}-{max(cs.values()):.2f}, "
          f"p {min(ps.values()):+.2f} to {max(ps.values()):+.2f}")
    print(f"  PUBLISHED constants, for applying to a new compound: "
          f"c = {c_pub:.3f}, p = {p_pub:+.3f}")
    print("  (the accuracy below uses the per-fold calibrations, NOT these constants)")

    if g.review.any():
        print(f"\n  FLAGGED FOR REVIEW (reference below {REVIEW_FLOOR} W/m/K) -- kept in every "
              f"statistic, listed so a reader can judge:")
        for _, r in g[g.review].iterrows():
            print(f"    {r.compound:<11}measured {r.k_ref:.2f}   raw {r.k_pred:.2f}   "
                  f"calibrated {r.kk:.2f}   {r.ape:.0f}% error")

    summary = {}
    print(f"\n  {'population':<34}{'n':>4}{'median err':>12}{'within 30%':>12}"
          f"{'within 2x':>11}{'bias':>8}")
    for name, msk in (("ALL compounds", g.index == g.index),
                      ("half Heuslers", g.klass == "half"),
                      ("half Heuslers, support >= 3", (g.klass == "half") & (g.sib >= 3)),
                      ("full Heuslers", g.klass == "full"),
                      ("uncorrected, ALL (baseline)", None)):
        if msk is None:
            gg = per_compound("k_none")
            s = score(gg)
        else:
            s = score(g[msk])
        print(f"  {name:<34}{s['n']:>4}{s['median_ape']:>11.1f}%"
              f"{s['n_within_30']:>8}/{s['n']:<3}{s['n_within_2x']:>8}/{s['n']:<3}"
              f"{s['bias']:>7.2f}x")
        summary[name] = s

    print(f"\n  {'compound':<11}{'cls':<6}{'measured':>10}{'raw':>9}{'calibrated':>12}"
          f"{'error':>9}{'ratio':>8}")
    for _, r in g.sort_values("ape").iterrows():
        mark = "  <-- review" if r.review else ""
        print(f"  {r.compound:<11}{str(r.klass):<6}{r.k_ref:>10.2f}{r.k_pred:>9.2f}"
              f"{r.kk:>12.2f}{r.ape:>8.1f}%{r.ratio:>7.2f}x{mark}")

    g.to_csv("data/exports/kappa_v2/blind_per_compound.csv", index=False)
    Path(OUT_JSON).write_text(json.dumps(dict(
        calibration=dict(form="kappa_bulk * min(c*(T/300)^p, 1)",
                         published_c=c_pub, published_p=p_pub,
                         objective="per-compound median absolute log error",
                         accuracy_holdout="leave-one-CHEMISTRY-CLUSTER-out",
                         fold_c_range=[min(cs.values()), max(cs.values())],
                         fold_p_range=[min(ps.values()), max(ps.values())]),
        n_compounds=int(len(g)), n_clusters=int(D.cluster.nunique()),
        by_population=summary,
        note="Accuracy uses per-fold calibrations. The published constants are for new compounds "
             "and must not be quoted as having produced these accuracy figures. The fully "
             "leakage-free headline (form and threshold also held out) is nested_validation.py."),
        indent=2))
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_JSON}")
    print("wrote data/exports/kappa_v2/blind_per_compound.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
