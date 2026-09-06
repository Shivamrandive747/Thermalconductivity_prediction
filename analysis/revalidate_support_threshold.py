"""Re-validate the chemistry-support exclusion rule under the STRICT (BTE-only) definition.

WHY THIS IS NECESSARY. The rule "refuse a target whose chemistry holds fewer than 3 training
compounds" was validated against blind-test errors while support was counted over ALL training
tiers -- which included 160 rows of semi-empirical AFLOW-AGL estimates spanning 143 compounds. The
count has since been tightened to first-principles (tier-1) compounds only, because three novel
predictions turned out to be "supported" by 12-20 compounds with ZERO BTE labels among them.

Tightening the definition without re-testing the threshold would be inheriting a number that was
measured on something else. That is exactly the kind of silent redefinition this project has been
bitten by before, so the rule is re-derived here from scratch under the new count.

WHAT IS MEASURED. For each candidate threshold: how many blind-test compounds it refuses, the median
error of those refused, and the median error and within-2x of those kept. A threshold earns its place
only if the compounds it refuses are genuinely worse than the ones it keeps. Half and full Heuslers
are reported separately, because the strict count collapses almost every full-Heusler cluster to zero
and a single pooled threshold would be dominated by that.
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BLIND = "data/exports/kappa_v2/blind_per_compound.csv"
SUPPORT = "data/exports/kappa_v2/cluster_support_real.csv"


def report(g: pd.DataFrame, col: str, label: str) -> None:
    print(f"\n  --- {label}  (support counted as: {col}) ---")
    print(f"    {'threshold':<14}{'refused':>9}{'refused err':>13}{'kept':>6}"
          f"{'kept err':>10}{'kept 2x':>9}")
    for th in (0, 1, 2, 3, 4, 5, 6, 8, 10):
        kept, ref = g[g[col] >= th], g[g[col] < th]
        if not len(kept):
            continue
        rv = f"{ref.ape.median():.1f}%" if len(ref) else "-"
        print(f"    support >= {th:<3}{len(ref):>9}{rv:>13}{len(kept):>6}"
              f"{kept.ape.median():>9.1f}%"
              f"{100 * kept.ratio.between(.5, 2).mean():>8.0f}%")
    rho = g[col].corr(g.ape, method="spearman")
    print(f"    Spearman(support, error) = {rho:+.3f}"
          f"   {'-- usable signal' if rho < -0.25 else '-- too weak to grade on'}")


def main() -> int:
    g = pd.read_csv(BLIND)
    sup = pd.read_csv(SUPPORT, index_col=0)
    if "cluster" not in g.columns:
        from run_loco_chemistry import cluster_of
        g["cluster"] = g.compound.map(cluster_of)
    g["bte"] = g.cluster.map(sup.n_compounds).fillna(0).astype(int)
    g["allt"] = g.cluster.map(sup.n_compounds_all_tiers).fillna(0).astype(int)

    print("=" * 88)
    print("SUPPORT THRESHOLD, RE-DERIVED UNDER THE STRICT (BTE-ONLY) COUNT")
    print("=" * 88)
    print(f"blind-test compounds {len(g)}   half {int((g.klass == 'half').sum())}"
          f"   full {int((g.klass == 'full').sum())}")
    print(f"\nhow much the definition change moves the counts:")
    ch = g[g.bte != g.allt]
    print(f"  compounds whose support count drops: {len(ch)} of {len(g)}")
    for kl in ("half", "full"):
        s = g[g.klass == kl]
        print(f"  {kl:<5} median support: all-tier {s.allt.median():.0f}"
              f" -> BTE {s.bte.median():.0f}"
              f"   with 0 BTE: {int((s.bte == 0).sum())} of {len(s)}")

    for kl in ("half", "full"):
        s = g[g.klass == kl]
        if len(s) >= 8:
            report(s, "bte", f"{kl.upper()} HEUSLERS")
            report(s, "allt", f"{kl.upper()} HEUSLERS (old lenient count, for comparison)")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    h = g[g.klass == "half"]
    f = g[g.klass == "full"]
    for kl, s in (("half", h), ("full", f)):
        if len(s) < 8:
            continue
        best = None
        for th in (2, 3, 4, 5, 6):
            kept, ref = s[s.bte >= th], s[s.bte < th]
            if len(ref) < 2 or len(kept) < 6:
                continue
            gain = float(ref.ape.median() - kept.ape.median())
            if best is None or gain > best[0]:
                best = (gain, th, len(ref), len(kept))
        if best is None:
            print(f"  {kl}: no threshold has enough compounds on both sides to test.")
            continue
        gain, th, nref, nkept = best
        verdict = ("earns its place" if gain > 15 else
                   "marginal -- refuses compounds that are not clearly worse")
        print(f"  {kl}: best threshold >= {th} (refuses {nref}, keeps {nkept}); "
              f"refused are {gain:+.1f} pp worse -- {verdict}")
    print("\n  NOTE for full Heuslers: the strict count collapses most clusters to zero, so a BTE")
    print("  threshold would refuse most of the class. Where a chemistry has >= 3 MEASURED")
    print("  compounds in the blind test, that direct evidence should be used instead of the proxy")
    print("  -- which is what make_final_results.py does for the Ru2 targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
