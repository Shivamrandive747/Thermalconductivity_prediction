"""Chemistry support per cluster, counting FIRST-PRINCIPLES compounds only.

WHY THIS IS ITS OWN SCRIPT NOW. The >=3-support exclusion rule was validated against blind-test
errors, then applied to the novel predictions -- but the counts it used came from the whole training
pool, which includes 160 rows of tier-3 semi-empirical AFLOW-AGL Debye-Gruneisen estimates spanning
143 compounds. So a novel compound could be declared "predictable" on the strength of chemistry
whose only labels are semi-empirical. Three of the thirteen novel predictions were in exactly that
position -- `Cu2GePd`, `ZnGaCu2` and `AlZnAg2` had 12-20 "supporting" compounds and **zero** with a
full-BTE label.

An AGL estimate is a fitted approximation to kappa, not a measurement and not a converged
Boltzmann-transport solution. Counting it as evidence that a chemistry is learnable overstates the
case, so `n_compounds` here is the count of tier-1 (full BTE) compounds. The total and the
semi-empirical-only count are kept alongside it so the difference stays visible rather than being
quietly redefined.

An earlier version of this count read `HEUSLER_KAPPA_TRAINING_SET.csv` filtered to tier <= 1, which
missed the `extra_csv` gap rows entirely and was wrong for 39 of 64 blind-test compounds. The count
must come from the pool the model ACTUALLY trains on, which is what `ds.build()` returns.
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pipeline import s19_kappa_dataset as ds
from run_loco_chemistry import cluster_of

GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
OUT = "data/exports/kappa_v2/cluster_support_real.csv"


def main() -> int:
    b = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=GAP, verbose=False)
    m = b["meta"].copy()
    m["cl"] = m.formula.map(cluster_of)
    m["tier"] = pd.to_numeric(m.method_tier, errors="coerce")

    tot = m.groupby("cl").formula.nunique()
    bte = m[m.tier == 1].groupby("cl").formula.nunique()
    semi = m[m.tier != 1].groupby("cl").formula.nunique()

    D = pd.DataFrame({"n_compounds": bte}).reindex(tot.index).fillna(0).astype(int)
    D["n_compounds_all_tiers"] = tot
    D["n_compounds_semiempirical"] = semi.reindex(tot.index).fillna(0).astype(int)
    D.index.name = "cl"
    D = D.sort_values("n_compounds", ascending=False)
    D.to_csv(OUT)

    print(f"clusters {len(D)}   compounds in the training pool {m.formula.nunique()}")
    print(f"  tier-1 (full BTE) compounds : {int(m[m.tier == 1].formula.nunique())}")
    print(f"  tier-3 semi-empirical only  : {int(m[m.tier != 1].formula.nunique())}")

    changed = D[D.n_compounds != D.n_compounds_all_tiers]
    print(f"\nclusters whose support count DROPS when semi-empirical labels stop counting: "
          f"{len(changed)} of {len(D)}")
    print(f"  {'cluster':<20}{'BTE':>6}{'all tiers':>11}{'semi-only':>11}")
    for cl, r in changed.head(18).iterrows():
        print(f"  {cl:<20}{r.n_compounds:>6}{r.n_compounds_all_tiers:>11}"
              f"{r.n_compounds_semiempirical:>11}")

    zero = D[(D.n_compounds == 0) & (D.n_compounds_all_tiers > 0)]
    if len(zero):
        print(f"\n  clusters with NO first-principles compound at all ({len(zero)}) -- any target "
              f"here rests entirely on semi-empirical estimates:")
        for cl, r in zero.iterrows():
            print(f"    {cl:<20}{int(r.n_compounds_all_tiers):>3} semi-empirical compounds")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
