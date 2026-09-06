"""Would a ONE-anchor family rule have been safe? Leave-one-out simulation.

The observation that prompted this: compounds whose family held exactly one usable anchor scored
6 of 6 within a factor of two, as good as those with two or more. If that holds, the family rule
could be k=1 instead of k=2, and Ni-Bi would need ONE measurement rather than two -- releasing
nine to eleven predictions for a single experiment.

But that observation is contaminated. "Families with one anchor" was computed with the target
compound's own family membership counted from the full table, so a compound can be its own
support. The honest question is counterfactual and must be asked leave-one-out:

    for each measured compound c, count the usable anchors in c's family EXCLUDING c.
    Under a k-anchor rule, c would have been predicted iff that count >= k.
    Then: of the compounds a k=1 rule ADMITS THAT k=2 DOES NOT, how accurate are they?

That last group is the entire question. If those are as good as the k=2 group, the rule can be
relaxed. If they are not, the observation was an artefact and k=2 stands.

Reports the decision either way. A relaxation is only worth adopting if the newly admitted
compounds hold up on their own, not if the pooled average survives.
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, "c:/Users/ojasw/Documents/Data_collection_setup")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy import stats

from make_paper_predictions import vec

SP = "paper/evidence"


def yz(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


g = pd.read_csv(f"{SP}/PROVEN_all_scored.csv")
g["fam"] = g.compound.map(yz)
g["_v"] = g.compound.map(vec)
d = g[(g._v == 18) & ~g.broken].copy()
print(f"in-domain, non-broken measured compounds: {len(d)}")

# a compound counts as an ANCHOR for its family if it is non-broken and landed within 2x
d["is_anchor"] = d.ratio.between(0.5, 2.0)
fam_anchor = d.groupby("fam").is_anchor.sum().to_dict()

rows = []
for r in d.itertuples():
    # anchors available to r, EXCLUDING r itself -- this is the correction
    n_excl = int(fam_anchor.get(r.fam, 0)) - (1 if r.is_anchor else 0)
    rows.append(dict(compound=r.compound, fam=r.fam, ape=r.ape_med, ratio=r.ratio,
                     anchors_excl_self=n_excl, within2=bool(r.is_anchor)))
S = pd.DataFrame(rows)

print("\nleave-one-out anchor count (excluding the compound itself):")
for k, grp in S.groupby("anchors_excl_self"):
    print(f"  {k} anchor(s): n={len(grp):<3} median err {grp.ape.median():>5.1f}%   "
          f"within 2x {grp.within2.mean()*100:>5.1f}%   {sorted(grp.compound)[:6]}")

k2 = S[S.anchors_excl_self >= 2]
k1_only = S[S.anchors_excl_self == 1]          # newly admitted by relaxing k=2 -> k=1
k0 = S[S.anchors_excl_self == 0]

print("\n" + "=" * 76)
print("THE DECISIVE COMPARISON")
print("=" * 76)
print(f"  k>=2 admits            n={len(k2):<3} median err {k2.ape.median():>5.1f}%   "
      f"within 2x {k2.within2.mean()*100:>5.1f}%")
print(f"  k=1 ADDITIONALLY admits n={len(k1_only):<3} median err "
      f"{k1_only.ape.median():>5.1f}%   within 2x {k1_only.within2.mean()*100:>5.1f}%")
print(f"  k=0 (never admitted)   n={len(k0):<3} median err {k0.ape.median():>5.1f}%   "
      f"within 2x {k0.within2.mean()*100:>5.1f}%")

if len(k1_only) >= 3 and len(k2) >= 3:
    u, p = stats.mannwhitneyu(k1_only.ape, k2.ape, alternative="two-sided")
    print(f"\n  Mann-Whitney on error, newly-admitted vs k>=2:  p = {p:.3f}")
    print(f"  the newly admitted compounds are: {sorted(k1_only.compound)}")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
if len(k1_only) == 0:
    print("  No compound is admitted by k=1 that k=2 does not already admit. The earlier")
    print("  'one anchor scored 6/6' observation was an artefact of counting a compound as its")
    print("  own support. There is nothing to relax. k=2 stands.")
else:
    worse = k1_only.ape.median() > k2.ape.median()
    miss = k1_only.within2.mean() < 1.0
    print(f"  newly admitted median error {k1_only.ape.median():.1f}% vs "
          f"{k2.ape.median():.1f}% for k>=2")
    print(f"  newly admitted within-2x rate {k1_only.within2.mean()*100:.0f}%")
    if len(k1_only) < 5:
        print(f"  n = {len(k1_only)} is too small to license a rule change on its own.")
    print("  ADOPT only if the newly admitted group is not materially worse AND n is adequate.")

print("\nwhat this would mean for Ni-Bi either way:")
print("  Ni-Bi has ZERO usable anchors (YNiBi is broken). Under k=1 it needs one sound")
print("  measurement; under k=2 it needs two. The rule change halves that cost but does not")
print("  eliminate it -- no rule admits a family with no measurement at all.")
