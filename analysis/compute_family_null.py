"""The family power-law baseline: does the method beat it, and where does it add anything?

A review found, and independent computation confirmed, that a power law fitted to the MEASURED
kappa(T) of a compound's own (Y,Z) bonding family matches this paper's transfer function on median
error. That baseline needs no first-principles calculation, no machine learning and no calibration.
It has to go in the paper.

But "matches on median error" is not the whole comparison, and two things decide whether the method
has any defensible advantage:

  1 COVERAGE. The family null needs at least two measured same-family compounds outside the
    held-out cluster. Where it has them the method must justify its extra machinery; where it does
    not, the method is the only option. The decisive case is the five compounds the paper actually
    issues predictions for.

  2 WITHIN-FAMILY DISCRIMINATION. This is the real question. A family power law returns ONE curve
    per family, so every member of a family receives the same prediction at a given temperature. It
    cannot rank ErSbPt against HoSbPt against TmSbPt -- and choosing between the members of a
    substitution series is precisely what a screening user does. The transfer function, carrying a
    compound-specific calculation, can.

Everything is leave-one-chemistry-out, in domain, per-compound median error.
"""
from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr, wilcoxon

import extend_blind_test as E
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

OUT = "data/exports/kappa_v2/family_null.json"
ISSUED = {"ErSbPt", "HoSbPt", "TmSbPt", "GdNiSb", "TbNiSb"}


def yz(f):
    e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
    return f"{e[1]}-{e[2]}" if len(e) == 3 else None


def main() -> int:
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    d["fam"] = d.compound.map(yz)

    model, fam = [], []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        c, p = E.fit_cp(tr)
        if np.isfinite(c):
            model.append(te.assign(k=te.k_pred.values * E.apply_cp(te["T"].values, c, p)))
        for f, g in te.groupby("fam"):
            trf = tr[tr.fam == f]
            if trf.compound.nunique() < 2:
                continue
            q, la = np.polyfit(np.log(trf["T"].values / 300.0), np.log(trf.k_ref.values), 1)
            fam.append(g.assign(k=np.exp(la) * (g["T"].values / 300.0) ** q))

    def pc(frames):
        D = pd.concat(frames)
        return (D.assign(a=(D.k - D.k_ref).abs() / D.k_ref * 100, r=D.k / D.k_ref)
                 .groupby("compound").agg(a=("a", "median"), r=("r", "median"),
                                          ref=("k_ref", "median"), k=("k", "median"),
                                          fam=("fam", "first")))

    M, F = pc(model), pc(fam)
    common = M.index.intersection(F.index)
    _, pv = wilcoxon(M.a[common], F.a[common])
    R = {"_generated_by": "compute_family_null.py",
         "model": dict(n=int(len(M)), median_ape=round(float(M.a.median()), 1),
                       within_2x=round(float(M.r.between(.5, 2).mean() * 100), 1)),
         "family_null": dict(n=int(len(F)), median_ape=round(float(F.a.median()), 1),
                             within_2x=round(float(F.r.between(.5, 2).mean() * 100), 1)),
         "head_to_head": dict(n=int(len(common)),
                              model=round(float(M.a[common].median()), 1),
                              family=round(float(F.a[common].median()), 1),
                              wilcoxon_p=round(float(pv), 3),
                              model_better_pct=round(float((M.a[common] < F.a[common]).mean() * 100), 1))}

    missed = sorted(set(M.index) - set(F.index))
    R["family_null_cannot_cover"] = missed
    R["issued_covered_by_family_null"] = sorted(ISSUED & set(F.index))
    print(f"  model      n={len(M)}  median {M.a.median():.1f}%")
    print(f"  family null n={len(F)}  median {F.a.median():.1f}%")
    print(f"  head to head on {len(common)}: model {M.a[common].median():.1f}% vs "
          f"{F.a[common].median():.1f}%, p={pv:.3f}, model better on "
          f"{(M.a[common] < F.a[common]).mean()*100:.0f}%")
    print(f"  family null cannot cover {len(missed)}: {missed}")

    # --- THE DECISIVE TEST: can either rank compounds WITHIN a family? --------------------
    print("\n  within-family discrimination (a family null gives every member the same value):")
    rows = []
    for f, g in M.join(F.k.rename("k_fam"), how="inner").groupby("fam"):
        if len(g) < 3:
            continue
        rm, _ = spearmanr(g.ref, g.k)
        rf, _ = spearmanr(g.ref, g.k_fam)
        spread_f = float(g.k_fam.max() / g.k_fam.min())
        rows.append(dict(fam=f, n=int(len(g)), rho_model=round(float(rm), 3),
                         rho_family=round(float(rf), 3), family_spread=round(spread_f, 2)))
        print(f"    {f:<8}n={len(g)}  model rho={rm:+.3f}   family-null rho={rf:+.3f}   "
              f"family-null spread {spread_f:.2f}x")
    R["within_family"] = rows
    if rows:
        mm = float(np.median([r["rho_model"] for r in rows]))
        ff = float(np.median([r["rho_family"] for r in rows if np.isfinite(r["rho_family"])]))
        R["within_family_median_rho"] = dict(model=round(mm, 3), family_null=round(ff, 3))
        print(f"    median within-family rho: model {mm:+.3f}  family null {ff:+.3f}")

    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
