"""Rebuild the conformal prediction intervals on the population the paper actually claims.

The bands quoted in Section 3.6 and applied to every issued prediction were taken from
`interval_validation.json`, whose "half, support>=3" set contains 46 compounds -- including
compounds outside the declared domain -- and whose residuals come from a calibration fitted on the
pooled half AND full Heusler blind set. Neither matches the predictor the paper reports:

  * the domain restriction is central to every other number in the paper;
  * `extend_blind_test.py` states in its own docstring that half and full Heuslers are never
    pooled, so a band built from a pooled fit contradicts the script that produced it.

The consequence is not cosmetic. A band that is too wide is not conservative in any useful sense:
it is a quantitative claim about coverage that the method does not make, and it makes every issued
prediction look less informative than the validation supports.

Construction, unchanged in principle from what the paper describes: absolute residuals in log space
(the error is multiplicative), and the half-width at nominal level alpha is the ceil((m+1)*alpha)-th
order statistic. Coverage is evaluated leave-one-chemistry-out -- the band applied to a held-out
cluster is built only from the residuals of the other clusters -- so the reported coverage is
out-of-sample rather than the level that was requested.
"""
from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import extend_blind_test as E
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

OUT = "data/exports/kappa_v2/conformal_indomain.json"
LEVELS = (0.50, 0.80, 0.90)


def band(res, alpha):
    """Conformal half-width in log10 units: the ceil((m+1)*alpha)-th smallest absolute residual."""
    r = np.sort(np.abs(np.asarray(res)))
    m = len(r)
    if m == 0:
        return np.nan
    k = int(np.ceil((m + 1) * alpha))
    return float(r[min(k, m) - 1])


def main() -> int:
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    clusters = sorted(set(d.chem))
    print(f"in-domain compounds {d.compound.nunique()}   chemistry clusters {len(clusters)}")

    # nested calibration -> the residuals the intervals must describe
    rows = []
    for cl in clusters:
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        c, p = E.fit_cp(tr)
        if not np.isfinite(c):
            continue
        k = te.k_pred.values * E.apply_cp(te["T"].values, c, p)
        rows.append(te.assign(k_cal=k, res=np.log10(k) - np.log10(te.k_ref.values)))
    D = pd.concat(rows)
    per = D.groupby(["compound", "chem"]).res.median().reset_index()
    print(f"per-compound log residuals: {len(per)}")

    R = {"_generated_by": "compute_conformal_indomain.py",
         "population": "the declared domain (VEC=18, cubic, non-polymorphic); "
                       f"{len(per)} compounds, {len(clusters)} chemistry clusters",
         "construction": "absolute log10 residuals; half-width = ceil((m+1)*alpha) order statistic",
         "levels": {}}

    print(f"\n  {'nominal':>8}{'factor':>10}{'coverage':>11}{'n inside':>10}")
    for a in LEVELS:
        # out-of-sample coverage: band for each cluster built from the other clusters only
        inside, tot, widths = 0, 0, []
        for cl in clusters:
            te = per[per.chem == cl]
            tr = per[per.chem != cl]
            if not len(te) or len(tr) < 4:
                continue
            h = band(tr.res.values, a)
            widths.append(h)
            inside += int((te.res.abs() <= h).sum())
            tot += len(te)
        cov = inside / tot * 100 if tot else float("nan")
        # the band a user would apply: fitted on all in-domain compounds
        h_all = band(per.res.values, a)
        R["levels"][f"{a:.2f}"] = dict(
            factor=round(float(10 ** h_all), 2),
            empirical_coverage_pct=round(float(cov), 1),
            n_inside=int(inside), n_scored=int(tot),
            factor_range_across_folds=[round(float(10 ** min(widths)), 2),
                                       round(float(10 ** max(widths)), 2)])
        print(f"  {a*100:>7.0f}%{10**h_all:>9.2f}x{cov:>10.1f}%{inside:>7}/{tot}")

    prev = {"0.50": 1.32, "0.80": 2.06, "0.90": 3.21}
    R["superseded_bands"] = dict(
        values=prev,
        why=("taken from interval_validation.json's 'half, support>=3' set: 46 compounds including "
             "out-of-domain ones, with residuals from a calibration fitted on the pooled half and "
             "full Heusler blind set. Both differ from the predictor this paper reports."))
    print("\n  superseded (46-compound pooled set): "
          + ", ".join(f"{k} -> {v}x" for k, v in prev.items()))
    print("  replacement (in-domain):             "
          + ", ".join(f"{k} -> {R['levels'][k]['factor']}x" for k in R["levels"]))

    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
