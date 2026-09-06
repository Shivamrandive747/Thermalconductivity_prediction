"""Does the choice of regressor change the number the paper actually reports?

The model comparison in Figure 7 scores fit to CALCULATED labels under compound-level
cross-validation. That is not what this paper claims. Its headline is the error against
MEASUREMENT after the transfer function, with each compound's whole chemistry cluster withheld
from both the model and the calibration. A model that fits DFT better need not calibrate to
experiment better, so the ranking in Figure 7 cannot be assumed to carry over -- it has to be run.

This scores two blind-test runs, identical in every respect except the regressor, through exactly
the pipeline Section 4 uses:

  * restrict to the declared domain (VEC = 18, cubic C1b on record, not polymorphic),
  * calibrate NESTED -- each compound's chemistry cluster is removed from the (c, p) fit, so the
    calibration never sees the compound it is scoring,
  * score per compound, never per row,
  * compare the two on the compounds they share, with a paired test.

Writes model_headline_comparison.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import extend_blind_test as E
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

RUNS = {
    "catboost": "data/exports/kappa_v2/target_blind_test_catboost.csv",
    "lightgbm": "data/exports/kappa_v2/target_blind_test_lightgbm.csv",
}
OUT = Path("data/exports/kappa_v2/model_headline_comparison.json")


def in_domain(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.klass == "half"].copy()
    d["VEC"] = d.compound.map(vec)
    st = {c: structure_status(str(c)) for c in d.compound.unique()}
    keep = (d.VEC == 18) & d.compound.map(lambda c: st[c][0] and not st[c][1])
    return d[keep]


def nested(d: pd.DataFrame) -> pd.DataFrame:
    """Calibrate each compound with its own chemistry cluster out of the (c, p) fit."""
    out, clus = [], d.compound.map(lambda c: cluster_of(c) or "?")
    for cl in sorted(set(clus)):
        te, tr = d[clus == cl], d[clus != cl]
        if not len(te) or len(tr) < 4:
            continue
        ci, pi = E.fit_cp(tr)
        if not np.isfinite(ci):
            continue
        out.append(te.assign(k_cal=te.k_pred.values * E.apply_cp(te["T"].values, ci, pi)))
    return pd.concat(out) if out else pd.DataFrame()


def score(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["ape"] = (d.k_cal - d.k_ref).abs() / d.k_ref * 100
    d["ratio"] = d.k_cal / d.k_ref
    per = d.groupby("compound").agg(ape=("ape", "median"), ratio=("ratio", "median"),
                                    kref=("k_ref", "median"))
    rho, pv = stats.spearmanr(per.kref, per.kref / per.ratio) if len(per) > 4 else (np.nan, np.nan)
    return dict(n_compounds=int(len(per)),
                median_ape=float(per.ape.median()),
                within2x=float((per.ratio.between(0.5, 2.0)).mean() * 100),
                median_ratio=float(per.ratio.median()),
                per_compound=per.ape.to_dict())


def main() -> int:
    res, per = {}, {}
    for name, path in RUNS.items():
        p = Path(path)
        if not p.exists():
            print(f"  {name}: {path} not found -- run "
                  f"run_target_blind_test.py --model {name} first")
            continue
        raw = pd.read_csv(p)
        dom = in_domain(raw)
        cal = nested(dom)
        s = score(cal)
        per[name] = s.pop("per_compound")
        res[name] = s
        print(f"  {name:<10} {s['n_compounds']:>3} compounds   "
              f"median {s['median_ape']:5.1f}%   within 2x {s['within2x']:5.1f}%   "
              f"median ratio {s['median_ratio']:.2f}")

    if len(per) == 2:
        a, b = "catboost", "lightgbm"
        shared = sorted(set(per[a]) & set(per[b]))
        va = np.array([per[a][c] for c in shared])
        vb = np.array([per[b][c] for c in shared])
        w, pv = stats.wilcoxon(va, vb)
        better = int((vb < va).sum())
        print()
        print(f"  paired on the {len(shared)} compounds both scored:")
        print(f"    catboost median {np.median(va):.1f}%   lightgbm median {np.median(vb):.1f}%")
        print(f"    lightgbm better on {better} of {len(shared)} compounds")
        print(f"    Wilcoxon signed-rank p = {pv:.3f}")
        verdict = ("no significant difference -- the regressor is not load-bearing"
                   if pv >= 0.05 else
                   f"{'lightgbm' if np.median(vb) < np.median(va) else 'catboost'} "
                   f"is significantly better")
        print(f"    -> {verdict}")
        res["paired"] = dict(n=len(shared), catboost_median=float(np.median(va)),
                             lightgbm_median=float(np.median(vb)),
                             lightgbm_better_on=better, wilcoxon_p=float(pv),
                             verdict=verdict)
        # the differences between the two are only meaningful against the spread the headline
        # already has across model seeds; anything smaller is noise dressed as a result.
        print("\n  for scale: the published headline moves 31.0-40.0% across five model seeds,")
        print("  so a difference smaller than that spread is not evidence of a better model.")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
