"""Does admitting the EXPERIMENTAL tier help full-Heusler prediction?

WHY THIS IS EVEN A QUESTION. Training runs `build(tier_min=1, tier_max=1)` -- full-BTE only.
That silently excludes tier 0, which is measured data: 2,891 rows, 491 of them full Heusler. The
exclusion was deliberate once, because the model predicts a DFT quantity and the experimental
values are the thing we transfer TO. But it also means ten transition-metal full Heuslers with real
measurements -- five of them Co2, an X2 site with ZERO tier-1 support -- never reach the model.

TWO COMPOUNDS ARE REMOVED BEFORE THE TEST, both verified earlier:
  SrMnBi2   a layered Dirac semimetal, not a Heusler. 2:1:1 composition shape is a known trap.
  NbGaRu2   this project already retracted its measured value as fabricated.

THE CAVEAT THAT MUST TRAVEL WITH ANY RESULT. Mixing experimental labels into a DFT-target model
changes what the model predicts. If this arm wins, the grain-boundary correction in s21/s22 has to
be re-derived, because part of the correction is now learned rather than applied. This experiment
measures whether it is worth that work -- it does not by itself authorise the change.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy import stats

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import grouped_oof, metrics

OUT = Path("data/exports/kappa_v2/tier0_experiment.json")
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
TIER0_CSV = "data/Target_Materials/HEUSLER_KAPPA_TIER0_ROWS.csv"
# NbGaRu2 was listed here while its kappa was the fabricated 18.0 W/m/K. That value has since been
# replaced by the real measurement (5.16-5.19 W/m/K over 210-298 K, doi 10.1016/j.jallcom.2020.156617),
# which sits inside the independently expected 4.6-5.5 range, so the compound is valid again and
# rejecting it would throw away real data. SrMnBi2 stays: it is genuinely not a Heusler, and it is
# now also quarantined at source in build_kappa_pool.NOT_HEUSLER_PROTOTYPE.
REJECT = {"SrMnBi2": "layered ZrCuSiAs-type Dirac semimetal, not a Heusler"}
TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au".split())


def cls(f):
    try:
        a = Composition(str(f)).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return "other"
    caps, _, _ = ds._choose_template(a, sum(a.values()))
    return {"[2.0, 1.0, 1.0]": "full", "[1.0, 1.0, 1.0]": "half"}.get(str(caps), "quat")


def build_tier0():
    d = pd.read_csv(TRAIN)
    t = pd.to_numeric(d.method_tier, errors="coerce")
    val = set()
    try:
        v = pd.ExcelFile("data/Target_Materials/Validation_Heuslers.xlsx").parse("Compounds")
        fc = next(c for c in ("formula", "compound") if c in v.columns)
        val = {str(x) for x in v[fc].dropna()}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read validation set ({exc}) -- ABORTING, because "
              f"admitting tier 0 without holding the validation compounds out would corrupt "
              f"the headline number")
        raise SystemExit(1)

    t0 = d[t == 0].copy()
    t0 = t0[~t0.formula.isin(val)]
    before = t0.formula.nunique()
    t0 = t0[~t0.formula.isin(REJECT)]
    print(f"  tier-0 rows outside validation : {len(t0)} ({t0.formula.nunique()} compounds)")
    for k, why in REJECT.items():
        print(f"    rejected {k:<10} {why}")
    print(f"    compounds {before} -> {t0.formula.nunique()}")
    t0.to_csv(TIER0_CSV, index=False)
    return t0


def evaluate(b, pred):
    y, groups = b["y"], b["groups"]
    c = b["meta"].formula.map(cls).values
    out = {"all": metrics(y, pred), "n_rows": int(len(y))}
    for k in ("full", "half"):
        m = c == k
        if m.sum() >= 20:
            out[k] = metrics(y[m], pred[m])
            out[f"{k}_n"] = int(m.sum())
            out[f"{k}_compounds"] = int(len(set(groups[m])))
    return out


def show(tag, r):
    a = r["all"]
    print(f"\n  {tag}   rows {r['n_rows']}")
    print(f"    ALL   R2 {a['r2_log']:.4f}  median err {a['median_ape']:.2f}%")
    for k in ("full", "half"):
        if k in r:
            m = r[k]
            print(f"    {k.upper():<5} R2 {m['r2_log']:.4f}  median err {m['median_ape']:.2f}%"
                  f"   ({r[f'{k}_n']} rows / {r[f'{k}_compounds']} compounds)")


def main(model: str = "catboost", splits: int = 5) -> int:
    print("=" * 74)
    print("preparing the tier-0 rows")
    print("=" * 74)
    build_tier0()

    print("\n" + "=" * 74)
    print("B0  tier-1 only (current production setting)")
    print("=" * 74)
    b0 = ds.build(tier_max=1, tier_min=1, group_by="element_system")
    p0 = grouped_oof(model, b0["X"], b0["y"], b0["groups"], b0["weights"], n_splits=splits)
    r0 = evaluate(b0, p0)
    show("B0", r0)

    print("\n" + "=" * 74)
    print("B1  tier 1 + tier 0 (experimental), validation compounds held out")
    print("=" * 74)
    b1 = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=TIER0_CSV)
    p1 = grouped_oof(model, b1["X"], b1["y"], b1["groups"], b1["weights"], n_splits=splits)
    r1 = evaluate(b1, p1)
    show("B1", r1)

    # ---- the only fair comparison: per compound, on the tier-1 full Heuslers common to both
    print("\n" + "=" * 74)
    print("PAIRED TEST on full Heuslers (per compound -- pooling rows weights by how many")
    print("temperature points a compound happens to have, which is not the question)")
    print("=" * 74)

    def per_compound(b, p):
        f = pd.DataFrame(dict(formula=b["meta"].formula.values, y=b["y"], p=p,
                              tier=b["tier"]))
        f["cls"] = f.formula.map(cls)
        f["ape"] = 100 * np.abs(10 ** f.p - 10 ** f.y) / 10 ** f.y
        f = f[(f.cls == "full") & (f.tier == 1)]      # score DFT-label rows only, both arms
        return f.groupby("formula").ape.median()

    m0, m1 = per_compound(b0, p0), per_compound(b1, p1)
    both = m0.index.intersection(m1.index)
    a, c = m0[both].values, m1[both].values
    print(f"  n = {len(both)} full Heuslers")
    print(f"  B0 tier-1 only        : {np.median(a):.2f}%")
    print(f"  B1 with experimental  : {np.median(c):.2f}%")
    print(f"  improved on {int((c < a).sum())}, worse on {int((c > a).sum())}")
    try:
        w = stats.wilcoxon(a, c)
        print(f"  Wilcoxon p = {w.pvalue:.4f}")
        pval = float(w.pvalue)
    except Exception:  # noqa: BLE001
        pval = float("nan")

    tm = [f for f in both if {str(e) for e in Composition(f).elements} & TM]
    if len(tm) >= 5:
        at, ct = m0[tm].values, m1[tm].values
        try:
            pt = float(stats.wilcoxon(at, ct).pvalue)
        except Exception:  # noqa: BLE001
            pt = float("nan")
        print(f"\n  transition-metal subset (n={len(tm)}): "
              f"{np.median(at):.2f}% -> {np.median(ct):.2f}%   p={pt:.3f}")

    delta = float(np.median(a) - np.median(c))
    keep = delta > 0 and pval < 0.05
    print(f"\n  change {delta:+.2f} pp, p={pval:.4f}  ->  "
          f"{'KEEP tier 0' if keep else 'NOT PROVEN - do not change production'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"B0": r0, "B1": r1, "n_compounds": int(len(both)),
                               "median_B0": float(np.median(a)), "median_B1": float(np.median(c)),
                               "delta_pp": delta, "p": pval, "keep": bool(keep)},
                              indent=2, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost")
    ap.add_argument("--splits", type=int, default=5)
    raise SystemExit(main(**vars(ap.parse_args())))
