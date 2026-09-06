"""Leave-one-CHEMISTRY-out cross-validation, and what it costs us to be honest.

WHY. The deliverable is predictions for compounds nobody has measured, in chemistries the training
set may barely contain. Ordinary cross-validation cannot describe that situation: it holds out
`ZrNiSn` while 507 other half Heuslers -- many with the same elements -- stay in training, so the
score reflects interpolation among neighbours, not discovery.

Meredig et al., Mol. Syst. Des. Eng. 2018 (231 citations), put it bluntly: "Traditional machine
learning (ML) metrics overestimate model performance for materials discovery." Their fix is
leave-one-cluster-out CV, and the cluster has to be the axis along which the new material is new.

For this project that axis is the X2 (doubled) site for full Heuslers -- it dominates the phonon
spectrum -- and the element system for half Heuslers. So each fold removes EVERY compound sharing
that chemistry, then predicts it. `AlZnAg2` is a silver-on-the-doubled-site compound; the only
honest estimate of its error comes from a model that has never seen silver on that site.

Both numbers are reported side by side. The gap between them IS Meredig's point, measured on our
own data, and it belongs in the paper.
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

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import grouped_oof, make_model, metrics

GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
OUT_JSON = Path("data/exports/kappa_v2/loco_chemistry.json")
OUT_CSV = "data/exports/kappa_v2/loco_predictions.csv"
VAL_XLSX = "data/Target_Materials/Validation_Heuslers.xlsx"


def sites(f):
    """('full', X2) / ('half', None) / None."""
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return None
    a = sorted(c.get_el_amt_dict().items(), key=lambda kv: -kv[1])
    if len(a) != 3:
        return None
    v = [x for _, x in a]
    m = min(v)
    r = [round(x / m, 3) for x in v]
    if r == [2.0, 1.0, 1.0]:
        return ("full", a[0][0])
    if r == [1.0, 1.0, 1.0]:
        return ("half", None)
    return None


def elsys(f):
    try:
        return "-".join(sorted(str(e) for e in Composition(str(f)).elements))
    except Exception:  # noqa: BLE001
        return str(f)


def cluster_of(f) -> str:
    """The chemistry a compound belongs to, for holding it out wholesale.

    FULL Heuslers cluster by X2 element -- the doubled site dominates the phonon spectrum, and it
    is the axis along which our targets are new.

    HALF Heuslers CANNOT cluster by element system. Doing so gave 167 clusters of exactly ONE
    compound each, so "leave-one-cluster-out" degenerated into ordinary leave-one-out and the
    resulting 15% was not an extrapolation measurement at all. Meredig et al. cluster in FEATURE
    space for exactly this reason. Here the X-site element (the most electropositive, which sets
    the sublattice) gives a coarse, chemically meaningful grouping with many compounds per
    cluster, so removing one genuinely removes a chemistry.
    """
    s = sites(f)
    if s is None:
        return f"other:{elsys(f)}"
    if s[0] == "full":
        return f"full:X2={s[1]}"
    try:
        c = Composition(str(f))
        items = sorted(((e, e.X if e.X else 99.0) for e in c.elements), key=lambda t: t[1])
        return f"half:X={items[0][0]}"
    except Exception:  # noqa: BLE001
        return f"half:{elsys(f)}"


def validation_formulas() -> set:
    try:
        v = pd.ExcelFile(VAL_XLSX).parse("Compounds")
        col = next(c for c in ("formula", "compound") if c in v.columns)
        return {str(x) for x in v[col].dropna()}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read the validation set ({exc})")
        return set()


def main(model: str = "catboost", extra: bool = True, min_cluster: int = 3) -> int:
    extra_csv = GAP if extra else None
    b = ds.build(tier_max=1, tier_min=1, group_by="element_system",
                 extra_csv=extra_csv, verbose=True)
    forms = b["meta"].formula.values
    clus = np.array([cluster_of(f) for f in forms])
    X, y, w = b["X"], b["y"], b["weights"]

    val = validation_formulas()
    leaked = sorted(set(forms) & val)
    print(f"\n  validation compounds present in this training set: {len(leaked)}")
    print("  (they are held out inside every LOCO fold that contains them; the experimental "
          "arm lives in run_chemistry_matched_validation.py)")

    # ---- 1. ordinary grouped CV, the number usually reported --------------------
    print("\n" + "=" * 78)
    print("ORDINARY grouped CV (grouped by element system) -- the usual reported number")
    print("=" * 78)
    oof_std = grouped_oof(model, X, y, b["groups"], w, n_splits=5)
    cls = np.array(["full" if (sites(f) or ("x",))[0] == "full" else
                    ("half" if (sites(f) or ("x",))[0] == "half" else "other") for f in forms])
    for k in ("full", "half"):
        m = cls == k
        if m.sum() > 20:
            mm = metrics(y[m], oof_std[m])
            print(f"  {k.upper():<5} R2 {mm['r2_log']:>7.4f}   median err {mm['median_ape']:>6.2f}%"
                  f"   ({int(m.sum())} rows)")

    # ---- 2. leave-one-CHEMISTRY-out ---------------------------------------------
    print("\n" + "=" * 78)
    print("LEAVE-ONE-CHEMISTRY-OUT -- every compound of that chemistry removed from training")
    print("=" * 78)
    sizes = pd.Series(clus).value_counts()
    testable = [c for c in sizes.index if sizes[c] >= min_cluster]
    print(f"  clusters: {len(sizes)}   with >= {min_cluster} rows: {len(testable)}")

    oof_loco = np.full(len(y), np.nan)
    rows = []
    for c in testable:
        te = clus == c
        tr = ~te
        if tr.sum() < 50 or te.sum() < 1:
            continue
        # ASSERT the holdout is real: no training row may share this chemistry
        assert not (clus[tr] == c).any(), f"cluster {c} leaked into training"
        mdl = make_model(model)
        mdl.fit(X[tr], y[tr], sample_weight=w[tr])
        p = mdl.predict(X[te])
        oof_loco[te] = p
        mm = metrics(y[te], p)
        std = metrics(y[te], oof_std[te])
        rows.append(dict(cluster=c, n_rows=int(te.sum()),
                         n_compounds=int(len(set(forms[te]))),
                         loco_median_ape=mm["median_ape"], loco_r2=mm["r2_log"],
                         cv_median_ape=std["median_ape"], cv_r2=std["r2_log"],
                         penalty_pp=mm["median_ape"] - std["median_ape"]))
    R = pd.DataFrame(rows).sort_values("n_rows", ascending=False)

    print(f"\n  {'chemistry':<26}{'rows':>6}{'cpds':>6}{'CV err':>9}{'LOCO err':>10}"
          f"{'penalty':>10}")
    print("  " + "-" * 68)
    for _, r in R.iterrows():
        print(f"  {r.cluster:<26}{r.n_rows:>6}{r.n_compounds:>6}{r.cv_median_ape:>8.1f}%"
              f"{r.loco_median_ape:>9.1f}%{r.penalty_pp:>+9.1f}")

    done = ~np.isnan(oof_loco)
    m_all = metrics(y[done], oof_loco[done])
    m_std = metrics(y[done], oof_std[done])
    print("\n  " + "-" * 68)
    print(f"  {'POOLED over tested clusters':<26}{int(done.sum()):>6}"
          f"{len(set(forms[done])):>6}{m_std['median_ape']:>8.1f}%"
          f"{m_all['median_ape']:>9.1f}%{m_all['median_ape'] - m_std['median_ape']:>+9.1f}")

    for k in ("full", "half"):
        m = done & (cls == k)
        if m.sum() > 20:
            a, c2 = metrics(y[m], oof_std[m]), metrics(y[m], oof_loco[m])
            print(f"  {k.upper():<26}{int(m.sum()):>6}{len(set(forms[m])):>6}"
                  f"{a['median_ape']:>8.1f}%{c2['median_ape']:>9.1f}%"
                  f"{c2['median_ape'] - a['median_ape']:>+9.1f}")

    pd.DataFrame(dict(formula=forms, cluster=clus, y_log=y,
                      pred_cv=oof_std, pred_loco=oof_loco,
                      klass=cls)).to_csv(OUT_CSV, index=False)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        per_cluster=R.to_dict("records"),
        pooled_cv=m_std, pooled_loco=m_all,
        n_clusters_tested=len(R), extra_used=bool(extra)), indent=2, default=float))

    print(f"\n  Meredig 2018: ordinary CV overestimates discovery performance. On our data the "
          f"gap is {m_all['median_ape'] - m_std['median_ape']:+.1f} pp.")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost")
    ap.add_argument("--no-extra", dest="extra", action="store_false",
                    help="tier-1 only (default: include the gap-filling tier-3 rows)")
    ap.add_argument("--min-cluster", type=int, default=3)
    raise SystemExit(main(**vars(ap.parse_args())))
