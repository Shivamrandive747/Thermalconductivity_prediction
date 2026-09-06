"""The doped-data experiment: does site-disorder-aware training help full Heuslers?

FOUR ARMS, each judged on held-out FULL HEUSLERS, never on a pooled number. Half Heuslers
outnumber full ones 14:1 in this training set, so a pooled score is dominated by the class we are
not short of and would hide the only result that matters.

    A0  tier-1 pure only, grouped by FORMULA          the historical baseline
    A1  tier-1 pure only, grouped by ELEMENT SYSTEM   the honest baseline
    A2  A1 + Klemens disorder features                must not hurt: gamma == 0 on pure data
    A3  A2 + 161 recovered doped rows                 the actual question

THE GROUPING CHANGE IS NOT COSMETIC. Seven of the 24 validation compounds have a doped variant in
the recoverable pool -- `Ta1.05Al1Ru1.95` is TaAlRu2 with a 2.5% site swap. Grouped by formula
those are different compounds and can straddle a train/test split; grouped by element system they
cannot. A0 vs A1 measures how much the old grouping was flattering us, which is worth knowing even
if every later arm fails.

A2 exists as a control. The disorder features are identically zero on stoichiometric compounds, so
A2 must reproduce A1. If it does not, the feature is broken and no conclusion from A3 would mean
anything.
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

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import grouped_oof, metrics

DOPED_CSV = "data/Target_Materials/HEUSLER_KAPPA_DOPED_ROWS.csv"
OUT = Path("data/exports/kappa_v2/doped_experiment.json")
GAMMA_COLS = ["gamma_mass", "gamma_radius", "n_sub_sites", "max_site_disorder",
              "stoich_deviation"]


def class_of(formula: str) -> str:
    """full / half / quat, from the STOICHIOMETRY TEMPLATE -- works for doped compounds too.

    `heusler_type()` returns 'other' for anything with four or five elements, so every doped
    variant of a full Heusler was being excluded from the full-Heusler score -- which is exactly
    the score this experiment exists to measure. The template fit does not care how many dopants
    are present: `Al0.85V1.15Fe1.85Cu0.15` still fits [2,1,1] and is still a full Heusler.
    """
    from pymatgen.core import Composition
    try:
        amts = Composition(str(formula)).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return "other"
    caps, _, _ = ds._choose_template(amts, sum(amts.values()))
    if caps == [2.0, 1.0, 1.0]:
        return "full"
    if caps == [1.0, 1.0, 1.0]:
        return "half"
    return "quat"


def evaluate(b, pred, label):
    """Score overall and, separately, on the classes we actually care about."""
    y, groups = b["y"], b["groups"]
    htype = b["meta"].formula.map(class_of).values
    rows = {"n_rows": int(len(y)), "n_groups": int(len(set(groups)))}
    rows["all"] = metrics(y, pred)
    for kind in ("full", "half", "quat"):
        m = htype == kind
        if m.sum() >= 20:
            rows[kind] = metrics(y[m], pred[m])
            rows[f"{kind}_n"] = int(m.sum())
            rows[f"{kind}_compounds"] = int(len(set(groups[m])))
    return rows


def show(label, r):
    a = r["all"]
    print(f"\n  {label}")
    print(f"    rows {r['n_rows']}  groups {r['n_groups']}")
    print(f"    ALL   R2 {a['r2_log']:.4f}  median err {a['median_ape']:.2f}%  "
          f"within30 {a['within_30pct']:.1f}%")
    for kind in ("full", "half"):
        if kind in r:
            m = r[kind]
            print(f"    {kind.upper():<5} R2 {m['r2_log']:.4f}  median err {m['median_ape']:.2f}%  "
                  f"within30 {m['within_30pct']:.1f}%   "
                  f"({r[f'{kind}_n']} rows / {r[f'{kind}_compounds']} groups)")


def main(model: str = "catboost", splits: int = 5) -> int:
    results: dict[str, dict] = {}

    print("=" * 74)
    print("A0  tier-1 pure only, grouped by FORMULA  (historical baseline)")
    print("=" * 74)
    b0 = ds.build(tier_max=1, tier_min=1, group_by="formula")
    X0 = b0["X"].drop(columns=[c for c in GAMMA_COLS if c in b0["X"].columns])
    p0 = grouped_oof(model, X0, b0["y"], b0["groups"], b0["weights"], n_splits=splits)
    results["A0_formula_grouping"] = evaluate(b0, p0, "A0")
    show("A0", results["A0_formula_grouping"])

    print("\n" + "=" * 74)
    print("A1  tier-1 pure only, grouped by ELEMENT SYSTEM  (honest baseline)")
    print("=" * 74)
    b1 = ds.build(tier_max=1, tier_min=1, group_by="element_system")
    X1 = b1["X"].drop(columns=[c for c in GAMMA_COLS if c in b1["X"].columns])
    p1 = grouped_oof(model, X1, b1["y"], b1["groups"], b1["weights"], n_splits=splits)
    results["A1_elemsys_grouping"] = evaluate(b1, p1, "A1")
    show("A1", results["A1_elemsys_grouping"])

    print("\n" + "=" * 74)
    print("A2  + Klemens disorder features  (CONTROL: gamma is 0 on pure data, must match A1)")
    print("=" * 74)
    nz = int((pd.to_numeric(b1["X"].get("gamma_mass", 0), errors="coerce").fillna(0)
              .abs() > 1e-9).sum())
    print(f"  non-zero gamma_mass on the pure training set: {nz}  (expected 0)")
    p2 = grouped_oof(model, b1["X"], b1["y"], b1["groups"], b1["weights"], n_splits=splits)
    results["A2_gamma_features"] = evaluate(b1, p2, "A2")
    show("A2", results["A2_gamma_features"])

    print("\n" + "=" * 74)
    print("A3  + recovered doped rows, disorder visible")
    print("=" * 74)
    b3 = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=DOPED_CSV)
    p3 = grouped_oof(model, b3["X"], b3["y"], b3["groups"], b3["weights"], n_splits=splits)
    results["A3_with_doped"] = evaluate(b3, p3, "A3")
    show("A3", results["A3_with_doped"])

    # A3 trains on more rows than A1/A2, so its overall score is not directly comparable.
    # The comparable question is: on the PURE full Heuslers common to both, did it improve?
    print("\n" + "=" * 74)
    print("A3 vs A1, restricted to the PURE rows they share (the only fair comparison)")
    print("=" * 74)
    dev3 = pd.to_numeric(b3["X"].get("stoich_deviation", 0), errors="coerce").fillna(0).values
    cls3 = b3["meta"].formula.map(class_of).values
    cls1 = b1["meta"].formula.map(class_of).values
    m3 = (cls3 == "full") & (dev3 < 1e-6)     # score only the PURE full Heuslers
    m1 = cls1 == "full"
    if m3.sum() >= 10 and m1.sum() >= 10:
        f1 = metrics(b1["y"][m1], p1[m1])
        f3 = metrics(b3["y"][m3], p3[m3])
        print(f"  A1 full-Heusler pure : R2 {f1['r2_log']:.4f}  median err {f1['median_ape']:.2f}%"
              f"   ({int(m1.sum())} rows)")
        print(f"  A3 full-Heusler pure : R2 {f3['r2_log']:.4f}  median err {f3['median_ape']:.2f}%"
              f"   ({int(m3.sum())} rows)")
        delta = f1["median_ape"] - f3["median_ape"]
        verdict = "KEEP doped rows" if delta > 0 else "DROP doped rows"
        print(f"\n  change in median error: {delta:+.2f} pp   ->  {verdict}")
        results["comparison"] = {"A1_full_pure": f1, "A3_full_pure": f3,
                                 "delta_median_ape_pp": float(delta), "verdict": verdict}
    else:
        print(f"  too few full-Heusler rows to compare (A1 {int(m1.sum())}, A3 {int(m3.sum())})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost")
    ap.add_argument("--splits", type=int, default=5)
    raise SystemExit(main(**vars(ap.parse_args())))
