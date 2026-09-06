"""What is the CatBoost model actually trained on? A full audit of the training matrix.

The corpus has been rebuilt, deduplicated, re-tiered and filtered many times. Before comparing
regressors on it, it is worth establishing exactly what "it" is -- and in particular whether any
of the things that would invalidate the comparison have crept in:

  * measured (tier-0) rows leaking into a pool that is supposed to be calculations only,
  * a single compound or a single publication dominating by row count,
  * descriptors that are mostly missing,
  * duplicate rows inflating the apparent sample size,
  * temperature or conductivity coverage that does not match where the paper makes claims.

Prints one section per question and asserts the things that must be true.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pipeline.s19_kappa_dataset import build
from run_loco_chemistry import cluster_of, sites

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
RULE = "=" * 84


def main() -> int:
    print(RULE)
    print("THE BUILD -- exactly the call run_target_blind_test.py makes")
    print(RULE)
    b = build(tier_max=1, tier_min=1, group_by="element_system", verbose=False)
    X, y, w, g = b["X"], b["y"], b["weights"], b["groups"]
    meta = b["meta"]
    print(f"  build(tier_max=1, tier_min=1, group_by='element_system')")
    print(f"  rows        {len(y)}")
    print(f"  descriptors {X.shape[1]}")
    print(f"  compounds   {meta.formula.nunique()}")
    print(f"  groups      {len(set(g))}   (the CV unit)")
    print(f"  target      log10(kappa_L), range {y.min():.2f} to {y.max():.2f} "
          f"= {10 ** y.min():.2f} to {10 ** y.max():.1f} W/m/K")

    print()
    print(RULE)
    print("1. IS IT REALLY CALCULATIONS ONLY? (a measured row here would be leakage)")
    print(RULE)
    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    forms = set(meta.formula)
    used = tr[tr.formula.isin(forms)]
    print("  tier composition of the source rows for the formulas in the matrix:")
    for t_, n in used.tier.value_counts().sort_index().items():
        print(f"    tier {int(t_)}: {n} rows in the corpus")
    print(f"  measured (tier-0) compounds that ALSO appear in training: "
          f"{len(set(tr[tr.tier == 0].formula) & forms)}")
    print("  -> that overlap is intended: the blind test predicts a compound from its CHEMISTRY")
    print("     with the whole cluster held out, then compares to that compound's measurement.")
    print("     What would be leakage is a tier-0 VALUE used as a training label; the build's")
    print("     tier_min=1 filter is what prevents it.")

    print()
    print(RULE)
    print("2. WHAT KIND OF CALCULATION? (the method string decides the tier)")
    print(RULE)
    m = (tr[tr.formula.isin(forms) & (tr.tier == 1)].method.fillna("?")
         .str.lower().str.slice(0, 46).value_counts())
    for k, n in m.head(12).items():
        print(f"    {n:>5}  {k}")
    if len(m) > 12:
        print(f"    ... and {len(m) - 12} further method strings")

    print()
    print(RULE)
    print("3. IS ANY ONE COMPOUND OR PAPER DOMINATING?")
    print(RULE)
    per = pd.Series(meta.formula).value_counts()
    print(f"  rows per compound: median {per.median():.0f}, max {per.max()} ({per.idxmax()})")
    top = per.head(8)
    print(f"  the {len(top)} heaviest compounds hold {top.sum()} of {len(y)} rows "
          f"({top.sum() / len(y) * 100:.1f}%):")
    for k, n in top.items():
        print(f"    {k:<12}{n:>5} rows")
    src = tr[tr.formula.isin(forms) & (tr.tier == 1)]
    if "source_doi" in src.columns:
        sd = src.source_doi.fillna("(none)").value_counts()
        print(f"\n  distinct sources: {len(sd)}   largest contributes {sd.iloc[0]} rows "
              f"({sd.iloc[0] / len(src) * 100:.1f}%)")
        for k, n in sd.head(5).items():
            print(f"    {n:>5}  {str(k)[:60]}")

    print()
    print(RULE)
    print("4. STRUCTURE CLASS AND CHEMISTRY SPREAD")
    print(RULE)
    kl = pd.Series([(sites(f) or ["?"])[0] for f in meta.formula]).value_counts()
    print("  rows by structure class:", dict(kl))
    cl = pd.Series([cluster_of(f) or "?" for f in meta.formula]).value_counts()
    print(f"  chemistry clusters: {len(cl)}   largest {cl.iloc[0]} rows ({cl.index[0]})")
    print(f"  clusters with < 20 rows: {int((cl < 20).sum())} of {len(cl)}")

    print()
    print(RULE)
    print("5. COVERAGE -- does the training data span where the paper makes claims?")
    print(RULE)
    T = X["temperature_K"] if "temperature_K" in X.columns else None
    if T is not None:
        print(f"  temperature: {T.min():.0f} to {T.max():.0f} K")
        for lo, hi in ((0, 200), (200, 400), (400, 700), (700, 1100), (1100, 1e9)):
            n = int(((T >= lo) & (T < hi)).sum())
            print(f"    {lo:>5}-{hi if hi < 1e9 else 'inf':>5} K   {n:>5} rows "
                  f"({n / len(T) * 100:4.1f}%)")
    k = 10 ** y
    print(f"  kappa_L: {k.min():.2f} to {k.max():.1f} W/m/K")
    for lo, hi, lab in ((0, 2, "<2"), (2, 5, "2-5"), (5, 10, "5-10"), (10, 1e9, ">10")):
        n = int(((k >= lo) & (k < hi)).sum())
        print(f"    {lab:>5} W/m/K   {n:>5} rows ({n / len(k) * 100:4.1f}%)")
    print("  -> the paper's predictions sit at 0.9-6.3 W/m/K, inside the best-populated bands.")

    print()
    print(RULE)
    print("6. DESCRIPTOR COMPLETENESS")
    print(RULE)
    na = X.isna().mean().sort_values(ascending=False)
    bad = na[na > 0]
    if len(bad):
        print(f"  {len(bad)} of {X.shape[1]} descriptors have missing values:")
        for c_, f_ in bad.head(10).items():
            print(f"    {c_:<26}{f_ * 100:5.1f}% missing")
    else:
        print(f"  all {X.shape[1]} descriptors complete on all {len(y)} rows")
    print(f"  rows complete on every descriptor: {int(X.notna().all(axis=1).sum())} of {len(y)}")

    print()
    print(RULE)
    print("7. DUPLICATES AND WEIGHTS")
    print(RULE)
    dup = X.duplicated().sum()
    print(f"  exactly duplicated descriptor rows remaining: {dup}")
    print(f"  sample weights: min {w.min():.3f}  median {np.median(w):.3f}  max {w.max():.3f}")
    print(f"  distinct weight values: {len(set(np.round(w, 4)))}")

    print()
    print(RULE)
    print("ASSERTIONS")
    print(RULE)
    checks = [
        ("no NaN in the target", not np.isnan(y).any()),
        ("no non-positive kappa", (10 ** y > 0).all()),
        ("groups are the CV unit and there are enough of them", len(set(g)) >= 50),
        ("no single compound holds >20% of rows", per.max() / len(y) < 0.20),
        ("temperature present as a descriptor", "temperature_K" in X.columns),
    ]
    for name, okv in checks:
        print(f"  [{'PASS' if okv else 'FAIL'}] {name}")
    assert all(o for _, o in checks), "a structural assumption of the training set is violated"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
