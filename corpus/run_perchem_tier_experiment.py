"""Admit tier-3 labels ONLY where tier-1 has no example of that X2 element.

THE IDEA. Excluding semi-empirical (AGL/Slack) labels was decided on a GLOBAL test, and globally
it is right: pooling them moved R2 0.886 -> 0.847. But "globally" is dominated by chemistries
where tier 1 is rich. For an X2 element with ZERO trainable examples the trade is different --
a 63%-error label is not competing with a good label, it is competing with nothing at all.

The gap, measured:

    X2   targets   tier-1   tier-3 available
    Ru      23        2            4
    Ag      11        0           12
    Co      10        0           20
    Zn       7        0            1
    Cu       6        0           20
    Ni       1        0           27
    Au       2        0           15

So: keep tier 1 alone wherever tier 1 already has >= MIN_SUPPORT examples of that X2 element, and
admit tier 3 only for the starved ones. Li2, Ba2, Fe2 keep their clean labels untouched.

Judged the same way everything else here is judged: out-of-fold, grouped by element system, on
held-out FULL Heuslers, per compound, with a paired test. Kept only on a significant win.
"""
from __future__ import annotations

import argparse
import collections
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

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
EXTRA = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
OUT = Path("data/exports/kappa_v2/perchem_tier_experiment.json")


def parts(f):
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return None
    a = sorted(c.get_el_amt_dict().items(), key=lambda kv: -kv[1])
    if len(a) != 3:
        return None
    v = [x for _, x in a]
    m = min(v)
    return (a[0][0], a[1][0], a[2][0]) if [round(x / m, 3) for x in v] == [2.0, 1.0, 1.0] else None


def build_gap_rows(min_support: int) -> tuple[pd.DataFrame, list]:
    d = pd.read_csv(TRAIN)
    t = pd.to_numeric(d.method_tier, errors="coerce")
    d["p"] = d.formula.map(parts)
    full = d[d.p.notna()]
    have = collections.Counter(p[0] for p in
                               full[t == 1].drop_duplicates("formula").p)
    starved = {x2 for p in full.p for x2 in [p[0]] if have.get(x2, 0) < min_support}
    gap = full[(t == 3) & full.p.map(lambda p: p[0] in starved)].copy()
    # a tier-3 label is worth less; say so in the weight rather than pretending otherwise
    gap["weight"] = 0.3
    gap = gap.drop(columns=["p"])
    gap.to_csv(EXTRA, index=False)
    return gap, sorted(starved)


def per_compound(b, pred, tier_wanted=1):
    f = pd.DataFrame(dict(formula=b["meta"].formula.values, y=b["y"], p=pred,
                          tier=b["tier"]))
    f["p3"] = f.formula.map(parts)
    f = f[f.p3.notna() & (f.tier == tier_wanted)]
    f["ape"] = 100 * np.abs(10 ** f.p - 10 ** f.y) / 10 ** f.y
    return f.groupby("formula").ape.median()


def main(min_support: int = 3, model: str = "catboost", splits: int = 5) -> int:
    gap, starved = build_gap_rows(min_support)
    print(f"X2 elements with < {min_support} tier-1 examples: {starved}")
    print(f"tier-3 rows admitted for them: {len(gap)}  "
          f"({gap.formula.nunique()} compounds)\n")

    print("=" * 70)
    print("C0  tier 1 only (production)")
    print("=" * 70)
    b0 = ds.build(tier_max=1, tier_min=1, group_by="element_system")
    p0 = grouped_oof(model, b0["X"], b0["y"], b0["groups"], b0["weights"], n_splits=splits)
    print(f"  ALL  R2 {metrics(b0['y'], p0)['r2_log']:.4f}  "
          f"median {metrics(b0['y'], p0)['median_ape']:.2f}%")

    print("\n" + "=" * 70)
    print("C1  tier 1 + tier 3 ONLY for starved X2 chemistries")
    print("=" * 70)
    b1 = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=EXTRA)
    p1 = grouped_oof(model, b1["X"], b1["y"], b1["groups"], b1["weights"], n_splits=splits)
    print(f"  ALL  R2 {metrics(b1['y'], p1)['r2_log']:.4f}  "
          f"median {metrics(b1['y'], p1)['median_ape']:.2f}%")

    m0, m1 = per_compound(b0, p0), per_compound(b1, p1)
    both = m0.index.intersection(m1.index)
    a, c = m0[both].values, m1[both].values
    print("\n" + "=" * 70)
    print(f"PAIRED TEST on {len(both)} tier-1 full Heuslers (scored on tier-1 rows only)")
    print("=" * 70)
    print(f"  C0 tier-1 only : {np.median(a):.2f}%")
    print(f"  C1 gap-filled  : {np.median(c):.2f}%")
    print(f"  improved {int((c < a).sum())}, worse {int((c > a).sum())}")
    try:
        p = float(stats.wilcoxon(a, c).pvalue)
    except Exception:  # noqa: BLE001
        p = float("nan")
    delta = float(np.median(a) - np.median(c))
    keep = delta > 0 and p < 0.05
    print(f"  change {delta:+.2f} pp   p={p:.4f}   -> "
          f"{'KEEP' if keep else 'NOT PROVEN - production unchanged'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(starved_x2=starved, gap_rows=len(gap),
                                   gap_compounds=int(gap.formula.nunique()),
                                   median_C0=float(np.median(a)), median_C1=float(np.median(c)),
                                   delta_pp=delta, p=p, keep=bool(keep)), indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--model", default="catboost")
    ap.add_argument("--splits", type=int, default=5)
    raise SystemExit(main(**vars(ap.parse_args())))
