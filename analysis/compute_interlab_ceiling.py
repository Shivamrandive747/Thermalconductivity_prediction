"""The inter-laboratory agreement ceiling, computed rather than quoted.

WHY THIS SCRIPT EXISTS. The project repeatedly compared its accuracy to a "79.5% within 2x
inter-laboratory ceiling" and concluded it was "essentially at the limit". That figure appears in
three places in the codebase and **is computed in none of them** -- it was a hard-coded literal. The
only inter-lab numbers actually derived anywhere are a 16.7% coefficient of variation and a
"53-55% within 25%", both over the whole thermoelectric corpus rather than Heuslers, and neither
using the within-2x metric. So the comparison was unsupported.

It also has to be computed on the MATCHED population -- the same compounds, the same temperature
range, the same metric -- or it is not a ceiling for our number at all.

THE WEIGHTING IS NOT A DETAIL, IT IS THE ANSWER. A compound measured by 40 papers contributes 780
pairwise comparisons and would dominate a pooled count, so the same data supports very different
"ceilings" depending on how it is aggregated. All three are reported, because a reader deserves to
see that the choice matters:

  pooled pairs        every inter-DOI pair counted once. Dominated by well-studied compounds.
  per compound, mean  each compound's own agreement rate, averaged. The like-for-like comparison
                      to our per-compound within-2x, and the strictest of the three.
  per compound, median  the typical compound.

Only pairs from DIFFERENT source DOIs count. Two rows from the same paper are one laboratory, and
Starrydata digitises many points from a single curve, so pooling within a DOI would manufacture
agreement.
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
BLIND = "data/exports/kappa_v2/blind_per_compound.csv"
OUT = "data/exports/kappa_v2/interlab_ceiling.json"
MIN_PAIRS = 3          # a compound needs at least this many inter-DOI pairs to get its own rate


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    d = pd.read_csv(TRAIN)
    d["t"] = pd.to_numeric(d.method_tier, errors="coerce")
    d["k"] = pd.to_numeric(d.kappa_L, errors="coerce")
    d["T"] = pd.to_numeric(d.temperature_K, errors="coerce")
    e = d[(d.t == 0) & (d.k > 0) & d["T"].notna() & d.source_doi.notna()].copy()
    e["red"] = e.formula.map(red)
    e["Tbin"] = (e["T"] / 100).round() * 100

    blind = set(pd.read_csv(BLIND).compound)
    m = e[e.red.isin(blind) & e["T"].between(200, 1200)]
    print(f"matched population: {m.red.nunique()} of {len(blind)} blind-test compounds, "
          f"{len(m)} rows, 200-1200 K")

    # one value per (compound, T-bin, DOI): a laboratory's answer for that condition
    cell = (m.groupby(["red", "Tbin", "source_doi"]).k.median().reset_index())

    pooled_n = pooled_w2 = 0
    ratios: list[float] = []
    per: list[dict] = []
    for c, g in cell.groupby("red"):
        n = w2 = 0
        for _, gg in g.groupby("Tbin"):
            v = gg.k.values
            for a, b in itertools.combinations(v, 2):
                r = max(a, b) / min(a, b)
                ratios.append(r)
                n += 1
                w2 += r <= 2.0
        pooled_n += n
        pooled_w2 += w2
        if n >= MIN_PAIRS:
            per.append(dict(compound=c, pairs=n, within_2x=w2 / n))
    P = pd.DataFrame(per)

    print(f"\ninter-DOI pairs (same compound, same 100 K bin): {pooled_n}")
    print(f"compounds with >= {MIN_PAIRS} such pairs: {len(P)}")
    if not pooled_n or not len(P):
        print("not enough multi-laboratory data to state a ceiling")
        return 1

    pooled = pooled_w2 / pooled_n
    print("\n" + "=" * 80)
    print("INTER-LABORATORY CEILING, within 2x, on the matched population")
    print("=" * 80)
    print(f"  {'weighting':<34}{'within 2x':>11}   note")
    print(f"  {'pooled pairs':<34}{100 * pooled:>10.1f}%   "
          f"{pooled_n} pairs, NOT independent")
    print(f"  {'per compound, mean':<34}{100 * P.within_2x.mean():>10.1f}%   "
          f"like-for-like with our per-compound rate")
    print(f"  {'per compound, median':<34}{100 * P.within_2x.median():>10.1f}%   "
          f"the typical compound")
    print(f"\n  median pairwise disagreement: {np.median(ratios):.2f}x"
          f"   90th percentile {np.quantile(ratios, 0.9):.2f}x")
    print(f"  compounds where labs agree within 2x LESS than 80% of the time: "
          f"{int((P.within_2x < 0.8).sum())} of {len(P)}")
    print("\n  worst-agreeing compounds (these set the mean):")
    for _, r in P.nsmallest(6, "within_2x").iterrows():
        print(f"    {r.compound:<11}{r.pairs:>4} pairs   labs agree {100 * r.within_2x:>5.1f}%")

    print(f"\n  NOTE: the previously quoted 79.5% is closest to the per-compound MEAN "
          f"({100 * P.within_2x.mean():.1f}%),")
    print(f"  but it was never computed in this repository. Any comparison must name its weighting;")
    print(f"  against the pooled or median figure our accuracy is NOT 'essentially at the limit'.")

    Path(OUT).write_text(json.dumps(dict(
        population=dict(compounds=int(P.shape[0]), matched_compounds=int(m.red.nunique()),
                        rows=int(len(m)), T_range=[200, 1200], min_pairs=MIN_PAIRS),
        within_2x=dict(pooled_pairs=pooled, per_compound_mean=float(P.within_2x.mean()),
                       per_compound_median=float(P.within_2x.median())),
        n_pairs=int(pooled_n),
        median_disagreement_x=float(np.median(ratios)),
        p90_disagreement_x=float(np.quantile(ratios, 0.9)),
        per_compound=P.sort_values("within_2x").to_dict("records"),
        note="The 79.5% literal previously quoted is not computed anywhere in this repository. "
             "Use these figures and always state the weighting."), indent=2))
    P.to_csv("data/exports/kappa_v2/interlab_per_compound.csv", index=False)
    print(f"\nwrote {OUT}")
    print("wrote data/exports/kappa_v2/interlab_per_compound.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
