"""Predicted values for the fourteen conditional candidates.

These are NOT issued predictions. Each sits in a bonding family that does not yet meet the
validation bar, so the method is not entitled to speak about it. What this computes is what the
prediction WOULD be once the family qualifies -- one further measurement in Co-Bi, Ni-Pb or Sb-Ir,
and one sound measurement anywhere in Ni-Bi.

Every candidate here has already passed, in scratchpad/audit_unlock.py:
  VEC = 18 | cubic 216 on record, not polymorphic | no actinide | a real transport calculation |
  an X site represented in training | independent calculations agreeing within 2x

Three candidates were removed by those checks and are deliberately absent: YbNiBi (no cubic record,
no transport calculation, X site unseen), TiNiPb (two sources 5.6x apart), HoNiBi (2.37x apart).
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

import extend_blind_test as E
from make_paper_predictions import FLUCT, RADIO, structure_status, vec
from run_loco_chemistry import cluster_of, sites

OUT = "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"

COND = {
    "HfCoBi": "Co-Bi", "TiCoBi": "Co-Bi",
    "HfNiPb": "Ni-Pb",
    "HfSbIr": "Sb-Ir", "TiSbIr": "Sb-Ir",
    "DyNiBi": "Ni-Bi", "ErNiBi": "Ni-Bi", "GdNiBi": "Ni-Bi", "LuNiBi": "Ni-Bi",
    "NdNiBi": "Ni-Bi", "PrNiBi": "Ni-Bi", "SmNiBi": "Ni-Bi", "TbNiBi": "Ni-Bi",
    "TmNiBi": "Ni-Bi",
}
UNLOCKED_BY = {"Co-Bi": "one further measured Co-Bi half Heusler",
               "Ni-Pb": "one further measured Ni-Pb half Heusler",
               "Sb-Ir": "one further measured Sb-Ir half Heusler",
               # TWO, not one. YNiBi is broken, so the family sits at zero qualifying members;
               # a single new measurement takes it to one, still short of the >=2 rule enforced
               # in make_paper_predictions.py. This string previously said "one" and contradicted
               # both that rule and Section 5, and it ships in the supplementary CSV.
               "Ni-Bi": ("two sound measurements: a re-measurement of YNiBi below its bipolar "
                         "onset plus one further Ni-Bi compound, or two new compounds")}


def caveat_for(cp_: str) -> str:
    """Apply the SAME element screens the issued predictions get.

    make_paper_predictions.py refuses radioactive elements and marks valence-fluctuating ones
    FLAGGED. This script applied neither: it worked from the hardcoded COND list, so SmNiBi was
    listed beside the other eight with no hint that Sm can be di- or trivalent, which makes the
    VEC=18 premise unsafe for it. A screen that runs in one script and not its sibling is not a
    screen. Importing FLUCT/RADIO rather than restating them is deliberate -- a copied constant
    is how the two drifted apart in the first place.
    """
    els = [str(e) for e in Composition(cp_).elements]
    v = vec(cp_)
    if v != 18:
        return f"OFF-DOMAIN: VEC {v} != 18"
    rad = [e for e in els if e in RADIO]
    if rad:
        return f"REFUSE: radioactive/actinide ({','.join(rad)})"
    fl = [e for e in els if e in FLUCT]
    if fl:
        return f"FLAGGED: {FLUCT[fl[0]]}"
    return ""


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    tr = pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv", low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    m = tr.method.astype(str).str.lower()
    bte = tr[(tr.t == 1) & m.str.contains(BTE, regex=True) & ~m.str.contains(NOT, regex=True)
             & (tr.k > 0)]

    D = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    Dh = D[D.klass == "half"].copy()
    Dh["_v"] = Dh.compound.map(vec)
    _s = {x: structure_status(str(x)) for x in Dh.compound.unique()}
    c, p = E.fit_cp(Dh[(Dh._v == 18) & Dh.compound.map(lambda x: _s[x][0] and not _s[x][1])])
    cf = json.load(open("data/exports/kappa_v2/conformal_indomain.json"))["levels"]
    b50, b90 = cf["0.50"]["factor"], cf["0.90"]["factor"]
    print(f"calibration c={c:.3f}, p={p:.3f}   bands x/div {b50} (50%), {b90} (90%)\n")

    rows = []
    for cp_, fam in COND.items():
        s = bte[bte.red == cp_]
        near = s.iloc[(s["T"] - 300).abs().values.argsort()[:1]]
        if not len(near) or abs(float(near["T"].iloc[0]) - 300) > 150:
            print(f"  {cp_}: no calculation near 300 K -- skipped")
            continue
        kb = float(near.k.iloc[0])
        k3 = kb * min(c, 1.0)
        srcs = s[s["T"].between(250, 350)].groupby("source_doi").k.median()
        spread = float(srcs.max() / srcs.min()) if len(srcs) > 1 else 1.0
        rows.append(dict(compound=cp_, family=fam, x_site=(cluster_of(cp_) or "?").replace("half:X=", ""),
                         kappa_BTE_300=round(kb, 2), kappa_pred_300=round(k3, 2),
                         lo50=round(k3 / b50, 2), hi50=round(k3 * b50, 2),
                         lo90=round(k3 / b90, 2), hi90=round(k3 * b90, 2),
                         n_bte_rows=len(s), n_sources=len(srcs),
                         source_spread=round(spread, 2),
                         status="CONDITIONAL", caveat=caveat_for(cp_),
                         unlocked_by=UNLOCKED_BY[fam]))
    R = pd.DataFrame(rows).sort_values(["family", "kappa_pred_300"])
    R.to_csv(OUT, index=False)

    print(f"  {'compound':<9}{'family':<8}{'X':<4}{'BTE300':>8}{'pred':>8}"
          f"{'50% range':>13}{'90% range':>14}{'src':>5}{'spread':>8}")
    for r in R.itertuples():
        print(f"  {r.compound:<9}{r.family:<8}{r.x_site:<4}{r.kappa_BTE_300:>8.2f}"
              f"{r.kappa_pred_300:>8.2f}{f'{r.lo50}-{r.hi50}':>13}"
              f"{f'{r.lo90}-{r.hi90}':>14}{r.n_sources:>5}{r.source_spread:>7.2f}x")
    print(f"\n  {len(R)} conditional predictions across {R.family.nunique()} families")
    for f, g in R.groupby("family"):
        print(f"    {f:<7}{len(g)} compounds, released by {UNLOCKED_BY[f]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
