"""The actual deliverable: experimental-scale kappa_L for the novel half Heuslers, with gates.

TWO BUGS THIS FIXES in `predict_novel_targets.py`, both of which made its numbers unusable as
experimental predictions:

1. NO CALIBRATION WAS APPLIED. That script trains on tier-1 BTE labels and reports the model's raw
   output, which lives on the DFT scale. Our own blind test measures raw DFT at ~92% away from
   experiment and the raw model at 1.66x high. The validated pipeline is
   `kappa_expt = kappa_DFT * min(c*(T/300)^p, 1)` with c = 0.480, p = +0.350, fit with the
   compound's whole chemistry withheld from both model and calibration. Without it `ZrGaCu` reads
   18.24 W/m/K where the calibrated estimate is 8.76.

2. THE INTERVAL CAME FROM THE WRONG POPULATION. Its band is a conformal band over FULL-Heusler
   out-of-fold residuals (`conformal((b["y"] - oof)[isfull])`), so a half-Heusler prediction was
   handed a full-Heusler band built from DFT-vs-DFT cross-validation. That measures how well the
   model reproduces other calculations, not how well it predicts a measurement. The bands here come
   from `validate_intervals.py`, calibrated on individual model-vs-experiment residuals with the
   chemistry cluster held out, and checked: 50/80/90 nominal -> 51/83/90 actual.

A THIRD TRAP, found while writing this. The spreadsheet holds all 25 CANDIDATES, but three of them
(`FeGeMo`, `NbGaNi`, `NbInNi`) were re-verified as already published. An earlier version of this
printout issued 7 compounds because those three were still counted as novel. Only the 22 rows whose
verdict in NOVELTY_REVERIFIED.csv is `novel` may appear here.

THE GATE. A prediction is issued only where the X-site chemistry has >= 3 tier-1 compounds behind
it. That threshold is not arbitrary: chemistries with 0-2 supporting compounds score 60% median
blind error against 29-42% above it. Below the gate we print a refusal and the reason, which for Zn
and Ag is that nobody has ever measured OR computed kappa for that site.
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

from run_loco_chemistry import cluster_of, sites

RAW = "data/Target_Materials/NOVEL_25_PREDICTIONS.xlsx"
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
IVAL = "data/exports/kappa_v2/interval_validation.json"
VERIFIED = "data/Target_Materials/NOVELTY_REVERIFIED.csv"
OUT = "data/Target_Materials/NOVEL_HALF_HEUSLER_PREDICTIONS.csv"

SUPPORT_MIN = 3
BAND_KEY = "half, support>=3"
WHY_BLOCKED = {
    "half:X=Zn": "no Zn-site half Heusler has a kappa label at any tier (64 structures known, 2 "
                 "with kappa) - nothing to learn the chemistry from",
    "half:X=Ag": "no Ag-site half Heusler has a kappa label at any tier (18 structures known, 0 "
                 "with kappa) - the chemistry is absent from the training pool entirely",
    "half:X=Mn": "only 2 Mn-site compounds carry a tier-1 label, below the 3-compound floor where "
                 "blind error jumps to ~60%",
}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def fitted_cp() -> tuple:
    """Refit (c, p) here rather than carry constants in the source.

    This file previously hardcoded c=0.480, p=0.350 as "published constants". They had gone stale:
    every other script in the project refits, and the current half-Heusler fit is c=0.500,
    p=0.950. A prediction file quoting one calibration while the paper quotes another is the
    defect that got three earlier headline numbers withdrawn.
    """
    import extend_blind_test as E
    from make_paper_predictions import structure_status, vec
    d = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    h = d[d.klass == "half"].copy()
    # Fit on the domain the constants are applied to, exactly as make_paper_predictions does.
    # Fitting on all half Heuslers here while the manuscript reports the in-domain fit would put
    # two different calibrations in one paper.
    h["_v"] = h.compound.map(vec)
    _s = {x: structure_status(str(x)) for x in h.compound.unique()}
    h["_i"] = (h._v == 18) & h.compound.map(lambda x: _s[x][0] and not _s[x][1])
    c, p = E.fit_cp(h[h._i])
    assert np.isfinite(c) and np.isfinite(p), "calibration fit failed - refusing stale constants"
    return float(c), float(p)


def calib(k, T, c, p):
    """The form re-selected in 73% of nested splits; capped at 1 so it can never scale kappa up."""
    return k * min(c * (T / 300.0) ** p, 1.0)


def main() -> int:
    raw = pd.read_excel(RAW, sheet_name="All_temperatures")
    raw = raw[raw.model == "high_confidence"].copy()   # tier-1 only: the validated basis
    raw = raw.rename(columns={"_formula": "formula"})  # itertuples mangles leading underscores
    raw["red"] = raw.formula.map(red)

    v = pd.read_csv(VERIFIED)
    novel = set(v[v.verdict.astype(str).str.lower() == "novel"].compound.map(red))
    dropped = sorted(set(raw.red.dropna()) - novel)
    raw = raw[raw.red.isin(novel)]
    raw["klass"] = raw.formula.map(lambda f: (sites(f) or ["?"])[0])
    H = raw[raw.klass == "half"].copy()
    print(f"candidates dropped as NOT NOVEL: {dropped}")
    print(f"verified-novel half Heuslers: {H.formula.nunique()}")

    tr = pd.read_csv(TRAIN)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr = tr[tr.red.notna()]
    tr["cluster"] = tr.red.map(lambda f: cluster_of(f) if sites(f) else None)
    sup = tr[tr.tier == 1].groupby("cluster").red.nunique().to_dict()

    # Bands from the in-domain conformal rebuild, matching make_paper_predictions.py. The former
    # source (interval_validation.json, "half, support>=3") covered 46 compounds including
    # out-of-domain ones and used residuals from a pooled half+full calibration, giving a 90% band
    # of 3.21x where the in-domain evidence supports 2.24x.
    cf = json.load(open("data/exports/kappa_v2/conformal_indomain.json"))["levels"]
    b = {a: cf[a]["factor"] for a in ("0.50", "0.80", "0.90")}
    emp = {a: cf[a]["empirical_coverage_pct"] / 100.0 for a in ("0.50", "0.80", "0.90")}

    C, P = fitted_cp()
    print(f"calibration refitted: kappa_expt = kappa_DFT x min({C:.3f}(T/300)^{P:+.3f}, 1)")

    H["cluster"] = H.formula.map(lambda f: cluster_of(f) if sites(f) else None)
    H["support"] = H.cluster.map(lambda c: sup.get(c, 0))
    H["kappa_expt"] = [calib(k, T, C, P) for k, T in zip(H.kappa_L, H.temperature_K)]
    for a in ("0.50", "0.80", "0.90"):
        H[f"lo{a}"] = H.kappa_expt / b[a]
        H[f"hi{a}"] = H.kappa_expt * b[a]

    # STRUCTURE GATE. Until now the only condition on `issued` was X-site support count, so
    # ZrGaCu, HfGaCu and HfInCu shipped as issued predictions on three separate regenerations
    # after each had been identified as hexagonal and removed BY HAND from the output. Editing the
    # artifact never worked; the predicate has to live here. A cubic 216/225 record from a source
    # that is not a prototype-decoration database is the requirement, and a compound recorded in
    # BOTH a cubic and a non-cubic spacegroup is polymorphic - the cubic kappa cannot be issued as
    # though that phase is the one that forms.
    from make_paper_predictions import structure_status
    st = {r: structure_status(r) for r in H.red.dropna().unique()}
    H["_cubic_ok"] = H.red.map(lambda r: st.get(r, (False, False, []))[0])
    H["_polymorph"] = H.red.map(lambda r: st.get(r, (False, False, []))[1])
    H["_sg_seen"] = H.red.map(lambda r: str(st.get(r, (False, False, []))[2]))

    H["issued"] = (H.support >= SUPPORT_MIN) & H._cubic_ok & ~H._polymorph

    def _why(s, c, ok, poly, sgs):
        if s < SUPPORT_MIN:
            return WHY_BLOCKED.get(c, f"only {s} tier-1 compounds")
        if not ok:
            return ("no cubic Heusler (216/225) structure on record outside prototype-decoration "
                    f"databases; spacegroups seen: {sgs}")
        if poly:
            return (f"POLYMORPHIC - also recorded non-cubic in {sgs}; the DFT kappa is for the "
                    "cubic C1b phase, which may not be the phase that forms")
        return ""

    H["reason"] = [_why(s, c, ok, po, sg) for s, c, ok, po, sg
                   in zip(H.support, H.cluster, H._cubic_ok, H._polymorph, H._sg_seen)]

    keep = ["formula", "cluster", "support", "temperature_K", "kappa_L", "kappa_expt",
            "lo0.50", "hi0.50", "lo0.80", "hi0.80", "lo0.90", "hi0.90", "issued", "reason",
            "_sg", "_a", "_src", "_sg_seen"]
    H[keep].sort_values(["issued", "formula", "temperature_K"],
                        ascending=[False, True, True]).to_csv(OUT, index=False)

    print("\n" + "=" * 90)
    print("NOVEL HALF HEUSLERS -- calibrated to the EXPERIMENTAL scale")
    print("=" * 90)
    print(f"  calibration  kappa_expt = kappa_DFT x min({C:.3f}*(T/300)^{P:+.3f}, 1)"
          f"  ->  x{calib(1, 300, C, P):.3f} at 300K, x{calib(1, 900, C, P):.3f} at 900K")
    print(f"  intervals    validated on 41 half Heuslers ({BAND_KEY}), chemistry held out:")
    for a in ("0.50", "0.80", "0.90"):
        print(f"                 {float(a)*100:.0f}% nominal   x/div {b[a]:.2f}   "
              f"empirical {emp[a]*100:.1f}%")

    iss = H[H.issued & (H.temperature_K == 300)].sort_values("kappa_expt")
    print(f"\n  --- ISSUED: {iss.formula.nunique()} compounds, at 300 K ---")
    hdr = f"  {'compound':<9}{'chem':<6}{'sup':>4}{'rawDFT':>8}{'PREDICT':>9}"
    print(hdr + f"{'50% range':>15}{'90% range':>17}")
    for _, r in iss.iterrows():
        r50 = f"{r['lo0.50']:.1f} - {r['hi0.50']:.1f}"
        r90 = f"{r['lo0.90']:.1f} - {r['hi0.90']:.1f}"
        print(f"  {r.formula:<9}{r.cluster.replace('half:X=', ''):<6}{int(r.support):>4}"
              f"{r.kappa_L:>8.2f}{r.kappa_expt:>9.2f}{r50:>15}{r90:>17}")

    print(f"\n  temperature dependence of the issued set:")
    for f_, g in H[H.issued].groupby("formula"):
        row = "  ".join(f"{int(t)}K {k:.2f}" for t, k in
                        zip(g.temperature_K, g.kappa_expt))
        print(f"    {f_:<9}{row}")

    ref = H[~H.issued & (H.temperature_K == 300)]
    print(f"\n  --- REFUSED: {ref.formula.nunique()} compounds ---")
    for _, r in ref.iterrows():
        print(f"  {r.formula:<9}{r.cluster.replace('half:X=', ''):<6}tier-1 support {int(r.support)}")
        print(f"    {r.reason}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
