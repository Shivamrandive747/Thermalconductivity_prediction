"""Add 7 experimentally-measured half Heuslers that were extracted years ago and then silently
dropped for missing structure metadata.

WHERE THESE CAME FROM. Not from new mining -- from our own `heusler.sqlite`. It holds 152 half-Heusler
compounds with a lattice-kappa value, but only a subset ever reached the training set, because
`resolve_structures.py` requires a sourced lattice constant and discards anything without one. The
measurements were fine; the structure metadata was thin. That is a pipeline gap, not a data gap, and
it will keep discarding future extractions until it is fixed.

FILTERING THAT MATTERS. The database mixes computed and measured rows in the same table -- 565 of
752 half-Heusler lattice-kappa rows are `data_origin = 'theoretical'` (DFT, ShengBTE, Slack's
model). An earlier pass that skipped this filter produced 83 "recoverable" compounds, nearly all of
them DFT. With the filter it is 7. `ZrSbIr` is the live trap: the same paper reports BOTH measured
and first-principles values, so it must be filtered per row, not per compound. `LuNiBi` looked like
an ultralow-kappa prize at 0.11-0.35 W/m/K until the same filter showed every row was theoretical --
its source (10.1039/d1tc02819g) is a pure DFT paper.

`MnSiNi` is excluded: its single 0.22 W/m/K point is documented in this project's own code as a
digitisation artefact.

LATTICE CONSTANTS, and the sqrt(2) trap. Four compounds carry XRD-refined constants from their own
papers. Three needed sourcing, and the sources disagree in a way that is entirely explained by
convention: AFLOW and JARVIS publish the PRIMITIVE fcc edge, others the CONVENTIONAL cubic edge, and
the ratio is sqrt(2). Two independent cross-checks confirm the reading rather than assuming it:

    LuSbPt   AFLOW 4.5799 x sqrt2 = 6.477   vs experimental 6.4677 (Hou 2015, Rietveld)
    AlVNi    JARVIS 4.0321 x sqrt2 = 5.702  vs our own XRD-refined 5.70

Every value below is the CONVENTIONAL cubic edge, and the provenance of each is recorded.

WHAT THIS IS AND IS NOT WORTH. It takes the experimental set from 46 to 53 compounds, ~15%. That
will not move the headline accuracy -- with 46 compounds the noise band is about +/-7 percentage
points and 7 more does not change that. What it does is thicken the 1.8-3.8 W/m/K region, where our
error is worst and our coverage thinnest, and it costs nothing because the data was already paid for.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT_NEW = "data/external/RECOVERED_from_db_rows.csv"

# conventional cubic edge (angstrom), with where each came from
LATTICE = {
    "ZrSbIr": (6.292, "XRD refined, 10.1088/1361-6463/ac1dd8"),
    "AlNiSb": (6.010, "XRD refined, 10.1016/j.jpcs.2023.111854"),
    "AlVNi":  (5.700, "XRD refined, 10.1016/j.jpcs.2023.111854; JARVIS primitive 4.0321 x sqrt2 = 5.702"),
    "VNiSn":  (6.030, "XRD refined, 10.1016/j.jpcs.2023.111854"),
    "LuSbPt": (6.4677, "experimental Rietveld, Hou et al. APL 106 102102 (2015); AFLOW primitive 4.5799 x sqrt2 = 6.477"),
    "TbSbPt": (6.620, "MP mp-16313 DFT 6.64 and AFLOW primitive 4.6715 x sqrt2 = 6.607; mean of the two"),
    "DySbPt": (6.590, "pool 6.5147 / HH130 6.6115 / AFLOW primitive 4.6579 x sqrt2 = 6.587; mean"),
}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def c1b_cell(formula):
    """Conventional C1b cell: X at 4a, Y at 4c, Z at 4b, sites assigned by electronegativity.

    The same convention `s19_kappa_dataset` uses -- most electropositive element to X, most
    electronegative to Z -- so the generated structure is consistent with how every other compound
    in the training set was featurised.
    """
    c = Composition(formula)
    els = sorted(c.elements, key=lambda e: (e.X if e.X else 99.0))
    X, Y, Z = [str(e) for e in els[:3]]
    base = {"X": [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)],
            "Y": [(.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)],
            "Z": [(.5, .5, .5), (.5, 0, 0), (0, .5, 0), (0, 0, .5)]}
    species, coords = [], []
    for sym, key in ((X, "X"), (Y, "Y"), (Z, "Z")):
        for p in base[key]:
            species.append(sym)
            coords.append(list(p))
    return species, coords


def main() -> int:
    # DB-FREE PATH. These 31 rows are the ONLY place the 98 MB heusler.sqlite enters the corpus
    # chain (0.76% of the experimental rows), and that database is far too large -- and too full of
    # publisher-derived extractions -- to publish. So if the database is absent but the frozen
    # output of a previous run is present, use that instead. The public release ships the frozen
    # CSV, which makes the whole corpus chain reproducible without the database.
    if not Path("data/heusler.sqlite").exists() and Path(OUT_NEW).exists():
        print(f"heusler.sqlite not present -- using the frozen {OUT_NEW}")
        N = pd.read_csv(OUT_NEW)
        tr = pd.read_csv(TRAIN)
        tr["red"] = tr.formula.map(red)
        already = set(tr[pd.to_numeric(tr.method_tier, errors="coerce") == 0].red.dropna())
        N = N[~N.formula.map(red).isin(already)]
        if N.empty:
            print("  every recovered compound is already in the training set -- nothing to add")
            return 0
        merged = pd.concat([tr.drop(columns=["red"]), N], ignore_index=True)
        merged.to_csv(TRAIN, index=False)
        print(f"  added {len(N)} rows for {N.formula.map(red).nunique()} compounds")
        print(f"updated {TRAIN}")
        return 0

    c = sqlite3.connect("data/heusler.sqlite")
    d = pd.read_sql("""select m.canonical_value val, m.temperature_k tk, m.doi,
        m.data_origin origin, m.measurement_method meth, m.location_in_paper lip,
        s.reduced_formula rf, s.composition comp, s.phase_purity pur
        from measurements m join samples s on m.sample_id = s.sample_id
        where m.property_name = 'thermal_conductivity_lattice'""", c)
    d["key"] = d.rf.fillna(d.comp).astype(str).str.replace(" ", "")
    d["red"] = d.key.map(red)

    # EXPERIMENTAL ONLY -- ZrSbIr carries theoretical rows from the same paper
    d = d[d.origin.astype(str).str.lower().str.contains("experiment", na=False)]
    d = d[d.val.notna() & (d.val > 0) & d.tk.between(200, 1300)]
    d = d[d.red.isin({red(k) for k in LATTICE})]

    tr = pd.read_csv(TRAIN)
    tr["red"] = tr.formula.map(red)
    already = set(tr[pd.to_numeric(tr.method_tier, errors="coerce") == 0].red.dropna())
    d = d[~d.red.isin(already)]
    print(f"experimental lattice-kappa rows to add: {len(d)}   compounds: {d.red.nunique()}")

    rows = []
    for r in d.itertuples():
        a, asrc = LATTICE[[k for k in LATTICE if red(k) == r.red][0]]
        sp, co = c1b_cell(r.red)
        multi = "multi" in str(r.pur).lower()
        rows.append(dict(
            formula=r.red, kappa_L=round(float(r.val), 4), kappa_units="W/m/K",
            temperature_K=float(r.tk),
            method=f"experimental, {r.meth}", method_tier=0, tier_name="experimental",
            weight=0.7 if multi else 1.0,
            kappa_basis="wf_subtracted",
            struct_a_A=a, struct_spacegroup=216, struct_sg_symbol="F-43m",
            struct_prototype="half_Heusler_C1b", struct_site_ordering="X:4a Y:4c Z:4b",
            struct_species=str(sp), struct_frac_coords=str(co), struct_grade="B",
            lattice_convention="conventional",
            source="recovered from heusler.sqlite (extracted, never merged)",
            source_doi=r.doi, source_url=f"https://doi.org/{r.doi}",
            source_location=r.lip,
            struct_source="C1b prototype + sourced lattice constant",
            struct_source_url="", struct_source_detail=asrc, struct_id="",
            struct_same_paper_as_kappa=(a is not None and "XRD refined" in asrc),
            stoichiometry="stoichiometric", parent_formula=r.red,
            phase_note="multiphase sample" if multi else "single-phase"))
    N = pd.DataFrame(rows)
    N.to_csv(OUT_NEW, index=False)

    print(f"\n{'compound':<9}{'rows':>5}{'T range':>13}{'kappa':>14}{'a':>8}{'weight':>8}  note")
    for k, g in N.groupby("formula"):
        print(f"{k:<9}{len(g):>5}{f'{g.temperature_K.min():.0f}-{g.temperature_K.max():.0f}K':>13}"
              f"{f'{g.kappa_L.min():.2f}-{g.kappa_L.max():.2f}':>14}{g.struct_a_A.iloc[0]:>8.3f}"
              f"{g.weight.iloc[0]:>8.1f}  {g.phase_note.iloc[0]}")

    keep = [c for c in tr.columns if c != "red"]
    for col in keep:
        if col not in N.columns:
            N[col] = pd.NA
    merged = pd.concat([tr[keep], N[keep]], ignore_index=True)
    merged.to_csv(TRAIN, index=False)
    t0 = pd.to_numeric(merged.method_tier, errors="coerce") == 0
    print(f"\ntraining set: {len(tr)} -> {len(merged)} rows")
    print(f"tier-0 compounds: {len(already)} -> {merged[t0].formula.map(red).nunique()}")
    print(f"wrote {OUT_NEW} and updated {TRAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
