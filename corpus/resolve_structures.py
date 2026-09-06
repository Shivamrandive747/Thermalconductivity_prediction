"""Attach a REAL, CITED structure to every compound in the kappa pool -- or admit there isn't one.

THE RULE THIS FILE ENFORCES
---------------------------
A structure is attached only when a named source supplies it, with a resolvable identifier. Nothing
is ever synthesised, guessed, averaged from neighbours, or estimated from an ionic-radius sum. A
compound with no sourced lattice constant is marked `grade D` and EXCLUDED from the
structure-required training set -- it is not quietly filled in.

This matters because the project has already had 1,571 fabricated measurements enter the database
once (see the Elsevier coredata stub incident). An invented lattice constant would be the same
failure in a new place, and harder to spot, because a plausible-looking `a` for a Heusler is easy to
produce and impossible to distinguish from a real one after the fact.

THE GRADES -- what "has a structure" actually means
--------------------------------------------------
  A  full explicit structure: lattice + species + fractional coordinates, all from the source
  B  lattice constant + space group + the source states WHICH element sits on which site
  C  lattice constant + space group + Heusler prototype, but the source does not state the site
     ordering. The Wyckoff positions of C1b/L2_1/XA are fixed by the prototype and are standard
     crystallography, not invention -- but which of the three elements occupies 4a/4b/4c changes
     kappa, and if the source is silent we say so rather than pick one.
  D  NO sourced lattice constant anywhere. Excluded. Reported by name, never filled in.

PROVENANCE ON EVERY ROW: `struct_source`, `struct_source_url`, `struct_source_detail` (the literal
description the harvest recorded, e.g. "alatt column (AFLOW T0003 relaxed lattice constant, same
table as kappa)") and `struct_same_paper_as_kappa`.

PREFERENCE ORDER. A structure from the SAME paper/file that produced the kappa value is preferred
over one from an unrelated database, because a kappa is only consistent with the cell it was
computed in. Where they disagree by more than 2% the disagreement is reported, not silently resolved.

PRIMITIVE vs CONVENTIONAL -- MEASURED, NOT ASSUMED
--------------------------------------------------
A first run showed 524 of 622 compounds "disagreeing" by more than 2%, worst 66%. That was not bad
data: some sources publish the PRIMITIVE fcc edge and others the CONVENTIONAL cubic edge, which
differ by exactly sqrt(2). Rather than assume which is which, every source was regressed against
JARVIS's explicitly-named `a_conventional_cubic_A`:

    source            n shared   median ratio    verdict
    MP                    480         1.0022     conventional
    OQMD                   21         1.0026     conventional
    Carrete 2014           56         1.0036     conventional
    Sci.Rep. 2021         103         0.9973     conventional
    repository mining     154         1.0200     conventional
    PhononDB `a`           44         1.0126     conventional
    PhononDB `prim_a`      44         1.4320     PRIMITIVE
    AFLOW `geometry[0]`   289         1.4103     PRIMITIVE     (sqrt2 = 1.4142)
    MP `structure.lattice.a` 401      1.4113     PRIMITIVE  <- despite the bare name `a`
    HH130 `prim_a`        110         1.4159     PRIMITIVE

Four columns are primitive and are multiplied by sqrt(2); everything else is left alone. The last
two were caught only on a second pass, when the cross-check 90th percentile sat at 43.8% -- almost
exactly sqrt(2)-1. Every row records `lattice_convention` so the conversion is visible and
reversible. This project once made the OPPOSITE error -- applying sqrt(2) where both sides were
already primitive, which manufactured a fake 29% mismatch -- which is why the direction is measured
each run rather than remembered.

AFTER all four corrections and the polymorph gate, 600 compounds carry two or more independent
lattice constants and agree to a median of 1.82%, 90% within 5%, 99% within 10%. That residue is
ordinary exchange-correlation-functional scatter (PBE vs PBEsol vs LDA), not error.
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from pymatgen.core import Composition

POOL = "data/external/KAPPA_POOL_MASTER.csv"
OUT = "data/external/KAPPA_POOL_WITH_STRUCTURE.csv"
NOSTRUCT = "data/external/KAPPA_POOL_no_structure_EXCLUDED.csv"

# Heusler prototypes: space group -> the conventional-cell arrangement is standard, but the
# site ORDERING is what the source must tell us.
PROTO_SG = {"half_C1b": 216, "C1b": 216, "full_L21": 225, "L21": 225, "inverse_XA": 216}
# The only space groups a Heusler can be in. Anything else with the same formula is a different
# polymorph and must not lend its lattice constant (see the polymorph gate in norm()).
HEUSLER_SG = {216, 225, 119, 139}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def num(x):
    return pd.to_numeric(x, errors="coerce")


def _first(d: pd.DataFrame, *names):
    for n in names:
        if n in d.columns:
            return d[n]
    return pd.Series([None] * len(d), index=d.index)


SQRT2 = 2.0 ** 0.5


def norm(path: str, *, src_label: str, formula, a=None, a_nm=None, prim_a=None,
         primitive: bool = False, same_paper: bool = False) -> pd.DataFrame | None:
    """Pull one harvest into the common structure schema, normalised to the CONVENTIONAL cubic
    lattice constant. `primitive=True` marks a source measured to publish the primitive fcc edge
    (see module docstring); its value is multiplied by sqrt(2) and the conversion is recorded."""
    p = Path(path)
    if not p.exists():
        print(f"  MISSING {path}")
        return None
    d = pd.read_csv(p)
    g = pd.DataFrame(index=d.index)
    g["formula"] = _first(d, *([formula] if isinstance(formula, str) else formula)).map(red)

    # lattice constant, in Angstrom, from whichever column this harvest uses
    lat = pd.Series([pd.NA] * len(d), index=d.index, dtype="object")
    conv = "conventional"
    if a:
        lat = lat.fillna(num(_first(d, *([a] if isinstance(a, str) else a))))
    if a_nm and a_nm in d.columns:
        lat = lat.fillna(num(d[a_nm]) * 10.0)          # nm -> Angstrom
    if prim_a and prim_a in d.columns:
        lat = lat.fillna(num(d[prim_a]) * SQRT2)
        conv = "primitive->conventional (x sqrt2)"
    lat = num(lat)
    if primitive:
        lat = lat * SQRT2
        conv = "primitive->conventional (x sqrt2)"
    g["a_A"] = lat
    g["lattice_convention"] = conv

    sg_raw = _first(d, "spacegroup_number", "spacegroup", "sg", "space_group")
    sg_num = num(sg_raw)
    if sg_num.isna().all():
        # some papers report it as text, e.g. "F-43m (216)"
        sg_num = num(sg_raw.astype(str).str.extract(r"\((\d{1,3})\)")[0])
    g["spacegroup"] = sg_num
    g["sg_symbol"] = _first(d, "spacegroup_symbol")
    g["species"] = _first(d, "species")
    g["frac_coords"] = _first(d, "frac_coords")
    g["prototype"] = _first(d, "prototype", "heusler_class", "heusler_type")
    g["sites"] = _first(d, "site_ordering_code")
    for c in ("site_4a", "site_4b", "site_4c"):
        if c in d.columns:
            g["sites"] = g["sites"].fillna(
                d.get("site_4a").astype(str) + "/" + d.get("site_4b").astype(str)
                + "/" + d.get("site_4c").astype(str))
            break
    g["struct_source"] = src_label
    g["struct_source_url"] = _first(d, "source_url", "aurl")
    g["struct_source_detail"] = _first(d, "structure_source", "source", "record_identifier")
    g["struct_id"] = _first(d, "material_id", "entry_id", "source_id", "auid", "record_identifier")
    g["same_paper"] = same_paper
    g = g[g.formula.notna() & g.a_A.notna() & (g.a_A > 0)]

    # THE POLYMORPH GATE. Matching on formula alone attached hexagonal entries to cubic
    # half-Heuslers: AFLOW holds CaAgP in space group 189 (P-62m, Fe2P-type) at a = 7.11 A,
    # a genuinely DIFFERENT material from the F-43m half-Heusler whose kappa we hold. Left in,
    # it read as a 120% lattice "disagreement". A structure only counts if it is in a Heusler
    # space group; anything else is a different polymorph and is dropped, not reconciled.
    if "spacegroup" in g.columns:
        sg = num(g.spacegroup)
        wrong = sg.notna() & ~sg.isin(HEUSLER_SG)
        if int(wrong.sum()):
            print(f"    dropped {int(wrong.sum())} non-Heusler-spacegroup rows "
                  f"(polymorphs: {sorted(sg[wrong].dropna().unique().astype(int))[:8]})")
            g = g[~wrong]
    print(f"  {p.name:<46}{len(g):>6} sourced structures  ({g.formula.nunique()} formulae)")
    return g


def grade(r) -> str:
    has_coords = isinstance(r.frac_coords, str) and len(str(r.frac_coords)) > 10 \
        and isinstance(r.species, str) and len(str(r.species)) > 2
    if has_coords:
        return "A"
    if isinstance(r.sites, str) and len(str(r.sites)) > 2 and str(r.sites).lower() != "nan":
        return "B"
    return "C"


def main() -> int:
    print("SOURCED STRUCTURE REGISTRY (nothing here is generated -- every row came from a file)")
    regs = [
        # structure published in the SAME table/file as the kappa value -- preferred
        norm("data/external/kappa_bte_phonondb_heusler_multiT.csv",
             src_label="PhononDB (Togo/NIMS MDR)", formula="formula", a="a", prim_a="prim_a",
             same_paper=True),
        norm("data/external/kappa_bte_scirep2021_halfheusler143.csv",
             src_label="Sci.Rep. 11:13809 (2021) Table S1", formula="formula", a="a",
             a_nm="lattice_parameter_nm", same_paper=True),
        norm("data/external/kappa_bte_carrete2014_halfheusler.csv",
             src_label="Carrete PRX 4:011019 (2014)", formula="formula", a="a", prim_a="prim_a",
             same_paper=True),
        norm("data/external/kappa_bte_npj2025_heusler.csv",
             src_label="npj Comput.Mater. (2025)", formula="formula", a="a", same_paper=True),
        norm("data/external/kappa_repos_mined.csv",
             src_label="repository mining", formula="formula", a="a_conventional_A",
             same_paper=True),
        norm("data/external/kappa_papers_mined.csv",
             src_label="paper mining", formula="formula", a="a_A", same_paper=True),
        # PDF/SI extractions carry `lattice_parameter_A` straight from the paper's own table --
        # the best provenance available, because the lattice and the kappa come from the same
        # table. Missing this call cost RbNbZn and RbNbCd, which were dropped for "no structure"
        # while their paper printed a = 7.39 A and 7.68 A next to the kappa values.
        norm("data/external/kappa_pdf_batch2.csv",
             src_label="paper mining (PDF)", formula="formula", a="lattice_parameter_A",
             same_paper=True),
        norm("data/external/heusler_structures_HH130_NO_KAPPA.csv",
             src_label="HH130 / MatHub-3d", formula="formula", prim_a="prim_a",
             same_paper=False),
        # independent databases, each with a resolvable entry id
        norm("data/external/aflow_agl_heusler.csv", src_label="AFLOW",
             formula=["reduced_formula", "compound"], a="a", primitive=True),
        norm("data/external/theory_kappa_pool.csv", src_label="AFLOW",
             formula=["reduced_formula", "compound"], a="a", primitive=True),
        # MP's `structure.lattice.a` is the PRIMITIVE fcc edge, measured at ratio 1.4113 vs
        # JARVIS conventional -- not the conventional cell, despite the bare name `a`.
        norm("data/external/mp_heusler_elastic.csv", src_label="Materials Project",
             formula=["reduced_formula", "compound"], a="a", primitive=True),
        norm("data/external/mp_optimade_structures_stability.csv", src_label="Materials Project",
             formula=["formula_reduced", "formula_reported"], a="a_conventional_cubic_A"),
        norm("data/external/oqmd_structures_stability.csv", src_label="OQMD",
             formula=["formula_reduced", "formula_reported"], a="a_conventional_cubic_A"),
        norm("data/external/jarvis_dft_thermal_elastic.csv", src_label="JARVIS-DFT (NIST)",
             formula=["formula_reduced", "formula_reported"], a="a_conventional_cubic_A"),
        norm("data/external/nomad_mechanical_phonon.csv", src_label="NOMAD",
             formula=["formula_reduced", "formula_reported"], a="a_conventional_cubic_A"),
        # Structures fetched from MP specifically for compounds no other source could resolve.
        # MP's `structure.lattice.a` is the PRIMITIVE edge -- see the convention table above.
        norm("data/external/structures_recovered_mp.csv", src_label="Materials Project",
             formula="formula", a="a_A", primitive=True),
        # Structures pulled from the OPTIMADE federation for compounds that had a measured or
        # computed kappa but NO structure anywhere in our files -- 20 of them were being dropped
        # at this gate, including TiSbRu2, which is the Ru2 chemistry the model is starved of.
        # `a` here is ALREADY the conventional cubic edge: fetch_structures_optimade.py builds a
        # pymatgen Structure from the raw lattice vectors and runs SpacegroupAnalyzer, so the
        # primitive-vs-conventional question is settled by computation rather than by trusting
        # whichever convention a provider happens to publish. Hence primitive=False.
        norm("data/external/optimade_recovered_structures.csv", src_label="OPTIMADE",
             formula=["formula_reduced", "formula_reported"], a="a"),
    ]
    reg = pd.concat([r for r in regs if r is not None], ignore_index=True)
    reg["grade"] = reg.apply(grade, axis=1)
    # best first: same-paper, then grade A > B > C
    reg["_rank"] = (~reg.same_paper).astype(int) * 10 + reg.grade.map({"A": 0, "B": 1, "C": 2})
    reg = reg.sort_values("_rank")
    print(f"\n  registry: {len(reg)} sourced structures covering {reg.formula.nunique()} formulae")

    pool = pd.read_csv(POOL)
    compounds = sorted(pool.formula.dropna().unique())
    print(f"  pool compounds needing a structure: {len(compounds)}")

    best = reg.drop_duplicates("formula").set_index("formula")
    hit = best.reindex(compounds)
    hit["has_struct"] = hit.a_A.notna()

    print(f"\n{'=' * 74}\nSTRUCTURE RESOLUTION\n{'=' * 74}")
    n_ok = int(hit.has_struct.sum())
    print(f"  WITH a real, cited structure : {n_ok} / {len(compounds)}")
    print(f"  WITHOUT one (grade D)        : {len(compounds) - n_ok}  -> EXCLUDED, not invented")

    print("\n  by evidence grade:")
    gl = {"A": "A  full structure (lattice + species + coordinates)",
          "B": "B  lattice + space group + stated site ordering",
          "C": "C  lattice + space group + prototype, site ordering not stated"}
    for g_, n in hit[hit.has_struct].grade.value_counts().sort_index().items():
        print(f"    {gl.get(g_, g_):<58}{n:>5}")

    print("\n  by source of the structure:")
    for s, n in hit[hit.has_struct].struct_source.value_counts().items():
        same = int(hit[(hit.struct_source == s) & hit.has_struct].same_paper.sum())
        print(f"    {str(s)[:44]:<46}{n:>5}   ({same} from the same paper as kappa)")

    # ---- cross-check: where two independent sources give the same compound a lattice -----
    multi = reg[reg.formula.isin(compounds)].groupby("formula").a_A.agg(["min", "max", "count"])
    multi = multi[multi["count"] > 1]
    if len(multi):
        spread = (multi["max"] - multi["min"]) / multi["min"] * 100
        print(f"\n  CROSS-CHECK on {len(multi)} compounds with 2+ independent lattice constants:")
        print(f"    median spread {spread.median():.2f}%   90th pct {spread.quantile(0.90):.2f}%"
              f"   max {spread.max():.1f}%")
        for th in (1, 2, 5, 10):
            print(f"    within {th:>2}% : {int((spread <= th).sum()):>4} "
                  f"({(spread <= th).mean() * 100:.0f}%)")
        # A few percent between databases is ordinary exchange-correlation-functional scatter
        # (PBE vs PBEsol vs LDA), not error. Only a gross outlier means a wrong structure, and
        # the polymorph gate above already removes the usual cause of those.
        worst = spread.sort_values(ascending=False).head(5)
        print(f"    largest remaining: {[f'{f} {v:.0f}%' for f, v in worst.items()]}")

    out = pool.merge(
        hit[["a_A", "lattice_convention", "spacegroup", "sg_symbol", "prototype", "sites",
             "species", "frac_coords", "grade", "struct_source", "struct_source_url",
             "struct_source_detail", "struct_id", "same_paper", "has_struct"]].rename(columns={
                 "a_A": "struct_a_A", "spacegroup": "struct_spacegroup",
                 "sg_symbol": "struct_sg_symbol", "prototype": "struct_prototype",
                 "sites": "struct_site_ordering", "species": "struct_species",
                 "frac_coords": "struct_frac_coords", "grade": "struct_grade",
                 "same_paper": "struct_same_paper_as_kappa"}),
        left_on="formula", right_index=True, how="left")
    out["has_struct"] = out.has_struct.fillna(False)

    keep = out[out.has_struct]
    drop = out[~out.has_struct]
    print(f"\n  TRAINING-READY (kappa + temperature + cited structure): "
          f"{keep.formula.nunique()} compounds, {len(keep)} rows")
    if len(drop):
        names = sorted(drop.formula.unique())
        print(f"  EXCLUDED for having no sourced structure: {len(names)} compounds")
        print(f"    {names[:24]}")
        drop.to_csv(NOSTRUCT, index=False)
        print(f"    written to {NOSTRUCT} (kept on disk, excluded from training)")

    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
