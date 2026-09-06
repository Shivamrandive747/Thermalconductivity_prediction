"""Recover the doped full-Heusler measurements that the stoichiometry gate discards.

WHAT THIS IS FOR. `build_kappa_pool.py` accepts a compound only when its stoichiometry rounds to
whole numbers, so every substituted sample is thrown away. That gate is correct for the pool -- a
model that sees `Fe2VAl` at 15 W/m/K and `Al0.9V1Fe2Si0.1` at 6 W/m/K, with near-identical
features, learns that the property is noise. But the data itself is real: 97 compounds and 227
measured lattice-kappa points, 157 of them experimental, concentrated in exactly the Fe2 and Ru2
chemistry the training set is starved of.

They become usable the moment the disorder is made VISIBLE to the model, which is what the Klemens
parameters in s19 do. This script only recovers the rows; it does not decide whether they help.
That is settled out-of-fold in s20.

TWO RULES THAT KEEP THIS HONEST:

1. LATTICE KAPPA ONLY. The database holds three physically distinct properties under names
   beginning `thermal_conductivity`. Only the lattice component is what we predict. Of 788 doped
   rows, 484 are total and 71 electronic; **227 are lattice**. The total-minus-electronic recovery
   that worked for the validation set yields nothing here -- the two are never reported for the
   same sample at the same temperature -- so it is attempted and reported, not assumed.

2. NO INVENTED STRUCTURES. A doped sample has no structure entry of its own. It inherits its
   parent's, which is defensible because a few per cent substitution moves the lattice constant by
   well under a per cent -- and the inherited value is stamped `struct_source_detail` so it is
   never mistaken for a measurement of that sample. A doped compound whose parent has no real
   structure is DROPPED. Nothing is fabricated.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from pymatgen.core import Composition

DB = "file:data/heusler.sqlite?mode=ro"
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT = "data/Target_Materials/HEUSLER_KAPPA_DOPED_ROWS.csv"

# A doped sample is not a pristine calculation. It is a real measurement of a real, imperfect
# sample -- worth less than a clean BTE label but far more than nothing. Tier 0 marks experimental
# provenance; the weight is what tells the model how much to trust it.
DOPED_WEIGHT = 0.6


def element_system(f) -> str | None:
    try:
        return "-".join(sorted(str(e) for e in Composition(str(f)).elements))
    except Exception:  # noqa: BLE001
        return None


def parent_system(f) -> str | None:
    """The element system of the PARENT compound -- the three MAJOR elements.

    Matching on the full element set was wrong and cost 196 of 219 rows: a dopant adds a fourth
    element, so `Al0.85V1.15Fe1.85Cu0.15` has the system Al-Cu-Fe-V and never matches its actual
    parent Fe2VAl (Al-Fe-V). A Heusler has three crystallographic sites; whatever else is present
    is a substituent sitting on one of them. Taking the three largest amounts recovers the parent.
    """
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return None
    amt = sorted(c.get_el_amt_dict().items(), key=lambda kv: -kv[1])
    if len(amt) < 3:
        return None
    return "-".join(sorted(e for e, _ in amt[:3]))


def full_heusler_kind(f) -> str | None:
    """'pure' / 'doped' when the composition is full-Heusler-shaped (2:1:1 per 4 atoms)."""
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return None
    els = c.get_el_amt_dict()
    if not (3 <= len(els) <= 5):
        return None
    tot = sum(els.values())
    if tot <= 0:
        return None
    norm = sorted([x * 4.0 / tot for x in els.values()], reverse=True)
    if abs(norm[0] - 2.0) > 0.35:
        return None
    return "pure" if all(abs(x - round(x)) < 0.02 for x in norm) else "doped"


def resolve_lattice(d: pd.DataFrame) -> pd.DataFrame:
    """Reduce the three thermal-conductivity properties to the one we predict."""
    key = ["sample_id", "formula", "doi", "T"]
    piv = d.pivot_table(index=key, columns="property", values="kappa", aggfunc="median")
    lat, tot, ele = ("thermal_conductivity_lattice", "thermal_conductivity_total",
                     "thermal_conductivity_electronic")
    for c in (lat, tot, ele):
        if c not in piv.columns:
            piv[c] = float("nan")
    value = pd.Series(float("nan"), index=piv.index)
    basis = pd.Series("none", index=piv.index)

    m_rep = piv[lat].notna() & (piv[lat] > 0)
    value[m_rep] = piv[lat][m_rep]
    basis[m_rep] = "reported_lattice"

    derived = piv[tot] - piv[ele]
    m_sub = (~m_rep) & derived.notna() & (derived > 0)
    value[m_sub] = derived[m_sub]
    basis[m_sub] = "total_minus_electronic"

    out = piv.assign(kappa_L=value, kappa_basis=basis).reset_index()
    print(f"  reported lattice          : {int(m_rep.sum())}")
    print(f"  recovered total-electronic: {int(m_sub.sum())}")
    print(f"  neither available (dropped): {int((~(m_rep | m_sub)).sum())}")
    return out[out.kappa_L.notna()]


def main(min_T: float = 100.0) -> int:
    con = sqlite3.connect(DB, uri=True)
    q = """
        select s.sample_id, s.reduced_formula, m.doi, m.property_name,
               m.canonical_value, m.temperature_k, m.data_origin
        from samples s join measurements m on m.sample_id = s.sample_id
        where m.property_name like 'thermal_conductivity%'
          and m.canonical_value is not null and m.temperature_k is not null
    """
    d = pd.DataFrame(con.execute(q).fetchall(), columns=[
        "sample_id", "formula", "doi", "property", "kappa", "T", "origin"])
    d["kind"] = d.formula.map(full_heusler_kind)
    d = d[d.kind == "doped"]
    print(f"doped full-Heusler-shaped rows in DB: {len(d)}  "
          f"({d.formula.nunique()} compounds)")

    k = resolve_lattice(d)
    origin = d.drop_duplicates("sample_id").set_index("sample_id").origin
    k["data_origin"] = k.sample_id.map(origin)

    # Low temperature is where boundary scattering dominates and the Umklapp 1/T law breaks; the
    # pool applies the same floor, so the doped rows must not be admitted on easier terms.
    before = len(k)
    k = k[pd.to_numeric(k["T"], errors="coerce") >= min_T]
    print(f"  T >= {min_T:.0f} K: {len(k)} rows (dropped {before - len(k)})")

    # ---- attach the parent's structure, never an invented one --------------------
    tr = pd.read_csv(TRAIN)
    tr["sys"] = tr.formula.map(element_system)
    pure = tr[tr.formula.map(full_heusler_kind) == "pure"]
    scols = ["struct_a_A", "struct_spacegroup", "struct_sg_symbol", "struct_prototype",
             "struct_site_ordering", "struct_species", "struct_frac_coords",
             "struct_grade", "lattice_convention"]
    have = [c for c in scols if c in pure.columns]
    parents = (pure[pure.struct_a_A.notna()]
               .sort_values("struct_grade" if "struct_grade" in pure.columns else "formula")
               .drop_duplicates("sys")
               .set_index("sys"))
    print(f"  candidate parent structures: {len(parents)} element systems")

    k["sys"] = k.formula.map(parent_system)      # match on the PARENT, not the doped set
    k["parent_formula"] = k.sys.map(parents.formula) if "formula" in parents.columns else None
    matched = k.sys.isin(parents.index)
    print(f"  doped rows with a real parent structure: {int(matched.sum())} of {len(k)}")
    lost = sorted(set(k[~matched].formula.unique()))
    print(f"  DROPPED for having no parent structure: {len(lost)} compounds")
    if lost:
        print(f"    {lost[:12]}")
    k = k[matched].copy()
    for c in have:
        k[c] = k.sys.map(parents[c])

    k["kappa_units"] = "W/m/K"
    k["method"] = "experimental, doped/substituted sample (lattice kappa as published)"
    k["method_tier"] = 0
    k["tier_name"] = "experimental"
    k["weight"] = DOPED_WEIGHT
    k["source"] = "doped_recovery"
    k["source_doi"] = k.doi
    k["source_url"] = ""
    k["source_location"] = ""
    k["struct_source"] = "parent_compound"
    k["struct_source_url"] = ""
    k["struct_source_detail"] = ("structure inherited from the stoichiometric parent; "
                                 "substitution shifts a by <1%")
    k["struct_id"] = ""
    k["struct_same_paper_as_kappa"] = False
    k["stoichiometry"] = "doped"
    k = k.rename(columns={"T": "temperature_K"})

    cols = [c for c in tr.columns if c in k.columns]
    out = k[cols + [c for c in ("kappa_basis", "data_origin") if c in k.columns]]
    out.to_csv(OUT, index=False)

    print(f"\n=== RECOVERED {len(out)} doped rows, {out.formula.nunique()} compounds, "
          f"{out.formula.map(element_system).nunique()} element systems ===")
    print(f"  T {out.temperature_K.min():.0f}-{out.temperature_K.max():.0f} K   "
          f"kappa {out.kappa_L.min():.2f}-{out.kappa_L.max():.2f} W/m/K")
    print(f"  experimental: {int((out.data_origin == 'experimental').sum())} rows")
    top = (out.assign(sys=out.formula.map(element_system))
           .groupby("sys").agg(rows=("formula", "size"), compounds=("formula", "nunique"))
           .sort_values("rows", ascending=False))
    print("\n  by element system:")
    print(top.head(15).to_string())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-T", type=float, default=100.0)
    raise SystemExit(main(**vars(ap.parse_args())))
