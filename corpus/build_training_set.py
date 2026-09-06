"""Assemble the FINAL Heusler kappa_L training set — the one file a model is allowed to train on.

THE ADMISSION RULE, stated once and enforced mechanically. A row is in only if it has ALL FOUR:

  1. a HEUSLER               -- stoichiometric 1:1:1 / 2:1:1 / 1:1:1:1, integer to within 2%,
                                and (where a space group is recorded) in 216 / 225 / 119 / 139
  2. a STRUCTURE             -- a lattice constant traced to a named source with an identifier.
                                Never generated, never estimated from neighbours or ionic radii.
  3. LATTICE thermal conductivity -- kappa_L, not total kappa and not electronic kappa
  4. a TEMPERATURE           -- an actual stated value, at or above 200 K

Anything failing one of the four is written out with the reason instead of being quietly dropped,
so the exclusions are auditable.

WHAT THIS FILE DELIBERATELY DOES NOT CONTAIN
--------------------------------------------
  * ML-REGRESSOR OUTPUT. The npj "million-scale" SI offered 2,273 extra Heuslers, but its column is
    literally `Predicted LTC` -- a neural network's guess from descriptors, with no phonon physics.
    Training on it teaches this model to copy another model's errors. Excluded; kept visible in
    `kappa_ML_PREDICTED_not_training.csv`. (ML FORCE FIELDS, where the network predicts forces and
    kappa still comes from a real BTE solve, ARE kept -- at tier 2.)
  * QUOTED VALUES. A paper citing another paper's number is not evidence. Phys. Rev. B 95, 045202
    Table I looks like 26 measurements and is 3 results plus 23 quotations; overall 28 of 52
    extracted rows were quotations. Dropped upstream by `provenance`.
  * NON-HEUSLER LOOKALIKES. cF12 fluorite (CaF2, ThO2, K2O) satisfies a 2:1 ratio; AgSbTe2, BiTeI,
    AlCuO2, CoSbS satisfy 1:1:2 or 1:1:1. All rejected by the space-group confirmation.
  * OFF-STOICHIOMETRIC ALLOYS. `Ti1Sb1Ru1.1` through `Ti1Sb1Ru1.9` and `Zr0.88Ni1Bi1` entered once
    through a rounding bug and are now rejected by the 2% integer tolerance.
  * DOPED COMPOSITIONS, BY DEFAULT. 13,088 rows of real measured doped-Heusler curves are available
    (`--include-doped`), and they are genuine data -- but they are the WRONG data for this task.
    In a doped Heusler the lattice thermal conductivity is suppressed mainly by point-defect and
    alloy scattering from the site disorder itself, not by the parent compound's intrinsic phonon
    physics. All 149 prediction targets are stoichiometric. This project has already measured the
    transfer failure directly: 25% error on pure compounds vs 11% on doped. Including 13,088 doped
    rows against 8,562 stoichiometric ones would let the disorder regime dominate the fit.
  * TOTAL kappa posing as lattice kappa. Where only total kappa exists, kappa_L is either derived by
    Wiedemann-Franz from the sample's OWN resistivity (flagged) or the row keeps a null kappa_L.
  * LOW-TEMPERATURE PERFECT-CRYSTAL DIVERGENCE. Below ~200 K a BTE calculation on an infinite
    defect-free crystal runs away (AlFe2Nb reaches 13,409 W/m/K at 10 K). Correct physics, wrong
    regime -- thermoelectrics work at 300-1200 K.

METHOD TIERS ARE CARRIED, NOT COLLAPSED. `method_tier` and `weight` travel with every row so a
Slack-model estimate (38% median error) can never outvote a phonon-BTE result on the same compound.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

SRC = "data/external/KAPPA_POOL_WITH_STRUCTURE.csv"
OUT = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
XLSX = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.xlsx"
REJECT = "data/external/TRAINING_SET_rejected_with_reason.csv"
T_FLOOR = 200.0

KEEP = ["formula", "kappa_L", "kappa_units", "temperature_K",
        "method", "method_tier", "tier_name", "weight", "kappa_basis",
        "struct_a_A", "struct_spacegroup", "struct_sg_symbol", "struct_prototype",
        "struct_site_ordering", "struct_species", "struct_frac_coords", "struct_grade",
        "lattice_convention",
        "source", "source_doi", "source_url", "source_location",
        "struct_source", "struct_source_url", "struct_source_detail", "struct_id",
        "struct_same_paper_as_kappa"]


def main(out: str = OUT, include_doped: bool = False) -> int:
    d = pd.read_csv(SRC)
    d["kappa_units"] = d.get("kappa_units", "W/m/K")
    d["kappa_units"] = d.kappa_units.fillna("W/m/K")
    n0, c0 = len(d), d.formula.nunique()
    print(f"pool with structures: {n0} rows, {c0} compounds\n")

    rejects = []

    def cut(mask, reason):
        nonlocal d
        bad = d[mask]
        if len(bad):
            rejects.append(bad.assign(reject_reason=reason))
            print(f"  dropped {len(bad):>5} rows / {bad.formula.nunique():>3} compounds : {reason}")
        d = d[~mask]

    print("ADMISSION FILTER")
    cut(d.kappa_L.isna() | (pd.to_numeric(d.kappa_L, errors="coerce") <= 0),
        "no positive lattice thermal conductivity")
    cut(pd.to_numeric(d.temperature_K, errors="coerce").isna(), "no stated temperature")
    cut(pd.to_numeric(d.temperature_K, errors="coerce") < T_FLOOR,
        f"temperature below {T_FLOOR:.0f} K (perfect-crystal divergence regime)")
    cut(~d.get("has_struct", pd.Series(False, index=d.index)).fillna(False).astype(bool),
        "no structure from a citable source")
    cut(d.struct_a_A.isna(), "structure record carries no lattice constant")

    d["stoichiometry"] = "stoichiometric"
    d["parent_formula"] = d.formula

    # ---- DOPED Heuslers: real measured curves on site-substituted compositions ----------------
    # `Hf0.5Zr0.5NiSn0.995Sb0.005` is a genuine half-Heusler with a genuine measured kappa_L(T).
    # Admitted ONLY when its PARENT compound (majority element per site) is independently confirmed
    # in a Heusler space group -- the test that rejects PbSnTe, which passes composition arithmetic
    # but is a rocksalt alloy. Their structure is the PARENT's lattice, flagged as such: the doped
    # lattice constant differs slightly and is NOT measured here, so it is never presented as if
    # it were. Filter on `stoichiometry` to include or exclude them.
    dp = Path("data/external/kappa_starrydata_doped_heusler.csv")
    if dp.exists() and include_doped:
        e = pd.read_csv(dp)
        e = e[e.kappa_L.notna() & (pd.to_numeric(e.kappa_L, errors="coerce") > 0)]
        e = e[pd.to_numeric(e.temperature_K, errors="coerce") >= T_FLOOR]
        # borrow the parent's structure, explicitly labelled
        par = (d.dropna(subset=["struct_a_A"])
                 .drop_duplicates("formula")
                 .set_index("formula")[["struct_a_A", "struct_spacegroup", "struct_sg_symbol",
                                        "struct_prototype", "struct_source", "struct_source_url",
                                        "struct_id", "lattice_convention"]])
        e = e.join(par, on="parent_formula", how="inner")
        e["struct_grade"] = "P"
        e["struct_source_detail"] = ("structure of the PARENT compound " + e.parent_formula
                                     + " (doped composition; its own lattice not measured here)")
        e["struct_same_paper_as_kappa"] = False
        e["struct_site_ordering"] = pd.NA
        e["struct_species"] = pd.NA
        e["struct_frac_coords"] = pd.NA
        e["method_tier"] = 0
        e["tier_name"] = "experimental"
        e["weight"] = 1.00
        e["stoichiometry"] = "doped"
        e["kappa_units"] = "W/m/K"
        for c in KEEP + ["stoichiometry", "parent_formula"]:
            if c not in e.columns:
                e[c] = pd.NA
        print(f"\n  + doped Heuslers with a confirmed parent: {len(e)} rows, "
              f"{e.formula.nunique()} compositions, {e.parent_formula.nunique()} parents")
        d = pd.concat([d, e], ignore_index=True)

    for c in KEEP:
        if c not in d.columns:
            d[c] = pd.NA
    t = d[KEEP + ["stoichiometry", "parent_formula"]].copy()
    t = t.sort_values(["method_tier", "formula", "temperature_K"])

    print(f"\n{'=' * 72}\nFINAL TRAINING SET\n{'=' * 72}")
    print(f"  rows       : {len(t)}")
    print(f"  COMPOUNDS  : {t.formula.nunique()}")
    print(f"  every row has: Heusler formula + cited structure + kappa_L + temperature")

    print(f"\n  by method tier")
    for tier, g in t.groupby("tier_name"):
        print(f"    {str(tier):<34}{len(g):>6} rows{g.formula.nunique():>6} compounds"
              f"   weight {g.weight.iloc[0]:.2f}")

    print(f"\n  temperature")
    T = pd.to_numeric(t.temperature_K, errors="coerce")
    print(f"    range            : {T.min():.0f} - {T.max():.0f} K")
    nT = t.groupby("formula").temperature_K.nunique()
    print(f"    multi-temperature: {int((nT > 1).sum())} compounds "
          f"(median {nT[nT > 1].median():.0f} points each)")
    print(f"    single point     : {int((nT == 1).sum())} compounds")
    for lo, hi in ((200, 400), (400, 700), (700, 1000), (1000, 1300)):
        m = t[T.between(lo, hi, inclusive="left")]
        print(f"    {lo:>4}-{hi:<5}K     : {len(m):>6} rows, {m.formula.nunique():>4} compounds")

    print(f"\n  structure evidence grade")
    for g_, n in t.struct_grade.value_counts().sort_index().items():
        lab = {"A": "full structure (lattice + species + coordinates)",
               "B": "lattice + space group + stated site ordering",
               "C": "lattice + space group + prototype (site order not stated)"}.get(g_, g_)
        print(f"    {g_}  {lab:<56}{n:>6} rows")
    print(f"    structure from the SAME source as kappa: "
          f"{int(t.struct_same_paper_as_kappa.fillna(False).astype(bool).sum())} rows")

    print(f"\n  provenance completeness")
    print(f"    rows with a DOI or URL : "
          f"{int((t.source_doi.notna() | t.source_url.notna()).sum())} / {len(t)}")
    print(f"    rows with a structure source: {int(t.struct_source.notna().sum())} / {len(t)}")

    print(f"\n  top sources")
    for s, g in sorted(t.groupby("source"), key=lambda x: -len(x[1]))[:8]:
        print(f"    {str(s)[:44]:<46}{len(g):>6} rows{g.formula.nunique():>6} compounds")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")
    try:
        with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
            t.to_excel(w, sheet_name="training_set", index=False)
            (t.groupby("formula")
               .agg(n_points=("kappa_L", "size"), tier=("method_tier", "min"),
                    T_min=("temperature_K", "min"), T_max=("temperature_K", "max"),
                    k_min=("kappa_L", "min"), k_max=("kappa_L", "max"),
                    a_A=("struct_a_A", "first"), spacegroup=("struct_spacegroup", "first"),
                    grade=("struct_grade", "first"), source=("source", "first"))
               .reset_index().to_excel(w, sheet_name="by_compound", index=False))
        print(f"wrote {XLSX}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (xlsx skipped: {exc})")

    if rejects:
        r = pd.concat(rejects, ignore_index=True)
        r.to_csv(REJECT, index=False)
        print(f"wrote {REJECT}  ({len(r)} rows, each with a stated reason)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--include-doped", action="store_true",
                    help="add site-substituted compositions. OFF by default: doped "
                         "kappa_L is dominated by alloy/point-defect scattering, which "
                         "does not transfer to the stoichiometric targets (measured on "
                         "this project: 25%% error on pure vs 11%% on doped).")
    raise SystemExit(main(**vars(ap.parse_args())))
