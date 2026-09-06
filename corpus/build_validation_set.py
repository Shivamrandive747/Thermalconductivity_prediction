"""Step 1a — lock the validation set to real Heuslers, and clean it.

USER DECISION: validate on extracted experimental Heuslers and nothing else. No perovskites, no
elements, no other thermoelectric families, no double half-Heuslers, no high-entropy variants.

The filter is mechanical, not a hand-picked list:
    stoichiometric  AND  exactly 3 distinct elements  AND  experimental kappa
    AND a linked structure whose prototype is C1b (half) / L21 (full) / XA (inverse)

Applied to the 86 pure experimental formulae that carry kappa, this keeps 44 and drops 42. What it
drops and why is written out in full, because a validation set you cannot audit is not a validation
set. Among the discards are `Gd` (a pure element), `LaCoO3` (a perovskite), `TePb` (PbTe),
`Zn4Sb3`, `Al5Co2` -- none of which are Heuslers -- and `VFeCoSb`, `Nb2FeCoSnSb`, `Ti2CoNiSnSb`,
`ZrTiCoNiSnSb` and friends, which are real materials but double half-Heuslers and high-entropy
variants rather than the ternary compounds we predict.

ALSO FIXED HERE: four compounds report a minimum kappa of exactly 0.0 W/m/K -- `ZrCoSb`, `ZrSbIr`,
`TaAlRu2`, `NbAlRu2`. Thermal conductivity cannot be zero; these are figure-extraction artefacts
(an axis misread, or a digitiser picking up the plot origin). The zero rows are dropped and the rest
of each temperature curve is kept, so no compound is lost over a single bad point.
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

from pipeline.s00d_external import is_stoichiometric

DB = "file:data/heusler.sqlite?mode=ro"
OUT = "data/Target_Materials/Validation_Heuslers.xlsx"
PROTO_OK = {"C1b": "half", "L21": "full", "XA": "inverse"}
# kappa below this is not a measurement. Real Heusler kappa runs ~0.2-100 W/m/K.
_KAPPA_FLOOR = 0.05


def n_elements(f: str) -> int:
    try:
        return len(Composition(f).elements)
    except Exception:  # noqa: BLE001
        return 0


def _resolve_lattice_kappa(d: pd.DataFrame) -> pd.DataFrame:
    """Reduce three different measured properties to the ONE we actually predict.

    THE BUG THIS FIXES. The query above asks for `property_name LIKE 'thermal_conductivity%'`,
    and that wildcard matches three physically distinct quantities:

        thermal_conductivity_lattice      heat carried by phonons   <- what the model predicts
        thermal_conductivity_total        phonons + electrons
        thermal_conductivity_electronic   heat carried by electrons <- a different mechanism

    Measured consequence: only 28% of the validation rows were the right target, and the model was
    being scored against electron-carried heat on 66 of them (median error 146%, bias 2.46x). No
    amount of better physics can survive being graded against the wrong quantity.

    RECOVER, DO NOT JUST DISCARD. Where the same sample at the same temperature and DOI reports both
    total and electronic, the lattice part is their difference -- the subtraction the paper's own
    authors performed. That converts ~54 otherwise-lost rows into usable ones.

    Every surviving row is stamped with `kappa_basis`, and the two bases are never pooled into one
    metric downstream.
    """
    d = d.copy()
    key = ["sample_id", "formula", "doi", "T"]
    piv = d.pivot_table(index=key, columns="property", values="kappa", aggfunc="median")
    lat_c, tot_c, ele_c = ("thermal_conductivity_lattice", "thermal_conductivity_total",
                           "thermal_conductivity_electronic")
    for c in (lat_c, tot_c, ele_c):
        if c not in piv.columns:
            piv[c] = float("nan")

    lattice = piv[lat_c]
    derived = piv[tot_c] - piv[ele_c]
    basis = pd.Series("none", index=piv.index)
    value = pd.Series(float("nan"), index=piv.index)

    m_rep = lattice.notna() & (lattice > 0)
    value[m_rep] = lattice[m_rep]
    basis[m_rep] = "reported_lattice"

    m_sub = (~m_rep) & derived.notna() & (derived > 0)
    value[m_sub] = derived[m_sub]
    basis[m_sub] = "total_minus_electronic"

    out = piv.assign(kappa=value, kappa_basis=basis).reset_index()
    out = out[out.kappa.notna()]
    print("\n  RESOLVING to lattice kappa only:")
    print(f"    reported lattice        : {int((out.kappa_basis == 'reported_lattice').sum())}")
    print(f"    recovered total-electronic: {int((out.kappa_basis == 'total_minus_electronic').sum())}")
    print(f"    dropped (neither available): "
          f"{int(len(piv) - len(out))} sample/temperature combinations")

    # carry back the descriptive columns the rest of the script expects
    meta = (d.sort_values("property")
             .drop_duplicates(subset=key)[key + ["heusler_type", "tier", "prototype",
                                                 "spacegroup", "physics_flag"]])
    out = out.merge(meta, on=key, how="left")
    out["property"] = "thermal_conductivity_lattice"
    return out[key + ["heusler_type", "tier", "prototype", "spacegroup", "physics_flag",
                      "property", "kappa", "kappa_basis"]]


def main(floor: float = _KAPPA_FLOOR) -> int:
    con = sqlite3.connect(DB, uri=True)
    q = """
        select s.sample_id, s.reduced_formula, s.heusler_type, s.structure_tier,
               ss.prototype, ss.spacegroup, m.doi, m.property_name, m.canonical_value,
               m.temperature_k, m.physics_flag
        from samples s
        join measurements m on m.sample_id = s.sample_id
        left join sample_structures ss on ss.sample_id = s.sample_id
        where m.property_name like 'thermal_conductivity%'
          and m.canonical_value is not null
          and m.data_origin = 'experimental'
    """
    d = pd.DataFrame(con.execute(q).fetchall(), columns=[
        "sample_id", "formula", "heusler_type", "tier", "prototype", "spacegroup",
        "doi", "property", "kappa", "T", "physics_flag"])
    print(f"experimental kappa rows: {len(d)}  ({d.formula.nunique()} formulae)")
    print(d.property.value_counts().to_string())

    d = _resolve_lattice_kappa(d)

    d["pure"] = [is_stoichiometric(f) for f in d.formula]
    d["nel"] = d.formula.map(n_elements)
    pure = d[d.pure]
    print(f"  stoichiometric/pure    : {len(pure)} rows ({pure.formula.nunique()} formulae)")

    # ---- the mechanical filter ----------------------------------------------
    keep = pure[(pure.nel == 3) & (pure.prototype.isin(PROTO_OK))].copy()
    dropped = sorted(set(pure.formula) - set(keep.formula))
    print(f"\n  KEPT   : {keep.formula.nunique()} Heuslers ({len(keep)} rows)")
    print(f"  DROPPED: {len(dropped)} formulae")

    # say WHY each was dropped
    reasons = []
    for f in dropped:
        g = pure[pure.formula == f]
        nel = int(g.nel.iloc[0])
        proto = g.prototype.dropna()
        if nel != 3:
            why = f"not ternary ({nel} distinct elements)"
        elif proto.empty:
            why = "no linked structure"
        else:
            why = f"prototype {sorted(set(proto))} is not C1b/L21/XA"
        reasons.append({"formula": f, "reason": why, "n_rows": len(g),
                        "heusler_type": g.heusler_type.iloc[0]})
    drop_df = pd.DataFrame(reasons)
    for why, grp in drop_df.groupby("reason"):
        print(f"    {why}: {len(grp)}")
        print(f"      {sorted(grp.formula)[:10]}")

    # ---- the unphysical-zero fix --------------------------------------------
    zero = keep[keep.kappa < floor]
    if len(zero):
        print(f"\n  UNPHYSICAL kappa < {floor} W/m/K: {len(zero)} rows across "
              f"{zero.formula.nunique()} compounds -> DROPPED")
        for f, g in zero.groupby("formula"):
            rest = keep[(keep.formula == f) & (keep.kappa >= floor)]
            print(f"    {f:<10} {len(g)} bad row(s), {len(rest)} good rows kept "
                  f"(range {rest.kappa.min():.2f}-{rest.kappa.max():.2f})")
    keep = keep[keep.kappa >= floor].copy()

    # ---- summarise -----------------------------------------------------------
    keep["class"] = keep.prototype.map(PROTO_OK)
    keep["rare_earth"] = [bool({str(e) for e in Composition(f).elements}
                               & set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Sc Y".split()))
                          for f in keep.formula]
    per = (keep.groupby("formula")
           .agg(cls=("class", "first"), proto=("prototype", "first"),
                n=("kappa", "size"), dois=("doi", "nunique"),
                T_min=("T", "min"), T_max=("T", "max"),
                k_min=("kappa", "min"), k_max=("kappa", "max"),
                rare_earth=("rare_earth", "first"))
           .reset_index().sort_values("n", ascending=False))

    print(f"\n=== VALIDATION SET: {len(per)} Heuslers, {len(keep)} kappa measurements ===")
    print(f"  half (C1b) {int((per.cls=='half').sum())} | "
          f"full (L21) {int((per.cls=='full').sum())} | "
          f"inverse (XA) {int((per.cls=='inverse').sum())}")
    print(f"  rare-earth bearing: {int(per.rare_earth.sum())}  "
          f"(kept -- real Heuslers with real data; note targets are rare-earth screened)")
    print(f"  by property: {dict(keep.property.value_counts())}")
    print(f"  T {keep['T'].min():.0f}-{keep['T'].max():.0f} K | "
          f"kappa {keep.kappa.min():.2f}-{keep.kappa.max():.2f} W/m/K")
    print(f"\n  {'compound':<11}{'class':<6}{'papers':>7}{'pts':>5}   T range      kappa")
    for _, r in per.head(20).iterrows():
        print(f"  {r.formula:<11}{r.cls:<6}{r.dois:>7}{r.n:>5}   "
              f"{r.T_min:.0f}-{r.T_max:.0f} K".ljust(46) + f"{r.k_min:.1f}-{r.k_max:.1f}")

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        per.to_excel(w, sheet_name="Compounds", index=False)
        keep.to_excel(w, sheet_name="Measurements", index=False)
        drop_df.to_excel(w, sheet_name="Excluded_with_reason", index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=_KAPPA_FLOOR)
    raise SystemExit(main(**vars(ap.parse_args())))
