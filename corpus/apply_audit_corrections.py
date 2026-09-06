"""Apply the two data corrections the 2026-09-03 audit established, in one pass.

Both changes are to the tier-0 (measured) reference data, which is what every accuracy figure in
the project is scored against. They are applied together so the whole chain is regenerated once
rather than twice.

CORRECTION 1 -- REMOVE ROWS WHOSE WIEDEMANN-FRANZ SUBTRACTION WAS A NO-OP.

`harvest_starrydata.py` computes kappa_L = kappa_total - L*T/rho from the sample's own resistivity
curve. It never checked that rho was physical. A dense thermoelectric intermetallic sits between
~1e-8 ohm.m (a good metal) and ~1e-2 ohm.m (a poor semiconductor); 136 rows carry rho of 10-34
ohm.m, about 1e6 too large. The failure is silent, not loud: kappa_e collapses to ~1e-6 W/m/K, the
subtraction removes nothing, and the row is emitted as `wf_subtracted` while actually carrying the
TOTAL conductivity under a lattice label.

The consequence is visible in the physics. Such a compound appears to have a kappa that RISES with
temperature, because the electronic part that was never removed grows with T. `TiCoSn` -- 47 of 47
rows affected -- reached this project's verified-correct list on exactly that reference, and its
apparent +0.111 log-log slope is the electronic contribution, not a phonon anomaly.

    TiCoSn   47 of  47 rows   rho 10.5-13.3     the entire reference is a total
    HfNiSn   63 rows          NbCoSn 23 of 146  ZrCoSb 3

The gate is now in `harvest_starrydata.py` for future harvests; this removes the rows already in
the training set, since re-running the harvester would need the multi-GB Starrydata download.

CORRECTION 2 -- MERGE THE LuNiSb CORROBORATION THAT WAS SITTING UNMERGED.

`LuNiSb` was the best-scoring compound on the evidence list (9.6% error) and rested on a single
paper, which is the weakest provenance a headline result can have. An independent measurement was
already in `heusler.sqlite` and had never been merged:

    10.1021/acsaem.6c00361 -- "Defect-Calculation-Guided Performance Optimization of n-Type LuNiSb
    Half-Heusler Thermoelectrics", ACS Applied Energy Materials (2026)
    bulk polycrystalline, F-43m (No. 216), a = 6.2263 A, mainly single phase

Only the four rows the paper itself labels "derived from kappa - kappa_e" (Fig. 5e) are taken. The
Fig. 5d series is the TOTAL conductivity and merging it would repeat correction 1's mistake in a
new place.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
STARRY = "data/external/kappa_starrydata_heusler.csv"
DB = "data/heusler.sqlite"
BACKUP = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.pre_audit.csv"
RHO_LO, RHO_HI = 1e-8, 1e-2
LU_DOI = "10.1021/acsaem.6c00361"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    tr = pd.read_csv(TRAIN)
    shutil.copy(TRAIN, BACKUP)
    print(f"backed up {len(tr)} rows -> {BACKUP}\n")
    tr["red"] = tr.formula.map(red)
    tr["_k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["_T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    t0 = pd.to_numeric(tr.method_tier, errors="coerce") == 0

    # ---- correction 1 ---------------------------------------------------------
    sd = pd.read_csv(STARRY, low_memory=False)
    sd["red"] = sd.formula.map(red)
    sd["rho"] = pd.to_numeric(sd.resistivity_ohm_m, errors="coerce")
    bad = sd[sd.rho.notna() & ~sd.rho.between(RHO_LO, RHO_HI)].copy()
    print(f"correction 1 -- unphysical resistivity in the Starrydata export: {len(bad)} rows, "
          f"{bad.red.nunique()} compounds")
    for r_, g in bad.groupby("red"):
        print(f"    {r_:<9}{len(g):>4} rows   rho {g.rho.min():.3g}-{g.rho.max():.3g} ohm.m")

    key = set(zip(bad.red, np.round(pd.to_numeric(bad.kappa_L, errors="coerce"), 4),
                  np.round(pd.to_numeric(bad.temperature_K, errors="coerce"), 2)))
    drop = t0 & pd.Series(
        [(r_, round(k, 4) if pd.notna(k) else None, round(T, 2) if pd.notna(T) else None) in key
         for r_, k, T in zip(tr.red, tr._k, tr._T)], index=tr.index)
    print(f"  matched in the training set: {int(drop.sum())} rows")
    for r_, n in tr[drop].groupby("red").size().items():
        before = tr[t0 & (tr.red == r_)]
        after = tr[t0 & (tr.red == r_) & ~drop]
        b = float(before._k.median()) if len(before) else np.nan
        a = float(after._k.median()) if len(after) else np.nan
        print(f"    {r_:<9} drop {n:>3} of {len(before):>3}   median kappa "
              f"{b:.2f} -> {'(none left)' if not np.isfinite(a) else f'{a:.2f}'}")
    tr = tr[~drop].copy()

    # ---- correction 2 ---------------------------------------------------------
    c = sqlite3.connect(DB)
    q = pd.read_sql(
        "select m.canonical_value v, m.temperature_k T, m.location_in_paper lip "
        "from measurements m join samples s on m.sample_id = s.sample_id "
        "where s.reduced_formula = 'LuNiSb' and m.doi = ? "
        "and m.property_name like 'thermal_conductivity%' "
        "and m.data_origin = 'experimental'", c, params=(LU_DOI,))
    lat = q[q.lip.astype(str).str.contains("kappa-kappa_e|κ-κe|derived", case=False, na=False)]
    print(f"\ncorrection 2 -- LuNiSb corroboration from {LU_DOI}")
    print(f"  rows in DB {len(q)}   of which explicitly lattice-derived: {len(lat)}")
    proto = tr[(tr.red == "LuNiSb") & (pd.to_numeric(tr.method_tier, errors="coerce") == 0)]
    if not len(proto) or not len(lat):
        print("  cannot merge -- no existing LuNiSb row to copy structure metadata from")
    else:
        base = proto.iloc[0]
        new = []
        for r_ in lat.itertuples():
            row = base.copy()
            row["kappa_L"] = round(float(r_.v), 4)
            row["temperature_K"] = float(r_.T)
            row["source_doi"] = LU_DOI
            row["source_url"] = f"https://doi.org/{LU_DOI}"
            row["source"] = ("ACS Appl. Energy Mater. 2026, Defect-Calculation-Guided "
                             "Performance Optimization of n-Type LuNiSb")
            row["source_location"] = str(r_.lip)
            row["method"] = "experimental, kappa_total - kappa_e (as reported by the authors)"
            row["kappa_basis"] = "wf_subtracted"
            new.append(row)
        add = pd.DataFrame(new)
        print(f"  merging {len(add)} rows: "
              f"{[(float(a), float(b)) for a, b in zip(add.temperature_K, add.kappa_L)]}")
        tr = pd.concat([tr, add], ignore_index=True)
        n_doi = tr[(tr.red == "LuNiSb")
                   & (pd.to_numeric(tr.method_tier, errors="coerce") == 0)].source_doi.nunique()
        print(f"  LuNiSb independent DOIs: 1 -> {n_doi}")

    out = tr.drop(columns=[c_ for c_ in ("red", "_k", "_T") if c_ in tr.columns])
    out.to_csv(TRAIN, index=False)
    n0 = int((pd.to_numeric(out.method_tier, errors="coerce") == 0).sum())
    print(f"\ntraining set: {len(pd.read_csv(BACKUP))} -> {len(out)} rows   tier-0 now {n0}")
    print(f"wrote {TRAIN}")
    print("\nNOW REGENERATE: run_target_blind_test -> extend_blind_test -> validate_intervals ->")
    print("compute_null_baselines -> nested_validation -> compute_nested_across_seeds -> publishers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
