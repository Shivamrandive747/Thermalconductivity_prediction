"""Step 2a — harvest AFLOW AGL as the COVERAGE layer of the theoretical kappa pool.

WHAT AFLOW IS, AND WHY IT IS TAGGED RATHER THAN TRUSTED
-------------------------------------------------------
AGL is a quasi-harmonic Debye-Gruneisen model, not a Boltzmann-transport calculation. Measured
against our own 44 experimental Heuslers on the 20 that overlap:

    median absolute error   38%
    median ratio            1.23   (AFLOW reads ~23% HIGH)
    Spearman rho            +0.463 (p = 0.040, n = 20)
    spread                  3% error (VGaFe2) to 259% (NbAlRu2); ratios 0.20x - 3.59x

So: real signal, but far too soft to be a primary training target -- our existing model already
reaches 13.6% on interpolation, and training on 38%-error labels would cap it below where it is.

Its value is COVERAGE. 64 of the 149 prediction targets contain Cu/Ag/Zn/In, and we hold at most
one or two measured compounds for each of those elements. On that chemistry the model currently
extrapolates blind; an approximate label at least teaches it how those elements behave.

The systematic direction is also physically sensible and therefore learnable: AGL computes a perfect
infinite crystal, while a measurement is on a real sample with grain boundaries, point defects and
porosity, all of which scatter phonons and lower kappa. TiNiSn is the clearest case -- AGL 8.94 vs
measured 4.70 -- because TiNiSn is routinely nanostructured specifically to suppress kappa. That is
not an AGL error; it is the perfect-crystal-to-real-sample offset that delta-learning exists to
absorb.

Every row is therefore stamped `method_class = AGL_Debye` and `weight = 0.25`, and must never be
pooled with phonon-BTE values or used for validation.

WHAT ELSE COMES FREE
--------------------
`agl_debye` and `agl_gruneisen` are the Slack-model backbone of kappa_L, and `ael_*` give elastic
moduli. ML_PLAN.md called physics intermediates of exactly this kind "the single highest-leverage
feature addition". They arrive with the same request as kappa and are worth having even where the
kappa label itself is weak.

A HEUSLER IS NOT JUST A SPACE GROUP. The first sg-225 ternary AFLOW returned was Na6O9S2 -- cubic,
three species, and not remotely a Heusler. Stoichiometry is checked as well: 1:1:1 at F-43m (216) is
half-Heusler C1b, 2:1:1 at Fm-3m (225) is full-Heusler L2_1, and 2:1:1 at F-43m is inverse XA.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from math import gcd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
import requests
from pymatgen.core import Composition

BASE = "https://aflow.org/API/aflux/?"
OUT = "data/external/aflow_agl_heusler.csv"
PAGE = 1000

# AFLUX drops any row that lacks a requested field, so an optional field silently
# shrinks the whole result set. Bisecting the sg-225 query field by field:
#     ...ael_shear_modulus_vrh -> 279 rows
#     + enthalpy_formation_atom -> 40      <-- collapses it
#     + Egap                    -> 40
# Both are sparse and neither is needed for kappa (formation energy for the targets
# already comes from DXMag). They are fetched separately as OPTIONAL, never required.
FIELDS = ["compound", "auid", "aurl", "spacegroup_relax", "Pearson_symbol_relax",
          "geometry", "species", "composition", "positions_fractional", "natoms",
          "volume_atom", "density", "agl_thermal_conductivity_300K", "agl_debye",
          "agl_gruneisen", "agl_heat_capacity_Cp_300K", "agl_bulk_modulus_static_300K",
          "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh"]
OPTIONAL_FIELDS = ["enthalpy_formation_atom", "Egap"]

# (space group, sorted stoichiometry ratio) -> Heusler class
HEUSLER = {(216, (1, 1, 1)): "half_C1b",
           (225, (1, 1, 2)): "full_L21",
           (216, (1, 1, 2)): "inverse_XA"}


def ratio(comp) -> tuple[int, ...] | None:
    """[6, 9, 2] -> (2, 3, 6) reduced and sorted; None if unusable."""
    try:
        v = [int(x) for x in comp]
    except Exception:  # noqa: BLE001
        return None
    if not v or any(x <= 0 for x in v):
        return None
    g = 0
    for x in v:
        g = gcd(g, x)
    return tuple(sorted(x // g for x in v)) if g else None


def classify(sg, comp) -> str | None:
    r = ratio(comp)
    if r is None:
        return None
    try:
        return HEUSLER.get((int(sg), r))
    except Exception:  # noqa: BLE001
        return None


def fetch(session, filt: str, page: int) -> list:
    q = filt + "," + ",".join(f"{f}(*)" for f in FIELDS) + f",$paging({page},{PAGE})"
    for attempt in range(4):
        try:
            r = session.get(BASE + q, timeout=120)
            if r.status_code == 200:
                j = r.json()
                return j if isinstance(j, list) else []
            if r.status_code in (429, 503):
                time.sleep(5 + 5 * attempt)
                continue
            return []
        except Exception:  # noqa: BLE001
            time.sleep(3 + 3 * attempt)
    return []


def harvest(session, filt: str, label: str) -> list[dict]:
    out, page = [], 1
    while True:
        batch = fetch(session, filt, page)
        if not batch:
            break
        out.extend(batch)
        print(f"    {label}: page {page} -> {len(batch)} (total {len(out)})", flush=True)
        if len(batch) < PAGE:
            break
        page += 1
        time.sleep(0.5)
        if page > 20:
            print(f"    {label}: page cap reached", flush=True)
            break
    return out


def main(gap_elements: str = "Cu,Ag,Zn,In,Sc,Pd") -> int:
    s = requests.Session()
    s.headers["User-Agent"] = "heusler-kappa-harvest/1.0"
    raw: dict[str, dict] = {}

    # 1. every ternary in a Heusler space group that carries an AGL kappa
    for sg in (216, 225):
        rows = harvest(s, f"nspecies(3),spacegroup_relax({sg}),"
                          "agl_thermal_conductivity_300K(*)", f"sg{sg}")
        for r in rows:
            raw[r.get("auid", r.get("compound", ""))] = r

    # 2. the gap chemistry specifically, at ANY space group -- a Cu/Ag/Zn/In compound
    #    with a computed kappa is worth having even if it is not a Heusler prototype,
    #    because it still teaches the model how those elements conduct heat.
    for el in [e.strip() for e in gap_elements.split(",") if e.strip()]:
        rows = harvest(s, f"species({el}),nspecies(3),agl_thermal_conductivity_300K(*)", el)
        for r in rows:
            raw.setdefault(r.get("auid", r.get("compound", "")), r)

    print(f"\n  {len(raw)} unique AFLOW entries retrieved")

    recs = []
    for r in raw.values():
        comp = r.get("composition")
        sg = r.get("spacegroup_relax")
        cls = classify(sg, comp)
        geo = r.get("geometry") or []
        sp = r.get("species") or []
        try:
            formula = Composition(str(r.get("compound", ""))).reduced_formula
        except Exception:  # noqa: BLE001
            formula = str(r.get("compound", ""))
        recs.append(dict(
            source="AFLOW_AGL", method_class="AGL_Debye", weight=0.25,
            auid=r.get("auid"), aurl=r.get("aurl"),
            compound=r.get("compound"), reduced_formula=formula,
            elements=",".join(str(x) for x in sp),
            n_elements=len(sp), composition=",".join(str(x) for x in (comp or [])),
            heusler_class=cls, is_heusler=cls is not None,
            spacegroup=sg, pearson=r.get("Pearson_symbol_relax"),
            a=geo[0] if len(geo) > 5 else None, b=geo[1] if len(geo) > 5 else None,
            c=geo[2] if len(geo) > 5 else None,
            alpha=geo[3] if len(geo) > 5 else None, beta=geo[4] if len(geo) > 5 else None,
            gamma=geo[5] if len(geo) > 5 else None,
            natoms=r.get("natoms"), volume_atom=r.get("volume_atom"),
            density=r.get("density"),
            kappa_300K=r.get("agl_thermal_conductivity_300K"),
            debye_K=r.get("agl_debye"), gruneisen=r.get("agl_gruneisen"),
            Cp_300K=r.get("agl_heat_capacity_Cp_300K"),
            bulk_static=r.get("agl_bulk_modulus_static_300K"),
            bulk_vrh=r.get("ael_bulk_modulus_vrh"),
            shear_vrh=r.get("ael_shear_modulus_vrh"),
            Ef_atom=r.get("enthalpy_formation_atom"), Egap=r.get("Egap"),  # optional, often null
            positions=json.dumps(r.get("positions_fractional")) if r.get("positions_fractional") else None))

    d = pd.DataFrame(recs)
    d["kappa_300K"] = pd.to_numeric(d.kappa_300K, errors="coerce")
    d = d[d.kappa_300K.notna() & (d.kappa_300K > 0)]

    print(f"\n=== HARVEST ===")
    print(f"  rows with a usable kappa : {len(d)}   distinct formulae: {d.reduced_formula.nunique()}")
    print(f"  genuine Heusler prototypes: {int(d.is_heusler.sum())}")
    print(d[d.is_heusler].heusler_class.value_counts().to_string())
    print(f"\n  kappa 300 K: min {d.kappa_300K.min():.2f}  median {d.kappa_300K.median():.2f}  "
          f"max {d.kappa_300K.max():.2f} W/m/K")
    print(f"  with Debye temperature   : {d.debye_K.notna().sum()}")
    print(f"  with Gruneisen parameter : {d.gruneisen.notna().sum()}")
    print(f"  with elastic moduli      : {d.bulk_vrh.notna().sum()}")

    print(f"\n  COVERAGE of the gap elements (formulae):")
    for el in [e.strip() for e in gap_elements.split(",")]:
        m = d[d.elements.str.contains(rf"\b{el}\b", regex=True, na=False)]
        print(f"    {el:<3} {m.reduced_formula.nunique():>5} formulae   "
              f"({m[m.is_heusler].reduced_formula.nunique()} in Heusler prototypes)")

    from pathlib import Path
    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-elements", default="Cu,Ag,Zn,In,Sc,Pd")
    raise SystemExit(main(**vars(ap.parse_args())))
