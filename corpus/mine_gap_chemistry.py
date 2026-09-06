"""Mine the chemistry the prediction targets need and the pool still lacks.

WHAT IS ACTUALLY MISSING. After merging every source, 906 Heuslers are in the pool and 19 of the 179
target element-pairs still have no support. The survivors cluster almost entirely on ZINC
(Al-Zn 7 targets, Ga-Zn 6, In-Zn 4) plus Ag-Cu (7), and on three thin elements: Zn (15 compounds),
Mo (9), Re (5).

SO THIS QUERIES THE HOLES, NOT MORE OF WHAT WE HAVE. Harvesting another 2,000 antimony rows would
raise the row count and close nothing.

WHAT A PROBE SHOWED BEFORE ANY OF THIS WAS BUILT -- the honest part:

    Ag-Cu              6 AFLOW entries, ALL in a Heusler space group   <- real gain
    Zn  sg216 / sg225  10 + 9 entries                                  <- real gain
    Re  ternary         3 in a Heusler space group
    Ru-Sc / Mo          1 each
    Al-Zn, Ga-Zn, In-Zn   7 / 3 / 9 entries, ZERO in a Heusler space group

That last line is a finding, not a failure: those pairs do not exist as computed Heuslers in AFLOW.
The script reports them as unfillable rather than substituting a non-Heusler compound to make the
coverage table look complete -- the user's constraint is Heuslers only, and a Cu-Zn compound in
space group 194 is not a Heusler.

TIER. Everything here is AGL quasi-harmonic Debye-Gruneisen at 300 K: tier 3, weight 0.25, 38%
median error against our measured Heuslers. It is COVERAGE for chemistry the model would otherwise
extrapolate on blind, never a primary training target.

PROVENANCE. Every row keeps the AFLOW auid, the resolvable aurl, and the literal AFLUX query string.
The lattice constant is AFLOW's `geometry[0]`, which is the PRIMITIVE fcc edge -- measured against
JARVIS at a median ratio of 1.4103 vs sqrt(2)=1.4142 -- so it is converted to the conventional cubic
edge and the conversion is recorded per row.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import time
import warnings
from math import gcd
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
import requests
from pymatgen.core import Composition

BASE = "https://aflow.org/API/aflux/?"
OUT = "data/external/kappa_aflow_gap.csv"
POOL = "data/external/KAPPA_POOL_MASTER.csv"
TARGETS = "data/Target_Materials/Target_Dossier_166.xlsx"
PAGE = 500
SQRT2 = 2.0 ** 0.5

FIELDS = ["compound", "auid", "aurl", "spacegroup_relax", "Pearson_symbol_relax", "geometry",
          "species", "composition", "positions_fractional", "natoms", "volume_atom", "density",
          "agl_thermal_conductivity_300K", "agl_debye", "agl_gruneisen",
          "agl_heat_capacity_Cp_300K", "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh"]

HEUSLER = {(216, (1, 1, 1)): "half_C1b", (225, (1, 1, 2)): "full_L21",
           (216, (1, 1, 2)): "inverse_XA", (225, (1, 1, 1, 1)): "quaternary_Y",
           (216, (1, 1, 1, 1)): "quaternary_Y", (139, (1, 1, 2)): "full_tetragonal",
           (119, (1, 1, 2)): "inverse_tetragonal", (119, (1, 1, 1)): "half_tetragonal"}


def els(f):
    return sorted(str(e) for e in Composition(f).elements)


def ratio(comp):
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


def classify(sg, comp):
    r = ratio(comp)
    try:
        return HEUSLER.get((int(sg), r)) if r else None
    except Exception:  # noqa: BLE001
        return None


def fetch(s, filt, page):
    q = filt + "," + ",".join(f"{f}(*)" for f in FIELDS) + f",$paging({page},{PAGE})"
    for attempt in range(4):
        try:
            r = s.get(BASE + q, timeout=120)
            if r.status_code == 200:
                j = r.json()
                return (j if isinstance(j, list) else []), BASE + q
            if r.status_code in (429, 503):
                time.sleep(4 + 4 * attempt)
                continue
            return [], BASE + q
        except Exception:  # noqa: BLE001
            time.sleep(3 + 2 * attempt)
    return [], BASE + q


def harvest(s, filt, label, sink, stats):
    page, got = 1, 0
    while True:
        batch, url = fetch(s, filt, page)
        if not batch:
            break
        for b in batch:
            b["_query"] = url
            sink.setdefault(b.get("auid"), b)
        got += len(batch)
        if len(batch) < PAGE or page > 8:
            break
        page += 1
        time.sleep(0.35)
    heus = sum(1 for b in sink.values() if classify(b.get("spacegroup_relax"), b.get("composition")))
    stats.append((label, got, heus))
    print(f"    {label:<26}{got:>6} rows", flush=True)
    return got


def main(sleep: float = 0.35) -> int:
    s = requests.Session()
    s.headers["User-Agent"] = "heusler-gap-mining/1.0"

    T = pd.read_excel(TARGETS, sheet_name="Targets")
    tgt_pairs, tgt_els = collections.Counter(), collections.Counter()
    for f in T.compound.unique():
        e = els(f)
        for x in e:
            tgt_els[x] += 1
        for p in itertools.combinations(e, 2):
            tgt_pairs[p] += 1

    pool = pd.read_csv(POOL)
    have_el, have_pair = collections.Counter(), collections.Counter()
    for f in pool.formula.dropna().unique():
        try:
            e = els(f)
        except Exception:  # noqa: BLE001
            continue
        for x in e:
            have_el[x] += 1
        for p in itertools.combinations(e, 2):
            have_pair[p] += 1

    thin = [e for e in tgt_els if have_el.get(e, 0) < 20]
    missing = [p for p, _ in tgt_pairs.most_common() if have_pair.get(p, 0) == 0]
    print(f"thin elements (<20 compounds in pool): {sorted(thin)}")
    print(f"uncovered target pairs: {len(missing)}\n")

    raw, stats = {}, []
    print("PASS 1 - uncovered pairs, any space group (filtered to Heusler afterwards)")
    for a, b in missing:
        harvest(s, f"species({a}),species({b}),agl_thermal_conductivity_300K(*)", f"{a}-{b}",
                raw, stats)
        time.sleep(sleep)

    print("\nPASS 2 - thin elements, restricted to Heusler space groups")
    for e in sorted(thin):
        for sg in (216, 225, 119, 139):
            harvest(s, f"species({e}),spacegroup_relax({sg}),agl_thermal_conductivity_300K(*)",
                    f"{e} sg{sg}", raw, stats)
            time.sleep(sleep)

    print("\nPASS 3 - quaternary Heuslers (1:1:1:1), never harvested before")
    for sg in (216, 225):
        harvest(s, f"nspecies(4),spacegroup_relax({sg}),agl_thermal_conductivity_300K(*)",
                f"quaternary sg{sg}", raw, stats)
        time.sleep(sleep)

    print(f"\n  {len(raw)} unique AFLOW entries retrieved")

    recs = []
    for r in raw.values():
        cls = classify(r.get("spacegroup_relax"), r.get("composition"))
        if cls is None:                      # HEUSLERS ONLY -- the standing constraint
            continue
        k = pd.to_numeric(pd.Series([r.get("agl_thermal_conductivity_300K")]),
                          errors="coerce").iloc[0]
        if pd.isna(k) or k <= 0:
            continue
        geo = r.get("geometry") or []
        sp = r.get("species") or []
        try:
            formula = Composition(str(r.get("compound", ""))).reduced_formula
        except Exception:  # noqa: BLE001
            continue
        aurl = r.get("aurl", "")
        recs.append(dict(
            formula=formula, compound=r.get("compound"),
            kappa_L=k, kappa_units="W/m/K", temperature_K=300.0,
            method="AGL quasi-harmonic Debye-Gruneisen",
            method_class="semi-empirical (AGL Debye-Gruneisen, NOT BTE)",
            heusler_class=cls, spacegroup_number=r.get("spacegroup_relax"),
            # geometry[0] is the PRIMITIVE fcc edge -> conventional cubic (measured ratio 1.4103)
            a_A=(float(geo[0]) * SQRT2) if len(geo) > 5 and geo[0] else None,
            a_primitive_A=geo[0] if len(geo) > 5 else None,
            lattice_convention="primitive->conventional (x sqrt2)",
            species=",".join(str(x) for x in sp),
            frac_coords=json.dumps(r.get("positions_fractional"))
            if r.get("positions_fractional") else None,
            natoms=r.get("natoms"), volume_atom=r.get("volume_atom"), density=r.get("density"),
            debye_K=r.get("agl_debye"), gruneisen=r.get("agl_gruneisen"),
            Cp_300K=r.get("agl_heat_capacity_Cp_300K"),
            bulk_vrh=r.get("ael_bulk_modulus_vrh"), shear_vrh=r.get("ael_shear_modulus_vrh"),
            data_type="computed", confidence="ok",
            source="AFLOW AGL (gap-targeted harvest)", source_id=r.get("auid"),
            source_url=("http://" + aurl.replace(":AFLOWDATA", "/AFLOWDATA")) if aurl else None,
            source_doi="10.1016/j.commatsci.2012.02.005",
            structure_source="AFLOW relaxed geometry, same record as the AGL kappa",
            query=r.get("_query")))

    d = pd.DataFrame(recs).drop_duplicates("source_id")
    known = set(pool.formula.dropna())
    d["is_new"] = ~d.formula.isin(known)

    print(f"\n=== GAP HARVEST ===")
    print(f"  Heusler rows with a usable kappa : {len(d)}   formulae {d.formula.nunique()}")
    if len(d):
        print(f"  NEW to the pool                  : {int(d.is_new.sum())} rows, "
              f"{d[d.is_new].formula.nunique()} formulae")
        print(d.heusler_class.value_counts().to_string())
        print(f"  with structure (lattice + coords): "
              f"{int(d.a_A.notna().sum())} / {len(d)}")

        cov = collections.Counter()
        for f in d[d.is_new].formula.unique():
            for p in itertools.combinations(els(f), 2):
                cov[p] += 1
        filled = [p for p in missing if cov.get(p, 0) > 0]
        still = [p for p in missing if cov.get(p, 0) == 0]
        print(f"\n  target pairs FILLED by this harvest: {len(filled)}  "
              f"{[f'{a}-{b}' for a, b in filled]}")
        print(f"  target pairs STILL empty           : {len(still)}  "
              f"{[f'{a}-{b}' for a, b in still]}")
        print("    (these do not exist as computed Heuslers in AFLOW -- reported, not substituted)")
        Path("data/external").mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.35)
    raise SystemExit(main(**vars(ap.parse_args())))
