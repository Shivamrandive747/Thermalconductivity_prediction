"""Step 2b — grow the theoretical kappa pool past 1,000, aimed at the chemistry we must predict.

The first harvest took every ternary in a Heusler space group and got 285 Heuslers. Surveying those
against the 149 targets shows the remaining hole is not elements but PAIRS: 57 of the 179 element
pairs the targets need still have zero coverage, and they cluster in the noble-metal / Zn / Ga block
-- Ag-Cu (7 targets), Al-Zn (7), Ga-Zn (6), Ag-Ga (5), In-Zn (4).

So this pass queries the GAPS directly rather than harvesting more of what we already hold. For each
uncovered pair it asks AFLOW for every compound containing both elements that carries a computed
kappa, at any space group and any arity. A Cu-Zn compound that is not a Heusler still teaches the
model how copper and zinc conduct heat together, which is exactly what is missing.

PROVENANCE IS RECORDED ON EVERY ROW. `source`, `source_id` (the AFLOW auid), `source_url` (the
resolvable aurl), `method_class` and `query` -- the literal AFLUX string that produced the row -- so
any number can be traced back to where it came from and re-fetched.

METHOD HONESTY. Everything here is AGL quasi-harmonic Debye-Gruneisen: measured against our 44
experimental Heuslers it runs 38% median error and reads ~23% high. It is the COVERAGE tier,
weighted 0.25, and must never be pooled with phonon-BTE values or used for validation.

A NOTE ON THE AFLUX API. It silently drops any row missing a requested field, so an optional field
shrinks the whole result set -- asking for `enthalpy_formation_atom` once collapsed a 279-row query
to 40. Only fields present on essentially every entry are requested.
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
PAGE = 1000
OUT = "data/external/theory_kappa_pool.csv"
EXISTING = "data/external/aflow_agl_heusler.csv"
TARGETS = "data/Target_Materials/Target_Dossier_166.xlsx"

FIELDS = ["compound", "auid", "aurl", "spacegroup_relax", "Pearson_symbol_relax",
          "geometry", "species", "composition", "positions_fractional", "natoms",
          "volume_atom", "density", "agl_thermal_conductivity_300K", "agl_debye",
          "agl_gruneisen", "agl_heat_capacity_Cp_300K", "agl_bulk_modulus_static_300K",
          "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh"]

HEUSLER = {(216, (1, 1, 1)): "half_C1b",
           (225, (1, 1, 2)): "full_L21",
           (216, (1, 1, 2)): "inverse_XA",
           (225, (1, 1, 1, 1)): "quaternary_Y",
           (139, (1, 1, 2)): "full_tetragonal",
           (119, (1, 1, 2)): "inverse_tetragonal",
           (119, (1, 1, 1)): "half_tetragonal"}


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
    if r is None:
        return None
    try:
        return HEUSLER.get((int(sg), r))
    except Exception:  # noqa: BLE001
        return None


def fetch(session, filt, page):
    q = filt + "," + ",".join(f"{f}(*)" for f in FIELDS) + f",$paging({page},{PAGE})"
    for attempt in range(4):
        try:
            r = session.get(BASE + q, timeout=120)
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


def harvest(session, filt, label):
    out, page = [], 1
    while True:
        batch, url = fetch(session, filt, page)
        if not batch:
            break
        for b in batch:
            b["_query"] = filt
        out.extend(batch)
        if len(batch) < PAGE:
            break
        page += 1
        time.sleep(0.4)
        if page > 15:
            break
    print(f"    {label:<28}{len(out):>6}", flush=True)
    return out


def to_record(r):
    comp = r.get("composition")
    sg = r.get("spacegroup_relax")
    geo = r.get("geometry") or []
    sp = r.get("species") or []
    try:
        formula = Composition(str(r.get("compound", ""))).reduced_formula
    except Exception:  # noqa: BLE001
        formula = str(r.get("compound", ""))
    cls = classify(sg, comp)
    aurl = r.get("aurl", "")
    return dict(
        source="AFLOW_AGL", source_id=r.get("auid"),
        source_url=("http://" + aurl.replace(":AFLOWDATA", "/AFLOWDATA")) if aurl else None,
        query=r.get("_query"), method_class="AGL_Debye", weight=0.25,
        data_origin="theoretical",
        compound=r.get("compound"), reduced_formula=formula,
        elements=",".join(str(x) for x in sp), n_elements=len(sp),
        composition=",".join(str(x) for x in (comp or [])),
        heusler_class=cls, is_heusler=cls is not None,
        spacegroup=sg, pearson=r.get("Pearson_symbol_relax"),
        a=geo[0] if len(geo) > 5 else None, b=geo[1] if len(geo) > 5 else None,
        c=geo[2] if len(geo) > 5 else None, alpha=geo[3] if len(geo) > 5 else None,
        beta=geo[4] if len(geo) > 5 else None, gamma=geo[5] if len(geo) > 5 else None,
        natoms=r.get("natoms"), volume_atom=r.get("volume_atom"), density=r.get("density"),
        kappa_300K=r.get("agl_thermal_conductivity_300K"),
        debye_K=r.get("agl_debye"), gruneisen=r.get("agl_gruneisen"),
        Cp_300K=r.get("agl_heat_capacity_Cp_300K"),
        bulk_static=r.get("agl_bulk_modulus_static_300K"),
        bulk_vrh=r.get("ael_bulk_modulus_vrh"), shear_vrh=r.get("ael_shear_modulus_vrh"),
        positions=json.dumps(r.get("positions_fractional")) if r.get("positions_fractional") else None)


def main(min_target: int = 1000) -> int:
    s = requests.Session()
    s.headers["User-Agent"] = "heusler-kappa-expand/1.0"

    T = pd.read_excel(TARGETS, sheet_name="Targets")
    tgt_pairs = collections.Counter()
    tgt_els = collections.Counter()
    for f in T.compound.unique():
        e = els(f)
        for x in e:
            tgt_els[x] += 1
        for p in itertools.combinations(e, 2):
            tgt_pairs[p] += 1

    raw = {}
    prev = pd.read_csv(EXISTING) if Path(EXISTING).exists() else pd.DataFrame()
    have_pairs = collections.Counter()
    if len(prev):
        for f in prev.reduced_formula.dropna().unique():
            try:
                for p in itertools.combinations(els(f), 2):
                    have_pairs[p] += 1
            except Exception:  # noqa: BLE001
                pass

    missing = [p for p, _ in tgt_pairs.most_common() if have_pairs.get(p, 0) == 0]
    print(f"target element pairs: {len(tgt_pairs)}   uncovered: {len(missing)}")
    print("\nPASS 1 — the uncovered pairs, any space group, any arity")
    for a, b in missing:
        rows = harvest(s, f"species({a}),species({b}),agl_thermal_conductivity_300K(*)",
                       f"{a}-{b}")
        for r in rows:
            raw[r.get("auid")] = r
        time.sleep(0.3)

    print(f"\n  after pass 1: {len(raw)} new entries")

    print("\nPASS 2 — every target element, any space group (breadth)")
    for e, _ in tgt_els.most_common():
        rows = harvest(s, f"species({e}),agl_thermal_conductivity_300K(*)", e)
        for r in rows:
            raw.setdefault(r.get("auid"), r)
        time.sleep(0.3)

    recs = [to_record(r) for r in raw.values()]
    d = pd.DataFrame(recs)
    if len(prev):
        prev = prev.rename(columns={"auid": "source_id", "aurl": "source_url"})
        for c in d.columns:
            if c not in prev.columns:
                prev[c] = None
        d = pd.concat([prev[d.columns], d], ignore_index=True)
    d["kappa_300K"] = pd.to_numeric(d.kappa_300K, errors="coerce")
    d = d[d.kappa_300K.notna() & (d.kappa_300K > 0)]
    d = d.drop_duplicates("source_id")

    # keep only chemistry the targets actually use
    tel = set(tgt_els)
    d["all_in_target_chem"] = [
        bool(set(str(e).split(",")) <= tel) if isinstance(e, str) else False for e in d.elements]

    print(f"\n=== POOL ===")
    print(f"  total rows            : {len(d)}   distinct formulae: {d.reduced_formula.nunique()}")
    print(f"  in target chemistry   : {int(d.all_in_target_chem.sum())}")
    print(f"  Heusler prototypes    : {int(d.is_heusler.sum())}")
    if d.is_heusler.any():
        print(d[d.is_heusler].heusler_class.value_counts().to_string())

    cov = collections.Counter()
    for f in d.reduced_formula.dropna().unique():
        try:
            for p in itertools.combinations(els(f), 2):
                cov[p] += 1
        except Exception:  # noqa: BLE001
            pass
    still = [p for p in tgt_pairs if cov.get(p, 0) == 0]
    print(f"\n  target pairs still uncovered: {len(still)} of {len(tgt_pairs)}")
    if still:
        print(f"    {[f'{a}-{b}' for a, b in still[:20]]}")

    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    if len(d) < min_target:
        print(f"  NOTE: {len(d)} < {min_target} target; further sources needed "
              f"(PhononDB, HH130, MP, literature)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-target", type=int, default=1000)
    raise SystemExit(main(**vars(ap.parse_args())))
