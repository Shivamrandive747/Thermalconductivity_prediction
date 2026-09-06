"""Fetch real, cited structures for compounds that have kappa but no structure.

WHY THIS EXISTS. `resolve_structures.py` excludes any compound without a sourced structure -- it
refuses to invent a lattice constant, which is correct and is the rule this project runs on. But
20 compounds were just excluded that way, and four of them are tier-1 full Heuslers we mined kappa
for this week (`K2AgBi`, `K2AgSb`, `Na2AgBi`, `Na2AgSb`) plus `TiSbRu2`, which is Ru2 chemistry --
the exact gap the model has. Those compounds are not structure-less in reality, only in our files:
the OPTIMADE probe confirmed every one exists in at least one public database.

TWO TRAPS THIS PROJECT HAS ALREADY PAID FOR, BOTH AVOIDED HERE BY CONSTRUCTION:

1. PRIMITIVE vs CONVENTIONAL. Four sources publish the primitive fcc edge, a factor sqrt(2) from
   the conventional cubic constant, and Materials Project's bare `a` is the trap. Rather than read
   a published number and guess which convention it follows, this pulls the full LATTICE VECTORS
   and SITE POSITIONS and hands them to pymatgen's SpacegroupAnalyzer, which returns the
   conventional cell. The number is computed, never assumed.

2. COMPOSITION IS NOT STRUCTURE. `Ag3Sn` at 3:1 and binary sg-225 fluorite both look like Heuslers
   by stoichiometry alone; this project once nearly ingested CaF2 and ThO2 as Heuslers. So the
   space group is taken from the symmetry analysis of the real structure, and anything outside the
   Heusler space groups is written out with its actual space group rather than silently kept.

Nothing is fabricated: every row carries the provider, the entry id and the immutable id.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests
import urllib3
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

urllib3.disable_warnings()

MAIL = "teammaterialscienceteam@gmail.com"
S = requests.Session()
S.headers["User-Agent"] = f"heusler-struct/1.0 (mailto:{MAIL})"
OUT = "data/external/optimade_recovered_structures.csv"
EXCLUDED = "data/external/KAPPA_POOL_no_structure_EXCLUDED.csv"

# Providers verified reachable earlier this session, best-first. Alexandria is a prototype-search
# database full of hypothetical polymorphs, so it is queried LAST and its space group is never
# trusted -- it reports the known L21 compound Sr2BiAu as space group 123.
PROVIDERS = [
    ("mp", "https://optimade.materialsproject.org/v1"),
    ("oqmd", "https://oqmd.org/optimade/v1"),
    ("cod", "https://www.crystallography.net/cod/optimade/v1"),
    ("nmd", "https://nomad-lab.eu/prod/v1/api/v1/optimade"),
    ("alexandria", "https://alexandria.icams.rub.de/pbe/v1"),
]
HEUSLER_SG = {216, 225, 119, 139, 221}     # C1b, L21/fluorite, and the inverse/tetragonal variants


def target_list(extra: list[str] | None = None) -> list[str]:
    names: list[str] = []
    try:
        d = pd.read_csv(EXCLUDED)
        col = next(c for c in ("formula", "reduced_formula", "compound") if c in d.columns)
        names = sorted({str(x) for x in d[col].dropna()})
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read {EXCLUDED}: {exc}")
    return sorted(set(names) | set(extra or []))


def query(base: str, flt: str, limit: int = 10):
    """Returns list on success, None on FAILURE. Never conflates the two."""
    for attempt in (0, 1, 2):
        try:
            r = S.get(f"{base}/structures", params={"filter": flt, "page_limit": limit},
                      timeout=60, verify=False)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json().get("data", [])
            except Exception:  # noqa: BLE001
                return None
        if r.status_code in (429, 503):
            time.sleep(3 * (attempt + 1))
            continue
        return None
    return None


def to_structure(attrs) -> Structure | None:
    """Build a pymatgen Structure from an OPTIMADE record."""
    try:
        lat = attrs.get("lattice_vectors")
        pos = attrs.get("cartesian_site_positions")
        sp = attrs.get("species_at_sites")
        if not lat or not pos or not sp:
            return None
        species = []
        smap = {s["name"]: s for s in (attrs.get("species") or [])}
        for name in sp:
            info = smap.get(name)
            if info and info.get("chemical_symbols"):
                species.append(info["chemical_symbols"][0])
            else:
                species.append(name)
        if any(s in ("vacancy", "X") for s in species):
            return None
        return Structure(Lattice(np.array(lat, dtype=float)), species,
                         np.array(pos, dtype=float), coords_are_cartesian=True)
    except Exception:  # noqa: BLE001
        return None


def analyse(st: Structure) -> dict | None:
    """Conventional cell and space group, COMPUTED from the structure."""
    try:
        sga = SpacegroupAnalyzer(st, symprec=0.1)
        sg = int(sga.get_space_group_number())
        conv = sga.get_conventional_standard_structure()
        a, b, c = conv.lattice.abc
        cubic = (abs(a - b) / a < 0.02) and (abs(a - c) / a < 0.02)
        return dict(spacegroup_number=sg, spacegroup_symbol=sga.get_space_group_symbol(),
                    a=round(a, 5), b=round(b, 5), c=round(c, 5),
                    alpha=round(conv.lattice.angles[0], 3),
                    beta=round(conv.lattice.angles[1], 3),
                    gamma=round(conv.lattice.angles[2], 3),
                    volume_A3=round(conv.volume, 4), natoms=len(conv),
                    volume_per_atom_A3=round(conv.volume / max(1, len(conv)), 5),
                    is_cubic=cubic,
                    frac_coords=json.dumps([[round(float(x), 6) for x in s.frac_coords]
                                            for s in conv]),
                    species=json.dumps([str(s.specie) for s in conv]))
    except Exception:  # noqa: BLE001
        return None


def main(limit: int | None = None, extra: str | None = None) -> int:
    targets = target_list([x.strip() for x in (extra or "").split(",") if x.strip()])
    if limit:
        targets = targets[:limit]
    print(f"compounds needing a structure: {len(targets)}")
    print(f"  {targets}\n")

    rows, failures = [], []
    for i, f in enumerate(targets, 1):
        try:
            comp = Composition(f)
        except Exception:  # noqa: BLE001
            continue
        els = sorted(str(e) for e in comp.elements)
        flt = ("elements HAS ALL " + ",".join(f'"{e}"' for e in els)
               + f" AND nelements={len(els)}")
        target_red = comp.reduced_formula
        best = None
        for pid, base in PROVIDERS:
            data = query(base, flt, limit=20)
            if data is None:
                failures.append(dict(formula=f, provider=pid, reason="request failed"))
                continue
            for rec in data:
                a = rec.get("attributes", {}) or {}
                red = a.get("chemical_formula_reduced") or ""
                try:
                    if Composition(red).reduced_formula != target_red:
                        continue
                except Exception:  # noqa: BLE001
                    continue
                st = to_structure(a)
                if st is None:
                    continue
                info = analyse(st)
                if info is None:
                    continue
                cand = dict(source=f"OPTIMADE:{pid}", source_url=f"{base}/structures/{rec.get('id')}",
                            entry_id=str(rec.get("id")),
                            secondary_id=str(a.get("immutable_id") or ""),
                            method="published crystal structure (OPTIMADE)",
                            formula_reduced=target_red, formula_reported=red,
                            elements=json.dumps(els), n_elements=len(els),
                            structure_source=pid, is_hypothetical=(pid == "alexandria"),
                            **info)
                # prefer a Heusler space group, then a cubic cell, then any
                rank = (0 if info["spacegroup_number"] in HEUSLER_SG else 1,
                        0 if info["is_cubic"] else 1)
                if best is None or rank < best[0]:
                    best = (rank, cand)
            if best and best[0] == (0, 0):
                break                                  # ideal match, stop querying providers
            time.sleep(0.3)
        if best:
            rows.append(best[1])
            c = best[1]
            flag = "" if c["spacegroup_number"] in HEUSLER_SG else "   <- NOT a Heusler sg"
            print(f"  [{i}/{len(targets)}] {f:<12} {c['source']:<20} sg={c['spacegroup_number']:<4}"
                  f" a={c['a']:.4f}{flag}", flush=True)
        else:
            print(f"  [{i}/{len(targets)}] {f:<12} no matching structure in any provider",
                  flush=True)
            failures.append(dict(formula=f, provider="all", reason="no composition match"))

    if not rows:
        print("\nnothing recovered")
        return 0
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    if failures:
        pd.DataFrame(failures).to_csv("data/external/optimade_structure_failures.csv", index=False)

    print(f"\n=== recovered {len(d)} structures for {d.formula_reduced.nunique()} compounds ===")
    print(d.source.value_counts().to_string())
    print("\nby space group:")
    print(d.spacegroup_number.value_counts().to_string())
    heus = d[d.spacegroup_number.isin(HEUSLER_SG)]
    print(f"\nin a Heusler space group: {len(heus)} of {len(d)}")
    print(f"cubic cells             : {int(d.is_cubic.sum())}")
    print(f"\nwrote {OUT}")
    if failures:
        print(f"{len(failures)} failures -> data/external/optimade_structure_failures.csv")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--extra", help="comma-separated extra formulas to fetch")
    raise SystemExit(main(**vars(ap.parse_args())))
