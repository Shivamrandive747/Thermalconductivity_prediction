"""Step 2c — harvest Materials Project for HEUSLER alloys only, in our target chemistry.

USER CONSTRAINT: Heusler alloys only. No perovskites, chalcogenides, oxides or unrelated
intermetallics -- a training pool padded with irrelevant chemistry is worse than a smaller clean one.
Every row here must pass BOTH a Heusler space group AND a Heusler stoichiometry, and every element
must be one the 149 prediction targets actually use.

WHAT MP'S `thermal_conductivity` FIELD ACTUALLY IS -- READ THIS BEFORE USING IT
------------------------------------------------------------------------------
It is NOT lattice thermal conductivity. It holds `clarke` and `cahill` minimum thermal
conductivity: the AMORPHOUS LIMIT, i.e. what the material would conduct if it were a glass with
phonon mean free paths cut to one interatomic spacing. For elemental Pd, MP reports clarke = 0.66
W/m/K while the real value is ~72. Treating kappa_min as kappa_L would poison the training pool with
values an order of magnitude low, so it is stored under `kappa_min_*` names and never as `kappa_L`.

WHAT MP IS GENUINELY WORTH HAVING FOR
-------------------------------------
`debye_temperature`, `sound_velocity` (transverse / longitudinal), `bulk_modulus`, `shear_modulus`,
`young_modulus`, `homogeneous_poisson`, `universal_anisotropy` -- and a full `structure`. These are
the Slack-model inputs that physically determine kappa_L, and ML_PLAN.md L4 calls physics
intermediates of exactly this kind "the single highest-leverage feature addition". 13,283 MP entries
carry them.

API NOTES (established by probing, not assumed)
-----------------------------------------------
`/materials/elasticity/` accepts ONLY `chemsys` and `material_ids`. It rejects `formula`,
`elements`, `nelements`, `crystal_system` and `spacegroup_number` outright. So Heusler filtering is
done client-side on the returned `symmetry` and `composition_reduced`.

PROVENANCE: every row carries `source`, `source_id` (the MP material_id), `source_url` (the
resolvable materialsproject.org page) and the query that produced it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
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

from pipeline.config import cfg  # noqa: F401  (loads .env)

API = "https://api.materialsproject.org"
OUT = "data/external/mp_heusler_elastic.csv"
TARGETS = "data/Target_Materials/Target_Dossier_166.xlsx"

# Heusler space groups and the stoichiometry each demands.
HEUSLER_SG = {216: "F-43m", 225: "Fm-3m", 119: "I-4m2", 139: "I4/mmm"}
HEUSLER = {(216, (1, 1, 1)): "half_C1b",
           (225, (1, 1, 2)): "full_L21",
           (216, (1, 1, 2)): "inverse_XA",
           (139, (1, 1, 2)): "full_tetragonal",
           (119, (1, 1, 2)): "inverse_tetragonal",
           (119, (1, 1, 1)): "half_tetragonal",
           (216, (1, 1, 1, 1)): "quaternary_Y"}

FIELDS = ("material_id,formula_pretty,composition_reduced,symmetry,structure,"
          "bulk_modulus,shear_modulus,young_modulus,homogeneous_poisson,"
          "universal_anisotropy,debye_temperature,sound_velocity,thermal_conductivity,"
          "density,nsites,elements")


def ratio_of(comp_reduced) -> tuple[int, ...] | None:
    """{'Cu': 2.0, 'Zn': 1.0, 'Al': 1.0} -> (1, 1, 2) sorted and reduced."""
    try:
        v = [int(round(float(x))) for x in comp_reduced.values()]
    except Exception:  # noqa: BLE001
        return None
    if not v or any(x <= 0 for x in v):
        return None
    g = 0
    for x in v:
        g = gcd(g, x)
    return tuple(sorted(x // g for x in v)) if g else None


def classify(sg_number, comp_reduced) -> str | None:
    r = ratio_of(comp_reduced or {})
    if r is None:
        return None
    try:
        return HEUSLER.get((int(sg_number), r))
    except Exception:  # noqa: BLE001
        return None


def main(sleep: float = 0.35) -> int:
    key = os.getenv("MP_API_KEY")
    if not key:
        print("MP_API_KEY not found in environment")
        return 1
    s = requests.Session()
    s.headers["X-API-KEY"] = key

    T = pd.read_excel(TARGETS, sheet_name="Targets")
    triples, elements = set(), set()
    for f in T.compound.unique():
        e = sorted(str(x) for x in Composition(f).elements)
        triples.add("-".join(e))
        elements.update(e)
    # also every 3-element combination of target elements -- Heuslers we do not target but
    # which share our chemistry still teach the structure-property relation
    extra = {"-".join(sorted(c)) for c in itertools.combinations(sorted(elements), 3)}
    chemsys = sorted(triples | extra)
    print(f"target element triples: {len(triples)}   all combinations of target elements: {len(extra)}")
    print(f"querying {len(chemsys)} chemical systems\n")

    rows, seen = [], set()
    hit = miss = 0
    for i, cs in enumerate(chemsys, 1):
        try:
            r = s.get(f"{API}/materials/elasticity/",
                      params={"chemsys": cs, "_fields": FIELDS, "_limit": 100}, timeout=45)
            if r.status_code != 200:
                miss += 1
                time.sleep(sleep)
                continue
            data = r.json().get("data", [])
        except Exception:  # noqa: BLE001
            miss += 1
            time.sleep(2)
            continue

        for x in data:
            mid = x.get("material_id")
            if mid in seen:
                continue
            sym = x.get("symmetry") or {}
            sgn = sym.get("number")
            cls = classify(sgn, x.get("composition_reduced"))
            if cls is None:                      # HEUSLERS ONLY
                continue
            seen.add(mid)
            bm = x.get("bulk_modulus") or {}
            sm = x.get("shear_modulus") or {}
            sv = x.get("sound_velocity") or {}
            tc = x.get("thermal_conductivity") or {}
            st = x.get("structure") or {}
            lat = (st.get("lattice") or {}) if isinstance(st, dict) else {}
            rows.append(dict(
                source="MaterialsProject", source_id=mid,
                source_url=f"https://materialsproject.org/materials/{mid}",
                query=f"elasticity?chemsys={cs}",
                data_origin="theoretical", method_class="DFT_elastic",
                compound=x.get("formula_pretty"),
                reduced_formula=x.get("formula_pretty"),
                elements=",".join(sorted(x.get("elements") or [])),
                n_elements=len(x.get("elements") or []),
                heusler_class=cls, is_heusler=True,
                spacegroup=sgn, spacegroup_symbol=sym.get("symbol"),
                crystal_system=sym.get("crystal_system"),
                a=lat.get("a"), b=lat.get("b"), c=lat.get("c"),
                alpha=lat.get("alpha"), beta=lat.get("beta"), gamma=lat.get("gamma"),
                volume=lat.get("volume"), nsites=x.get("nsites"), density=x.get("density"),
                # the Slack-model backbone -- the real reason to harvest MP
                debye_K=x.get("debye_temperature"),
                sound_transverse=sv.get("transverse"), sound_longitudinal=sv.get("longitudinal"),
                sound_snyder_acoustic=sv.get("snyder_acoustic"),
                bulk_vrh=bm.get("vrh"), shear_vrh=sm.get("vrh"),
                young_modulus=x.get("young_modulus"),
                poisson=x.get("homogeneous_poisson"),
                anisotropy=x.get("universal_anisotropy"),
                # MINIMUM thermal conductivity -- the amorphous limit, NOT kappa_L
                kappa_min_clarke=tc.get("clarke"), kappa_min_cahill=tc.get("cahill"),
                structure_json=json.dumps(st)[:60000] if st else None))
            hit += 1
        if i % 25 == 0:
            print(f"  {i}/{len(chemsys)} systems -> {len(rows)} Heuslers", flush=True)
        time.sleep(sleep)

    d = pd.DataFrame(rows)
    print(f"\n=== MATERIALS PROJECT HARVEST (Heuslers only) ===")
    print(f"  chemical systems queried : {len(chemsys)}  (failed: {miss})")
    print(f"  HEUSLER records          : {len(d)}")
    if len(d):
        print(d.heusler_class.value_counts().to_string())
        print(f"\n  with Debye temperature : {d.debye_K.notna().sum()}/{len(d)}")
        print(f"  with sound velocity    : {d.sound_transverse.notna().sum()}/{len(d)}")
        print(f"  with elastic moduli    : {d.bulk_vrh.notna().sum()}/{len(d)}")
        print(f"  with full structure    : {d.structure_json.notna().sum()}/{len(d)}")
        print(f"  with kappa_min (Clarke): {d.kappa_min_clarke.notna().sum()}/{len(d)}"
              f"   [AMORPHOUS LIMIT, not kappa_L]")
        tel = set(elements)
        d["in_target_chem"] = [set(str(e).split(",")) <= tel for e in d.elements]
        print(f"  entirely in target chemistry: {int(d.in_target_chem.sum())}")
        Path("data/external").mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.35)
    raise SystemExit(main(**vars(ap.parse_args())))
