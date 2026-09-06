"""Harvest the Heusler subset of the Phonix database — first-principles anharmonic kappa_L.

WHAT PHONIX IS. An automated VASP + ALAMODE workflow (`auto-kappa`) applied brute-force to ~20,000
Materials Project / Phonondb compounds, yielding real three-phonon Boltzmann-transport thermal
conductivity for 6,800+ of them. Published as Ohnishi et al., npj Comput. Mater. 12, 150 (2026),
DOI 10.1038/s41524-026-02033-w (arXiv:2504.21245). Licensed CC BY 4.0, so it is ours to use with
attribution.

    kappa_L = kp + kc     Peierls (particle) + coherence contributions, i.e. the Wigner formulation

THE TEMPERATURE IS 300 K, AND THAT IS QUOTED, NOT ASSUMED. The summary table has no temperature
column, which is exactly the kind of gap that invites a guess. The paper settles it:

    "thermal conductivity at 300 K obtained using the densest q-mesh in auto-kappa -- 1500
     q-points^3/atom -- for all ..."                                            (arXiv:2504.21245)
    "The average kappa_lat at 300 K was 2.4 Wm-1K-1"

So every row here is stamped `temperature_K = 300.0` with that citation. Had it not been
confirmable, the correct move was to write null and let the pool's 200 K floor drop the rows --
never to assume 300 because it is the common convention.

WHY IT IS WORTH HARVESTING DESPITE BEING SMALL. Measured before writing this file: 1,428 rows sit in
Heusler space groups, 209 pass a Heusler stoichiometry test, 139 distinct compounds carry a kappa --
and 110 of those we already hold. The genuine gain is about 29 compounds. That is a modest number,
but they are TIER 1 (real BTE) and they arrive with a complete `structure` JSON, which makes them
grade-A structures rather than a bare lattice constant.

HONESTY ABOUT WHAT THE NEW ONES ARE. Several of the new entries (AcOF, KCuO, CsAgO, LaOF) are oxides
and fluorides that merely satisfy a 1:1:1 ratio in a Heusler space group. They are not
thermoelectric intermetallics. They are kept, because they are real first-principles data on the
Heusler lattice and teach the structure-property relation, but `chemistry_class` flags them so they
can be excluded from any intermetallic-only analysis.

ACCESS. The Hugging Face datasets-server `/filter` endpoint, no authentication required. It is
paginated at 100 rows; the whole Heusler slice is 1,428 rows, so ~15 requests.
"""
from __future__ import annotations

import argparse
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

API = "https://datasets-server.huggingface.co/filter"
DATASET = "phonix-db/phonix-summary"
OUT = "data/external/kappa_phonix_heusler.csv"
POOL = "data/external/KAPPA_POOL_MASTER.csv"

DOI = "10.1038/s41524-026-02033-w"
CITE = ("Ohnishi et al., npj Comput. Mater. 12, 150 (2026) -- Phonix database, "
        "auto-kappa (VASP+ALAMODE) three-phonon BTE")
DS_URL = "https://huggingface.co/datasets/phonix-db/phonix-summary"
TEMPERATURE_K = 300.0          # quoted from the paper -- see module docstring

HEUSLER_SG = (216, 225, 119, 139)
# Elements that make a compound an intermetallic Heusler rather than an oxide/halide lookalike.
NONMETAL = {"O", "F", "Cl", "Br", "I", "N", "S", "Se", "H"}

COLUMNS = ("formula,spg_number,klat[W/mK],kp[W/mK],kc[W/mK],structure,mp_id,unique_id,"
           "volume[A^3],min_phfreq[cm^-1],max_phfreq[cm^-1],nac,scph,four,qmesh,"
           "modulus[GPa],fc2_error[%],fc3_error[%],natoms_prim,natoms_conv")


def heusler_ratio(f) -> tuple[int, ...] | None:
    try:
        c = Composition(str(f)).reduced_composition
    except Exception:  # noqa: BLE001
        return None
    v = [int(round(x)) for x in c.values()]
    if not v or any(x <= 0 for x in v) or len(v) not in (3, 4):
        return None
    g = 0
    for x in v:
        g = gcd(g, x)
    r = tuple(sorted(x // g for x in v)) if g else None
    return r if r in {(1, 1, 1), (1, 1, 2), (1, 1, 1, 1)} else None


def classify(sg, r) -> str | None:
    return {(216, (1, 1, 1)): "half_C1b", (225, (1, 1, 2)): "full_L21",
            (216, (1, 1, 2)): "inverse_XA", (225, (1, 1, 1, 1)): "quaternary_Y",
            (216, (1, 1, 1, 1)): "quaternary_Y", (139, (1, 1, 2)): "full_tetragonal",
            (119, (1, 1, 2)): "inverse_tetragonal", (119, (1, 1, 1)): "half_tetragonal",
            (139, (1, 1, 1)): "half_tetragonal"}.get((int(sg), r))


def fetch(s: requests.Session, sg: int, offset: int, limit: int = 100):
    for attempt in range(6):
        try:
            r = s.get(API, params={"dataset": DATASET, "config": "default", "split": "train",
                                   "where": f'"spg_number"={sg}', "offset": offset,
                                   "limit": limit, "columns": COLUMNS}, timeout=110)
            if r.status_code == 200:
                return r.json()
            # the server builds an index on first use and answers 500 until it is ready
            time.sleep(6 + 3 * attempt)
        except Exception:  # noqa: BLE001
            time.sleep(5 + 2 * attempt)
    return None


def lattice_from_structure(sj):
    """Phonix ships the real cell; take a and the space group from it rather than deriving one."""
    try:
        d = json.loads(sj) if isinstance(sj, str) else sj
        cell = d.get("cell")
        if not cell:
            return None, None, None
        import numpy as np
        m = np.array(cell, dtype=float)
        a, b, c = (float(np.linalg.norm(v)) for v in m)
        return a, b, c
    except Exception:  # noqa: BLE001
        return None, None, None


def main(sleep: float = 0.25) -> int:
    s = requests.Session()
    s.headers["User-Agent"] = "heusler-kappa-harvest/1.0"

    rows = []
    for sg in HEUSLER_SG:
        j = fetch(s, sg, 0)
        if not j:
            print(f"  sg{sg}: FAILED")
            continue
        total = j.get("num_rows_total", 0)
        got = [r["row"] for r in j.get("rows", [])]
        off = 100
        while off < total:
            j2 = fetch(s, sg, off)
            if not j2:
                break
            got += [r["row"] for r in j2.get("rows", [])]
            off += 100
            time.sleep(sleep)
        print(f"  sg{sg}: {len(got)} / {total} rows")
        rows += got

    print(f"\n  fetched {len(rows)} rows in Heusler space groups")

    recs = []
    for r in rows:
        f = r.get("formula")
        ratio = heusler_ratio(f)
        if ratio is None:                      # HEUSLERS ONLY
            continue
        cls = classify(r.get("spg_number"), ratio)
        if cls is None:
            continue
        k = pd.to_numeric(pd.Series([r.get("klat[W/mK]")]), errors="coerce").iloc[0]
        if pd.isna(k) or k <= 0:
            continue
        try:
            red = Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            continue
        a, b, c = lattice_from_structure(r.get("structure"))
        els = {str(e) for e in Composition(str(f)).elements}
        recs.append(dict(
            formula=red, formula_raw=f, kappa_L=float(k), kappa_units="W/m/K",
            temperature_K=TEMPERATURE_K,
            kappa_particle_kp=r.get("kp[W/mK]"), kappa_coherence_kc=r.get("kc[W/mK]"),
            method="three-phonon BTE (auto-kappa: VASP + ALAMODE), kappa = Peierls + coherence",
            method_class="BTE (phono3py-class, Phonix auto-kappa)",
            heusler_class=cls, spacegroup_number=r.get("spg_number"),
            a_A=a, b_A=b, c_A=c, lattice_convention="from published cell (see structure JSON)",
            structure_json=json.dumps(r.get("structure"))
            if not isinstance(r.get("structure"), str) else r.get("structure"),
            volume_A3=r.get("volume[A^3]"), natoms_prim=r.get("natoms_prim"),
            natoms_conv=r.get("natoms_conv"),
            min_phfreq_cm1=r.get("min_phfreq[cm^-1]"), max_phfreq_cm1=r.get("max_phfreq[cm^-1]"),
            bulk_modulus_GPa=r.get("modulus[GPa]"), qmesh=r.get("qmesh"),
            nac=r.get("nac"), scph=r.get("scph"), four_phonon=r.get("four"),
            fc2_error_pct=r.get("fc2_error[%]"), fc3_error_pct=r.get("fc3_error[%]"),
            # an oxide/fluoride in a Heusler space group is real data but not an intermetallic
            chemistry_class=("intermetallic" if not (els & NONMETAL) else "non-intermetallic"),
            data_type="computed", confidence="ok",
            source="Phonix database", source_doi=DOI, source_url=DS_URL,
            source_id=r.get("unique_id") or r.get("mp_id"), mp_id=r.get("mp_id"),
            source_location="phonix-summary, HF datasets-server",
            structure_source="Phonix relaxed cell, same record as kappa",
            citation=CITE, provenance="primary",
            temperature_basis="300 K, stated in arXiv:2504.21245 (not assumed)"))

    d = pd.DataFrame(recs).drop_duplicates(subset=["formula", "kappa_L", "source_id"])
    print(f"  Heusler stoichiometry + usable kappa: {len(d)} rows, {d.formula.nunique()} compounds")

    if Path(POOL).exists():
        known = set(pd.read_csv(POOL).formula.dropna())
        d["is_new"] = ~d.formula.isin(known)
        print(f"  NEW vs current pool                 : {d[d.is_new].formula.nunique()} compounds")
    print(f"\n  {d.heusler_class.value_counts().to_string()}")
    print(f"\n  chemistry: {dict(d.chemistry_class.value_counts())}")
    print(f"  with full structure JSON : {int(d.structure_json.notna().sum())}/{len(d)}")
    print(f"  four-phonon rows         : {int(pd.to_numeric(d.four_phonon, errors='coerce').fillna(0).sum())}")
    print(f"  kappa range              : {d.kappa_L.min():.2f} - {d.kappa_L.max():.2f} W/m/K")

    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.25)
    raise SystemExit(main(**vars(ap.parse_args())))
