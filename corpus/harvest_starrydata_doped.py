"""Harvest DOPED (site-substituted) Heuslers with measured kappa-vs-T from Starrydata.

WHY DOPED COMPOUNDS ARE WORTH HAVING. `Hf0.6Zr0.4NiSn` is a genuine half-Heusler: the X site simply
carries two elements. It has a real, measured lattice thermal conductivity across a real temperature
range. Thermoelectric Heuslers are almost always studied doped, so restricting to perfectly
stoichiometric compositions throws away most of the experimental literature.

That is NOT the same as `Ti1Sb1Ru1.1`, which is excess ruthenium with no clean site assignment and
was correctly rejected as off-stoichiometric noise.

THE TEST THAT FAILED, AND WHY -- read before loosening anything here.
A first attempt asked only whether the element amounts could be PARTITIONED into groups summing to
1:1:1 or 2:1:1. That admitted `Pb0.5Sn0.5Te1` -- a PbTe-SnTe rocksalt alloy, not a Heusler in any
sense -- along with a flood of other chalcogenides. Composition arithmetic alone cannot identify a
Heusler; it never could, and this is the third time in this project that assuming otherwise let
non-Heuslers in (fluorite CaF2 from AFLOW, AgSbTe2 from Starrydata, now PbSnTe).

SO THE RULE HERE HAS TWO PARTS, AND BOTH MUST PASS:
  1. the amounts partition into 3 groups at 1:1:1, or 3 groups at 2:1:1, or 4 at 1:1:1:1
  2. the PARENT formula -- built from the majority element of each group -- is a compound we have
     independently CONFIRMED sits in a Heusler space group (216/225/119/139)

Part 2 is what kills PbSnTe: its parent would be `PbTe`, which is not a confirmed Heusler.

Every surviving row is labelled `stoichiometry = doped` and carries its `parent_formula`, so a doped
measurement can never be mistaken for a stoichiometric one, and either can be filtered out.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
import warnings
from math import gcd
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

OUT = "data/external/kappa_starrydata_doped_heusler.csv"
REL = "https://github.com/starrydata/starrydata_datasets/releases/download/latest"
HEUSLER_SG = {216, 225, 119, 139}
TOL = 0.06
LORENZ = 2.44e-8

STRUCT_SOURCES = [
    ("data/external/mp_optimade_structures_stability.csv", "formula_reduced", "spacegroup_number"),
    ("data/external/oqmd_structures_stability.csv", "formula_reduced", "spacegroup_number"),
    ("data/external/jarvis_dft_thermal_elastic.csv", "formula_reduced", "spacegroup_number"),
    ("data/external/aflow_agl_heusler.csv", "reduced_formula", "spacegroup"),
    ("data/external/theory_kappa_pool.csv", "reduced_formula", "spacegroup"),
    ("data/external/kappa_repos_mined.csv", "formula", "spacegroup_number"),
]


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def confirmed_heuslers() -> set[str]:
    ok: set[str] = set()
    for path, fcol, sgcol in STRUCT_SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=lambda c: c in (fcol, sgcol))
        except Exception:  # noqa: BLE001
            continue
        if fcol not in d.columns or sgcol not in d.columns:
            continue
        sg = pd.to_numeric(d[sgcol], errors="coerce")
        for f, s in zip(d[fcol], sg):
            if pd.notna(s) and int(s) in HEUSLER_SG:
                r = red(f)
                if r:
                    ok.add(r)
    # plus everything already accepted into the pool
    pool = Path("data/external/KAPPA_POOL_MASTER.csv")
    if pool.exists():
        ok |= set(pd.read_csv(pool).formula.dropna())
    return ok


def parent_of(f, confirmed: set[str]):
    """Return (parent_formula, kind) if the composition is a doped Heusler, else None."""
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return None
    el = [(str(k), float(v)) for k, v in c.get_el_amt_dict().items() if float(v) > 0]
    if not (3 <= len(el) <= 6):
        return None
    tot = sum(a for _, a in el)
    if tot <= 0:
        return None

    for basis, targets, kind in ((3.0, (1, 1, 1), "half_doped"),
                                 (4.0, (2, 1, 1), "full_doped"),
                                 (4.0, (1, 1, 1, 1), "quaternary_doped")):
        scaled = [(k, a * basis / tot) for k, a in el]
        n = len(scaled)
        if n > 6:
            continue
        for assign in itertools.product(range(len(targets)), repeat=n):
            if len(set(assign)) != len(targets):
                continue
            sums = [0.0] * len(targets)
            for i, g in enumerate(assign):
                sums[g] += scaled[i][1]
            if any(abs(sums[i] - targets[i]) > TOL for i in range(len(targets))):
                continue
            # majority element of each group -> the parent compound
            parts = []
            for g in range(len(targets)):
                members = [(scaled[i][0], scaled[i][1]) for i in range(n) if assign[i] == g]
                parts.append(max(members, key=lambda t: t[1])[0])
            if len(set(parts)) != len(targets):
                continue
            cnt = {}
            for e, t in zip(parts, targets):
                cnt[e] = cnt.get(e, 0) + t
            try:
                parent = Composition(" ".join(f"{e}{v}" for e, v in cnt.items())).reduced_formula
            except Exception:  # noqa: BLE001
                continue
            if parent in confirmed:                     # <- the rule that rejects PbSnTe
                return parent, kind
    return None


def arr(x):
    try:
        v = ast.literal_eval(str(x))
        return list(v) if isinstance(v, (list, tuple)) else []
    except Exception:  # noqa: BLE001
        return []


def main(curves: str, out: str = OUT) -> int:
    p = Path(curves)
    if not p.exists():
        print(f"not found: {curves}")
        return 1
    conf = confirmed_heuslers()
    print(f"confirmed Heusler parents available: {len(conf)}")

    c = pd.read_csv(p, compression="gzip",
                    usecols=["SID", "DOI", "composition", "sample_id", "prop_x", "prop_y", "x", "y"])
    c = c[c.prop_x == "Temperature"]
    comps = c.composition.dropna().astype(str).unique()
    print(f"compositions with a temperature curve: {len(comps)}")

    mapping = {}
    for f in comps:
        r = parent_of(f, conf)
        if r:
            mapping[f] = r
    print(f"DOPED Heuslers (parent confirmed): {len(mapping)} compositions")
    if not mapping:
        return 0

    c = c[c.composition.isin(mapping)]
    kl = c[c.prop_y == "Lattice thermal conductivity"]
    kt = c[c.prop_y == "Thermal conductivity"]
    # SAME BUG AS THE PURE HARVESTER, SAME FIX. Starrydata files the electronic contribution
    # under four names; reading only resistivity discarded ~half of it. Units verified against
    # the file: resistivity ohm*m, conductivity ohm^-1*m^-1, kappa_e W/m/K, Seebeck V/K.
    # `log(Electrical conductivity)` is excluded: its values reach -124, impossible for a log.
    rho = c[c.prop_y == "Electrical resistivity"]
    sig = c[c.prop_y == "Electrical conductivity"]
    kel = c[c.prop_y == "Electronic thermal conductivity"]
    seeb = c[c.prop_y == "Seebeck coefficient"]
    print(f"  electrical curves: resistivity {len(rho)}  conductivity {len(sig)}  "
          f"kappa_e {len(kel)}  Seebeck {len(seeb)}")
    print(f"  curves: kappa_L {len(kl)}  kappa_total {len(kt)}  resistivity {len(rho)}")

    rho_by = {}
    for r in rho.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            rho_by[r.sample_id] = (np.array(xs, float), np.array(ys, float))

    for r in sig.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            a = np.array(xs, dtype=float); b = np.array(ys, dtype=float)
            good = b > 0
            if good.any() and r.sample_id not in rho_by:
                rho_by[r.sample_id] = (a[good], 1.0 / b[good])

    ke_by_sample = {}
    for r in kel.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            ke_by_sample[r.sample_id] = (np.array(xs, dtype=float), np.array(ys, dtype=float))

    seeb_by_sample = {}
    for r in seeb.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            seeb_by_sample[r.sample_id] = (np.array(xs, dtype=float),
                                           np.abs(np.array(ys, dtype=float)) * 1e6)

    def lorenz_at(sample_id, T):
        """Kim, Gibbs, Tang, Wang & Snyder, APL Materials 3, 041506 (2015).
        L = 1.5 + exp(-|S|/116) in 1e-8 W.ohm/K^2, S in microV/K. The degenerate 2.44e-8 is the
        metallic limit and the paper reports deviations of 40%+ for non-degenerate bands."""
        pair = seeb_by_sample.get(sample_id)
        if not pair:
            return LORENZ
        S_uV = float(np.interp(T, pair[0], pair[1]))
        if not (0 < S_uV < 1000):
            return LORENZ
        return (1.5 + np.exp(-S_uV / 116.0)) * 1e-8

    rows = []

    def emit(r, kind, T, K, note, rho_used=None):
        parent, hkind = mapping[r.composition]
        is_total = kind == "total_only"
        rows.append(dict(
            formula=r.composition, parent_formula=parent, heusler_class=hkind,
            stoichiometry="doped",
            kappa_L=None if is_total else K, kappa_total=K if is_total else None,
            kappa_units="W/m/K", temperature_K=T, kappa_basis=kind,
            method=note, method_class="experimental", data_type="experimental",
            provenance="primary", resistivity_ohm_m=rho_used,
            source="Starrydata (digitised from published plots)",
            source_doi=r.DOI, source_id=r.SID, sample_id=r.sample_id,
            source_url=f"https://doi.org/{r.DOI}" if pd.notna(r.DOI) else None,
            source_location="Starrydata all_curves.csv (release 'latest')",
            dataset_url=f"{REL}/all_curves.csv.gz", confidence="ok"))

    for r in kl.itertuples():
        for T, K in zip(arr(r.x), arr(r.y)):
            try:
                T, K = float(T), float(K)
            except Exception:  # noqa: BLE001
                continue
            if T > 0 and K > 0:
                emit(r, "reported_lattice", T, K,
                     "experimental, lattice thermal conductivity as published (doped composition)")

    for r in kt.itertuples():
        # A MEASURED kappa_e beats any Wiedemann-Franz estimate, because it assumes no Lorenz
        # number at all. Use it whenever the same sample has one.
        ke_pair = ke_by_sample.get(r.sample_id)
        if ke_pair is not None:
            for T, K in zip(arr(r.x), arr(r.y)):
                try:
                    T, K = float(T), float(K)
                except Exception:  # noqa: BLE001
                    continue
                if not (T > 0 and K > 0):
                    continue
                kL = K - float(np.interp(T, ke_pair[0], ke_pair[1]))
                if kL > 0:
                    emit(r, "total_minus_measured_ke", T, kL,
                         "experimental, kappa_total - MEASURED kappa_e (no Lorenz assumption), "
                         "doped composition")
            continue
        pair = rho_by.get(r.sample_id)
        for T, K in zip(arr(r.x), arr(r.y)):
            try:
                T, K = float(T), float(K)
            except Exception:  # noqa: BLE001
                continue
            if not (T > 0 and K > 0):
                continue
            if pair is None:
                emit(r, "total_only", T, K, "experimental TOTAL kappa, no resistivity to subtract")
                continue
            xs, ys = pair
            if T < xs.min() - 25 or T > xs.max() + 25:
                continue
            rr = float(np.interp(T, xs, ys))
            if rr <= 0:
                continue
            kL = K - lorenz_at(r.sample_id, T) * T / rr
            if kL > 0:
                emit(r, "wf_subtracted", T, kL,
                     f"experimental, Wiedemann-Franz kappa_total - L*T/rho "
                     f"(L={lorenz_at(r.sample_id, T):.2e}, Kim2015 from Seebeck), "
                     f"doped composition", rr)

    d = pd.DataFrame(rows).drop_duplicates(
        subset=["formula", "temperature_K", "kappa_L", "kappa_total", "source_doi"])
    usable = d[d.kappa_L.notna()]
    print(f"\n=== DOPED HEUSLER HARVEST ===")
    print(f"  rows {len(d)}   compositions {d.formula.nunique()}   "
          f"parent compounds {d.parent_formula.nunique()}")
    print(f"  usable as kappa_L : {len(usable)} rows, {usable.formula.nunique()} compositions")
    if len(usable):
        print(f"  T range           : {usable.temperature_K.min():.0f}-"
              f"{usable.temperature_K.max():.0f} K")
        print(f"  papers            : {usable.source_doi.nunique()}")
        print(f"  parent breakdown  : {dict(usable.heusler_class.value_counts())}")
        print(f"  example parents   : {sorted(usable.parent_formula.unique())[:14]}")
    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out", default=OUT)
    raise SystemExit(main(**vars(ap.parse_args())))
