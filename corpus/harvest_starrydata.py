"""Harvest EXPERIMENTAL kappa-vs-temperature curves for Heuslers from Starrydata.

WHY THIS SOURCE MATTERS MORE THAN ITS SIZE SUGGESTS. The pool is overwhelmingly computational: at
last count only ~14 compounds had a measured kappa. Starrydata is the opposite -- 235,269 curves
digitised from plots in 13,000+ published papers, every one of them a real measurement. It also
gives what almost nothing else does: kappa at MANY temperatures for the same sample, which is what
a thermoelectric actually needs (300-1200 K), rather than a single 300 K number.

Of the 32,135 temperature-resolved kappa curves, 6,680 are already reported by their authors as
LATTICE thermal conductivity -- kappa_L directly, with no subtraction needed.

THE TRAP THIS FILE EXISTS TO AVOID. Starrydata records composition but NOT structure. Filtering on
stoichiometry alone accepts 215 "Heusler-ratio" formulae, and many are nothing of the sort:
AgSbTe2, BiTeI, AlCuO2, CoSbS, CuSbSe2 all satisfy a 1:1:2 or 1:1:1 ratio while being chalcogenides,
halides or delafossites. This is the same mistake that nearly put fluorite (CaF2, ThO2) into the
pool from AFLOW.

So every formula is CONFIRMED against a structure database (MP / OQMD / JARVIS / AFLOW) and kept
only if that formula is recorded in a Heusler space group (216 / 225 / 119 / 139). Measured:

    215 Heusler-ratio formulae
     63 confirmed in a Heusler space group      -> kept
     17 found, but in another space group       -> rejected (AlCuO2 sg194, CoSbS sg198, ...)
    135 no structure record anywhere            -> rejected, because "cannot confirm" is not
                                                   the same as "is a Heusler"

TOTAL vs LATTICE. Where the author published `Lattice thermal conductivity`, it is taken as-is.
Where only total kappa exists, kappa_L = kappa_total - L*T/rho is computed by Wiedemann-Franz using
the sample's OWN resistivity curve, and the row is flagged `wf_subtracted` -- never silently mixed
with directly-reported kappa_L. Rows where no matching resistivity exists are kept as
`thermal_conductivity_total` and are NOT used as kappa_L labels.
"""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from math import gcd
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

OUT = "data/external/kappa_starrydata_heusler.csv"
POOL = "data/external/KAPPA_POOL_MASTER.csv"
REL = "https://github.com/starrydata/starrydata_datasets/releases/download/latest"
HEUSLER_SG = {216, 225, 119, 139}
# Sommerfeld value. Real Lorenz numbers run ~1.5-2.44e-8; using the degenerate limit makes the
# electronic subtraction an UPPER bound, so kappa_L here is if anything slightly low. Flagged.
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


def heusler_formula(f):
    """Stoichiometric Heusler RATIO only -- structure is confirmed separately."""
    try:
        comp = Composition(str(f))
        rc = comp.reduced_composition
    except Exception:  # noqa: BLE001
        return None
    v = [float(x) for x in rc.values()]
    if any(abs(x - round(x)) > 0.02 for x in v):        # doped/off-stoichiometric
        return None
    iv = [int(round(x)) for x in v]
    if len(iv) not in (3, 4) or any(x <= 0 for x in iv):
        return None
    g = 0
    for x in iv:
        g = gcd(g, x)
    r = tuple(sorted(x // g for x in iv))
    return comp.reduced_formula if r in {(1, 1, 1), (1, 1, 2), (1, 1, 1, 1)} else None


def structure_registry() -> dict[str, set[int]]:
    reg: dict[str, set[int]] = {}
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
            r = red(f)
            if r and pd.notna(s):
                reg.setdefault(r, set()).add(int(s))
    return reg


def arr(x):
    """Curve points arrive as a stringified list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        v = ast.literal_eval(str(x))
        return list(v) if isinstance(v, (list, tuple)) else []
    except Exception:  # noqa: BLE001
        return []


def main(curves: str, out: str = OUT) -> int:
    cpath = Path(curves)
    if not cpath.exists():
        print(f"not found: {curves}\n  download from {REL}/all_curves.csv.gz")
        return 1

    print("reading Starrydata curves ...")
    c = pd.read_csv(cpath, compression="gzip",
                    usecols=["SID", "DOI", "composition", "sample_id", "prop_x", "prop_y",
                             "unit_x", "unit_y", "x", "y"])
    c = c[c.prop_x == "Temperature"]
    print(f"  temperature-resolved curves: {len(c)}")

    c["formula"] = c.composition.map(heusler_formula)
    c = c[c.formula.notna()]
    print(f"  Heusler-ratio formulae      : {c.formula.nunique()}")

    reg = structure_registry()
    ok = {f for f in c.formula.unique() if (reg.get(f) or set()) & HEUSLER_SG}
    # Formulae the local registry did not know, confirmed by querying Materials Project directly
    # for their space group. Same standard of proof, different lookup -- so they are kept, not
    # assumed. `CuAgTe` arrived this way and is the pool's first Ag-Cu compound.
    extra = Path("data/external/starrydata_mp_confirmed_heuslers.csv")
    if extra.exists():
        e = set(pd.read_csv(extra).formula.dropna())
        gained = (e & set(c.formula.unique())) - ok
        if gained:
            print(f"  + {len(gained)} confirmed via a direct MP space-group lookup: {sorted(gained)}")
        ok |= e
    rejected = sorted(set(c.formula.unique()) - ok)
    print(f"  CONFIRMED in a Heusler space group: {len(ok)}")
    print(f"  rejected (wrong or unknown structure): {len(rejected)}  e.g. {rejected[:8]}")
    c = c[c.formula.isin(ok)]

    kl = c[c.prop_y == "Lattice thermal conductivity"]
    kt = c[c.prop_y == "Thermal conductivity"]
    # THREE SOURCES OF THE ELECTRONIC PART, NOT ONE.
    # Reading only "Electrical resistivity" ignored 21,865 conductivity curves and 653 curves of
    # kappa_e itself -- roughly half the electrical data in Starrydata -- purely because of how a
    # paper chose to plot it. Units verified against the file: resistivity is ohm*m, conductivity
    # is ohm^-1*m^-1, kappa_e is W/m/K, Seebeck is V/K.
    # `log(Electrical conductivity)` is DELIBERATELY EXCLUDED: its values run to -124, which no
    # logarithm of a conductivity can be, so the column is mislabeled or corrupt.
    rho = c[c.prop_y == "Electrical resistivity"]
    sig = c[c.prop_y == "Electrical conductivity"]
    kel = c[c.prop_y == "Electronic thermal conductivity"]
    seeb = c[c.prop_y == "Seebeck coefficient"]
    print(f"  electrical curves: resistivity {len(rho)}  conductivity {len(sig)}  "
          f"kappa_e {len(kel)}  Seebeck {len(seeb)}")
    print(f"\n  curves: kappa_L {len(kl)}   kappa_total {len(kt)}   resistivity {len(rho)}")

    rows = []

    def emit(r, kind, T, K, note, rho_used=None):
        # A TOTAL thermal conductivity must never occupy the kappa_L column. It is stored under
        # kappa_total with kappa_L left null, so the pool's "no numeric kappa" gate drops it
        # automatically instead of relying on a downstream filter that someone could forget.
        is_total = kind == "total_only"
        rows.append(dict(
            formula=r.formula, composition_reported=r.composition,
            kappa_L=None if is_total else K,
            kappa_total=K if is_total else None,
            kappa_units="W/m/K", temperature_K=T,
            method=note, method_class="experimental",
            data_type="experimental", provenance="primary",
            kappa_basis=kind, resistivity_ohm_m=rho_used,
            source="Starrydata (digitised from published plots)",
            source_doi=r.DOI, source_id=r.SID, sample_id=r.sample_id,
            source_url=f"https://doi.org/{r.DOI}" if pd.notna(r.DOI) else None,
            source_location="Starrydata all_curves.csv (release 'latest')",
            dataset_url=f"{REL}/all_curves.csv.gz", confidence="ok"))

    # 1. author-reported lattice kappa -- no assumption at all
    for r in kl.itertuples():
        for T, K in zip(arr(r.x), arr(r.y)):
            try:
                T, K = float(T), float(K)
            except Exception:  # noqa: BLE001
                continue
            if T > 0 and K > 0:
                emit(r, "reported_lattice", T, K,
                     "experimental, lattice thermal conductivity as published")

    # 2. total kappa minus Wiedemann-Franz electronic part, using the SAME sample's resistivity
    rho_by_sample: dict = {}
    for r in rho.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            rho_by_sample[r.sample_id] = (np.array(xs, dtype=float), np.array(ys, dtype=float))

    # conductivity curves become resistivity curves: rho = 1/sigma
    for r in sig.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            a = np.array(xs, dtype=float)
            b = np.array(ys, dtype=float)
            good = b > 0
            if good.any() and r.sample_id not in rho_by_sample:
                rho_by_sample[r.sample_id] = (a[good], 1.0 / b[good])

    # kappa_e measured directly -- better than any Wiedemann-Franz estimate, because it needs no
    # assumption about the Lorenz number at all
    ke_by_sample: dict = {}
    for r in kel.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            ke_by_sample[r.sample_id] = (np.array(xs, dtype=float), np.array(ys, dtype=float))

    # Seebeck lets the Lorenz number be estimated instead of assumed. The degenerate value
    # 2.44e-8 is the metallic limit; for a semiconductor it overestimates kappa_e and therefore
    # UNDERestimates kappa_L. Kim, Gibbs, Tang, Wang & Snyder, APL Materials 3, 041506 (2015):
    #     L = 1.5 + exp(-|S|/116)   [L in 1e-8 W.ohm/K^2, S in microV/K]
    # This is the standard correction in the thermoelectric literature.
    seeb_by_sample: dict = {}
    for r in seeb.itertuples():
        xs, ys = arr(r.x), arr(r.y)
        if xs and ys:
            seeb_by_sample[r.sample_id] = (np.array(xs, dtype=float),
                                           np.abs(np.array(ys, dtype=float)) * 1e6)

    def lorenz_at(sample_id, T):
        pair = seeb_by_sample.get(sample_id)
        if not pair:
            return LORENZ, "degenerate 2.44e-8"
        xs, ys = pair
        S_uV = float(np.interp(T, xs, ys))
        if not (0 < S_uV < 1000):
            return LORENZ, "degenerate 2.44e-8"
        return (1.5 + np.exp(-S_uV / 116.0)) * 1e-8, f"Kim2015 L(S={S_uV:.0f}uV/K)"

    wf = 0
    bad_rho = 0   # rows rejected by the resistivity plausibility gate
    direct = 0
    for r in kt.itertuples():
        ke_pair = ke_by_sample.get(r.sample_id)
        if ke_pair is not None:
            for T, K in zip(arr(r.x), arr(r.y)):
                try:
                    T, K = float(T), float(K)
                except Exception:  # noqa: BLE001
                    continue
                if not (T > 0 and K > 0):
                    continue
                ke = float(np.interp(T, ke_pair[0], ke_pair[1]))
                kL = K - ke
                if kL > 0:
                    direct += 1
                    emit(r, "total_minus_measured_ke", T, kL,
                         "experimental, kappa_total - MEASURED kappa_e (no Lorenz assumption)")
            continue
        pair = rho_by_sample.get(r.sample_id)
        for T, K in zip(arr(r.x), arr(r.y)):
            try:
                T, K = float(T), float(K)
            except Exception:  # noqa: BLE001
                continue
            if not (T > 0 and K > 0):
                continue
            if pair is None:
                emit(r, "total_only", T, K,
                     "experimental TOTAL thermal conductivity (no resistivity to subtract "
                     "kappa_e) -- NOT a kappa_L label")
                continue
            xs, ys = pair
            if T < xs.min() - 25 or T > xs.max() + 25:
                continue
            rr = float(np.interp(T, xs, ys))
            if rr <= 0:
                continue
            # PLAUSIBILITY GATE ON THE RESISTIVITY, before it is used.
            #
            # A dense thermoelectric intermetallic sits between roughly 1e-8 ohm.m (a good metal)
            # and 1e-2 ohm.m (a poor semiconductor). Values far outside that are a unit error in
            # the source curve -- microohm-cm or mohm-cm read as ohm.m -- and they are silent
            # rather than loud: kappa_e = L*T/rho collapses to ~1e-6 W/m/K, the subtraction removes
            # nothing, and the row is emitted as `wf_subtracted` carrying the TOTAL conductivity
            # under a lattice label. It then looks like a compound whose kappa rises with
            # temperature, because the electronic part that was never removed grows with T.
            #
            # Measured on the current export: 136 rows across 4 compounds, with rho of 10-34 ohm.m
            # -- about 1e6 too large. TiCoSn is 47 of 47 rows, so its whole "lattice" reference was
            # really a total, and it had reached the project's verified-correct list on that basis.
            # NbCoSn is 23 of 146. HfNiSn 63, ZrCoSb 3.
            if not (1e-8 <= rr <= 1e-2):
                bad_rho += 1
                continue
            L_use, L_note = lorenz_at(r.sample_id, T)
            ke = L_use * T / rr
            kL = K - ke
            if kL <= 0:
                continue
            wf += 1
            emit(r, "wf_subtracted", T, kL,
                 f"experimental, Wiedemann-Franz: kappa_total - L*T/rho ({L_note})", rr)

    d = pd.DataFrame(rows)
    if not len(d):
        print("no usable rows")
        return 0
    d = d.drop_duplicates(subset=["formula", "temperature_K", "kappa_L", "kappa_total",
                                  "source_doi"])

    print(f"\n=== STARRYDATA HEUSLER HARVEST ===")
    print(f"  rows {len(d)}   compounds {d.formula.nunique()}")
    print(d.kappa_basis.value_counts().to_string())
    usable = d[d.kappa_L.notna()]
    print(f"\n  usable as kappa_L : {len(usable)} rows, {usable.formula.nunique()} compounds")
    print(f"  T range           : {usable.temperature_K.min():.0f}-"
          f"{usable.temperature_K.max():.0f} K")
    nT = usable.groupby("formula").temperature_K.nunique()
    print(f"  multi-temperature : {int((nT > 1).sum())} compounds")

    if Path(POOL).exists():
        pool = set(pd.read_csv(POOL).formula.dropna())
        new = sorted(set(usable.formula) - pool)
        print(f"\n  NEW vs pool: {len(new)}  {new}")

    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True, help="path to all_curves.csv.gz")
    ap.add_argument("--out", default=OUT)
    raise SystemExit(main(**vars(ap.parse_args())))
