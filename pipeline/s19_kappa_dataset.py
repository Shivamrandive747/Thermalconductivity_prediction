"""Stage 19 — load HEUSLER_KAPPA_TRAINING_SET.csv and build the candidate descriptor pool.

WHY A NEW LOADER. Every existing stage reads the old SQLite export
(`data/exports/heusler_v1_0/`). Nothing in the repo reads the new 952-compound CSV, and nothing
consumes its `weight` / `method_tier` columns. This module is the bridge; the modelling code in
s20 reuses the existing helpers rather than reimplementing them.

THE DESCRIPTOR POOL IS DELIBERATELY MODEST (~45, not 159)
---------------------------------------------------------
Miyazaki et al., Sci. Rep. 11, 13809 (2021), machine-learned kappa_L for 143 half-Heuslers -- a
dataset we already hold -- and found that MORE DESCRIPTORS MADE IT WORSE:

    55 descriptors  ->  ~8% deviation, explicitly attributed to over-fitting
     4 descriptors  ->  ~4% deviation, R2 = 0.84

Their surviving four, by permutation importance:
    1. the lattice parameter
    2. (r1+r2+r3)/3 - r1     mean atomic radius minus the 4c-site radius
    3. (m1+m2+m3)/3 - m1     mean atomic mass  minus the 4c-site mass
    4. m1+m2+m3              sum of atomic masses

Note what ranks 2nd and 3rd: not averages but MEAN-MINUS-SITE DIFFERENCES -- one atom being the odd
one out is what scatters phonons. Those are built here for all three sites, since which site is
"4c" depends on the prototype, and the selection step in s20 decides which matter.

THE LATTICE-CONSTANT TRAP. `a` is simultaneously the strongest descriptor (Spearman rho = -0.604
against log kappa on our own data) and the most dangerous: training rows are conventional-cell,
DXMag prediction targets are primitive, and this CSV carries a `lattice_convention` column
precisely because four sources needed a sqrt(2) correction. So raw `a` is NEVER emitted as a
feature. `volume_per_atom` carries the same physics and is convention-independent -- the same
reasoning behind `_conv_safe_tier2()` in s16.

TWO FINDINGS THAT SHAPED THIS FILE, both measured rather than assumed:

1. `s14_feature_study.tier1_features()` could not site-assign **333 of 952 formulae (35%)** --
   `Ba`, `Sr`, `Mn`, `K` are absent from its element buckets, and `CuAgTe` has two late transition
   metals so no X site exists. Rather than lose a third of the data, site assignment here orders
   the distinct elements by ELECTRONEGATIVITY: most electropositive -> X, most electronegative
   -> Z. That is the chemical logic of a Heusler anyway, and it covers every compound.

2. The Klemens disorder parameters Gamma_mass and Gamma_radius are **identically zero on this
   dataset**. They measure WITHIN-site mixing, and every compound here is stoichiometric -- one
   element per site. They ranked #1 and #2 in this project's earlier study only because that
   dataset contained doped compositions. Emitting them would have added two constant columns and
   corrupted the importance ranking. They are not built.

Everything here is per-row; grouping, weighting and splitting belong to s20.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element


TRAINING_CSV = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
VALIDATION_XLSX = "data/Target_Materials/Validation_Heuslers.xlsx"
DEDUP_LOG = "data/external/BUILD_deduplicated_rows.csv"

# Atoms in the CONVENTIONAL cubic cell, by Heusler stoichiometry. Standard crystallography:
# C1b half-Heusler is 4 formula units of XYZ (12 atoms) with the 4d site vacant; L2_1/XA full
# Heusler is 4 units of X2YZ (16 atoms).
_N_CONV = {(1, 1, 1): 12, (1, 1, 2): 16, (1, 1, 1, 1): 16}


def _stoich_key(formula: str):
    try:
        c = Composition(str(formula)).reduced_composition
    except Exception:  # noqa: BLE001
        return None
    v = [int(round(float(x))) for x in c.values()]
    if not v or any(x <= 0 for x in v):
        return None
    from math import gcd
    g = 0
    for x in v:
        g = gcd(g, x)
    return tuple(sorted(x // g for x in v)) if g else None


def heusler_type(formula: str) -> str:
    """half (XYZ) / FULL (X2YZ) / quat (XX'YZ) / binary-or-other, from the reduced ratio."""
    k = _stoich_key(formula)
    return {(1, 1, 1): "half", (1, 1, 2): "FULL", (1, 1, 1, 1): "quat"}.get(k, "other")


def _bond_stats(species_s, coords_s, a_A) -> dict:
    """Nearest-neighbour statistics from the published fractional coordinates.

    90.9% of rows carry `struct_species` and `struct_frac_coords`, and NO featurizer in the repo
    currently reads them -- this is unused signal. Distances are computed with the minimum-image
    convention in the cubic cell, so they need only `a`, and the RATIOS used downstream are
    convention-independent even though `a` itself is not emitted.
    """
    out = {"bond_min": np.nan, "bond_mean": np.nan, "bond_std": np.nan, "bond_ratio": np.nan}
    try:
        coords = coords_s if isinstance(coords_s, list) else ast.literal_eval(str(coords_s))
        if isinstance(coords, str):
            coords = json.loads(coords)
        pts = np.array([[float(x) for x in p] for p in coords], dtype=float)
        if pts.ndim != 2 or len(pts) < 2 or not np.isfinite(a_A) or a_A <= 0:
            return out
        d = pts[:, None, :] - pts[None, :, :]
        d -= np.round(d)                      # minimum image, cubic cell
        dist = np.linalg.norm(d, axis=-1) * float(a_A)
        iu = np.triu_indices(len(pts), k=1)
        dv = dist[iu]
        dv = dv[dv > 1e-6]
        if not len(dv):
            return out
        nn = np.array([np.min(dist[i][dist[i] > 1e-6]) for i in range(len(pts))])
        out["bond_min"] = float(np.min(dv))
        out["bond_mean"] = float(np.mean(nn))
        out["bond_std"] = float(np.std(nn))
        out["bond_ratio"] = float(np.min(dv) / np.mean(nn)) if np.mean(nn) > 0 else np.nan
    except Exception:  # noqa: BLE001
        pass
    return out


_COV_R: dict = {}


def _cov_radius(sym: str) -> float:
    """Covalent radius in angstrom, cached. Falls back to the atomic radius, then to a mean."""
    if sym in _COV_R:
        return _COV_R[sym]
    r = np.nan
    try:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        r = float(CovalentRadius.radius.get(sym, np.nan))
    except Exception:  # noqa: BLE001
        pass
    if not np.isfinite(r):
        try:
            ar = Element(sym).atomic_radius
            r = float(ar) if ar else np.nan
        except Exception:  # noqa: BLE001
            r = np.nan
    _COV_R[sym] = r
    return r


def _anharmonic_stats(species_s, coords_s, a_A) -> dict:
    """Rattling-atom and bonding-heterogeneity descriptors -- the anharmonicity signal the rest of
    the feature set is blind to.

    WHY THESE AND NOT MORE ELASTIC DATA. Every strong published kappa_L model carries at least one of
    {Gruneisen parameter, Debye temperature, elastic moduli}; ours carried none, and we hold those
    quantities for only ~35% of the training pool. These descriptors instead come from the crystal
    structure itself, which 91% of rows have, so they cover nearly the whole pool.

    THE PHYSICS. An atom sitting loose in an oversized cage rattles, and a rattling atom scatters
    phonons hard -- it is the standard explanation for ultralow lattice thermal conductivity. Xia et
    al. (JACS 2022) screened >500 ultralow-kappa materials on exactly this criterion: an atom whose
    distance to its nearest neighbours EXCEEDS the sum of the two covalent radii is under-bonded.
    The ratio (observed bond) / (sum of covalent radii) makes that a continuous number rather than a
    yes/no flag, so a model can use the degree of looseness and not just its presence.

    Bonding heterogeneity is the companion quantity: a lattice where some bonds are tight and others
    slack is anharmonic in a way a uniform lattice is not, and it is the mechanism behind the
    lone-pair and under-bonded-cation families of low-kappa compounds.

    Aimed squarely at the compounds we predict worst: below 2 W/m/K our error is ~150% and we
    over-predict by 1.8x, which for a thermoelectric screen means missing genuinely good materials.

    TESTED AND REJECTED 2026-09-01 -- NOT WIRED INTO THE FEATURE SET. The descriptors carry real,
    non-redundant signal (rattler_mean Spearman -0.52 against log kappa, only 0.27 correlated with
    any existing feature), but on the nested held-out test they moved accuracy from one band to
    another rather than improving it:

        band          45 features      + these      compounds/split
        < 2 W/m/K     70.0%            45.9%        2      <- better, but unresolvable
        2-5           36.9%            35.0%        11
        > 5           39.5%            45.9%        10     <- worse, and better measured
        overall       40.8%            41.4%        23

    The low-kappa gain DID survive out of sample, unlike the magnitude-calibration attempt, so this
    is not the same failure. But it rests on 2 compounds per split with a 5th-95th range of
    [17-211%], while the high-kappa harm rests on 10 and is unambiguous. With only 6 ultralow-kappa
    compounds in the entire experimental blind test, the question cannot be settled with the data
    that exists -- so the better-measured effect decides, and it is the harm.

    Re-test if the low-kappa experimental set ever grows. The code is left intact for that.
    """
    out = {"rattler_max": np.nan, "rattler_mean": np.nan,
           "n_rattlers": np.nan, "bond_hetero": np.nan}
    try:
        sp = species_s if isinstance(species_s, list) else ast.literal_eval(str(species_s))
        co = coords_s if isinstance(coords_s, list) else ast.literal_eval(str(coords_s))
        if isinstance(sp, str):
            sp = json.loads(sp)
        if isinstance(co, str):
            co = json.loads(co)
        syms = [str(s) for s in sp]
        pts = np.array([[float(x) for x in p] for p in co], dtype=float)
        if len(syms) != len(pts) or len(pts) < 2 or not np.isfinite(a_A) or a_A <= 0:
            return out
        d = pts[:, None, :] - pts[None, :, :]
        d -= np.round(d)                                    # minimum image, cubic cell
        dist = np.linalg.norm(d, axis=-1) * float(a_A)
        radii = np.array([_cov_radius(s) for s in syms], dtype=float)
        if not np.isfinite(radii).all():
            return out
        ratios = []
        for i in range(len(pts)):
            others = np.array([j for j in range(len(pts)) if dist[i, j] > 1e-6])
            if not len(others):
                continue
            j = others[int(np.argmin(dist[i, others]))]      # this atom's nearest neighbour
            expected = radii[i] + radii[j]
            if expected > 0:
                ratios.append(dist[i, j] / expected)
        if not ratios:
            return out
        r = np.asarray(ratios, dtype=float)
        out["rattler_max"] = float(np.max(r))               # the loosest atom in the cell
        out["rattler_mean"] = float(np.mean(r))
        out["n_rattlers"] = float(np.sum(r > 1.0))          # the Xia et al. under-bonded criterion
        out["bond_hetero"] = float(np.std(r))               # tight-and-slack mixture
    except Exception:  # noqa: BLE001
        pass
    return out


def _site_assign(formula: str):
    """Order the distinct elements by electronegativity: X (most electropositive) / Y / Z.

    This replaces the bucket-based `_assign_sites()` in s14, which failed on 35% of our formulae
    (Ba, Sr, Mn, K missing from its element sets; CuAgTe having two late transition metals). The
    electronegativity ordering is the chemistry a Heusler actually follows -- X donates, Z is the
    main-group acceptor -- and it resolves every compound in the set.

    Returns [(site, element, amount)] for X, Y, Z. A quaternary's two middle elements are merged
    onto Y, weighted by amount.
    """
    try:
        comp = Composition(str(formula))
    except Exception:  # noqa: BLE001
        return None
    items = [(e, float(comp[e])) for e in comp.elements]
    if len(items) < 3:
        return None
    items.sort(key=lambda t: (t[0].X if t[0].X else 99.0))
    if len(items) == 3:
        return [("X", items[0]), ("Y", items[1]), ("Z", items[2])]
    if len(items) == 4:
        return [("X", items[0]), ("Y", items[1]), ("Y2", items[2]), ("Z", items[3])]
    return None


# ---------------------------------------------------------------------------
# Site disorder: what makes a doped sample usable instead of noise
# ---------------------------------------------------------------------------
def element_system(formula: str) -> str:
    """The set of elements, ignoring amounts. Doping does NOT change it.

    This is the correct grouping unit for cross-validation once doped rows are admitted.
    `Ta1.05Al1Ru1.95` is TaAlRu2 with a 2.5% site swap; to a model they are near-identical
    samples. Grouping by formula would put one in train and the other in test and report a
    number that means nothing. Measured on our own data: 7 of the 24 validation compounds have
    a doped variant in the recoverable pool, so this is a live risk, not a hypothetical one.
    """
    try:
        return "-".join(sorted(str(e) for e in Composition(str(formula)).elements))
    except Exception:  # noqa: BLE001
        return str(formula)


def _choose_template(amts: dict, tot: float):
    """Pick the site template that best fits this composition, and say how well it fits.

    A 3-site full Heusler [2,1,1] and a 3-site half Heusler [1,1,1] are indistinguishable by the
    LENGTH of the site list, so the template must be returned explicitly rather than re-guessed.
    Guessing it a second time is what made `ZrNiSn` report stoich_deviation = 1.33 -- a pristine
    half Heusler measured against a full-Heusler template.

    Returns (caps, scale, deviation).
    """
    best = None
    for caps in ([2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]):
        n_at = sum(caps)
        scale = n_at / tot
        vals = sorted((a * scale for a in amts.values()), reverse=True)
        padded = vals[:len(caps)] + [0.0] * max(0, len(caps) - len(vals))
        dev = sum(abs(p - c) for p, c in zip(padded, sorted(caps, reverse=True)))
        if best is None or dev < best[0]:
            best = (dev, caps, scale)
    dev, caps, scale = best
    return caps, scale, dev


def _site_occupancies(formula: str):
    """Split a (possibly doped) composition onto its crystallographic sites.

    THE TEMPLATE MUST MATCH THE STOICHIOMETRY. A first version assumed 2:1:1 for every compound
    and therefore forced `ZrNiSn` -- a pristine 1:1:1 half Heusler -- into a full-Heusler template,
    reporting gamma_mass = 0.0396 for a compound with no disorder whatsoever. Since half Heuslers
    are ~504 of the training compounds, that would have injected fabricated disorder into most of
    the dataset. So the template is CHOSEN by best fit, not assumed:

        [2,1,1]     full / inverse Heusler  X2YZ   (4 atoms per formula unit)
        [1,1,1]     half Heusler            XYZ    (3 atoms)
        [1,1,1,1]   quaternary              XX'YZ  (4 atoms)

    Elements are then poured into the sites in order of decreasing amount, overflowing when a site
    fills. On our own compounds this reproduces the chemistry:

        Fe2VAl              -> X{Fe:2}  Y{V:1}  Z{Al:1}             (no disorder)
        Al0.9V1Fe2Si0.1     -> X{Fe:2}  Y{V:1}  Z{Al:.9, Si:.1}     (Z substituted)
        Al0.85V1.15Fe2      -> X{Fe:2}  Y{V:1}  Z{V:.15, Al:.85}    (V onto the Al site)
        ZrNiSn              -> X{Zr:1}  Y{Ni:1} Z{Sn:1}             (no disorder)

    Returns [{element: fractional occupancy}] per site, or None if nothing sensible fits.
    """
    try:
        comp = Composition(str(formula))
    except Exception:  # noqa: BLE001
        return None
    amts = comp.get_el_amt_dict()
    tot = sum(amts.values())
    if tot <= 0 or not (3 <= len(amts) <= 5):
        return None

    caps, scale, _ = _choose_template(amts, tot)

    items = sorted(((e, a * scale) for e, a in amts.items()), key=lambda t: -t[1])
    sites: list[dict[str, float]] = [{} for _ in caps]
    si = 0
    for el, amt in items:
        while amt > 1e-9 and si < len(caps):
            room = caps[si] - sum(sites[si].values())
            if room <= 1e-9:
                si += 1
                continue
            take = min(room, amt)
            sites[si][el] = sites[si].get(el, 0.0) + take
            amt -= take
        if si >= len(caps):
            break
    return [{e: f / caps[i] for e, f in sites[i].items()} for i in range(len(caps))]


def klemens_gamma(formula: str) -> dict:
    """Klemens/Abeles point-defect disorder parameters for a substituted compound.

    THE POINT OF THIS FEATURE. Substituting an atom onto a site puts a mass and a size mismatch
    where the lattice expects uniformity, and phonons scatter off that mismatch -- which is why a
    doped sample conducts less heat than its parent. Without a descriptor for it, a model sees
    `Fe2VAl` at 15 W/m/K and `Al0.9V1Fe2Si0.1` at 6 W/m/K with nearly identical features and can
    only conclude the property is noise. With it, the model can learn "parent value, reduced by
    this much disorder", and setting gamma to zero recovers the pure compound.

    Per site with fractional occupancies f_i:  Gamma = SUM f_i ((Q_i - Qbar)/Qbar)^2
    for Q = mass and Q = radius, combined across sites with the Abeles mass weighting.

    **Every one of these is exactly 0.0 for a stoichiometric compound**, so adding the feature
    cannot disturb the existing pure-only model -- which is what arm 1 of the experiment checks.
    """
    out = {"gamma_mass": 0.0, "gamma_radius": 0.0, "n_sub_sites": 0.0,
           "max_site_disorder": 0.0, "stoich_deviation": 0.0}
    sites = _site_occupancies(formula)
    if sites is None:
        return out
    try:
        amts = Composition(str(formula)).get_el_amt_dict()
        caps, _, dev = _choose_template(amts, sum(amts.values()))
        out["stoich_deviation"] = float(dev)
    except Exception:  # noqa: BLE001
        caps = [1.0] * len(sites)

    m_all = 0.0
    for i, occ in enumerate(sites):
        for el, f in occ.items():
            try:
                m_all += f * caps[i] * float(Element(el).atomic_mass)
            except Exception:  # noqa: BLE001
                pass
    m_all = (m_all / sum(caps)) if m_all > 0 else 1.0

    gm = gr = 0.0
    for i, occ in enumerate(sites):
        if len(occ) < 2:
            continue
        out["n_sub_sites"] += 1
        ms, rs, fs = [], [], []
        for el, f in occ.items():
            try:
                e = Element(el)
                ms.append(float(e.atomic_mass))
                r = e.average_ionic_radius or e.atomic_radius
                rs.append(float(r) if r else float("nan"))
                fs.append(float(f))
            except Exception:  # noqa: BLE001
                continue
        if len(fs) < 2:
            continue
        mbar = sum(f * m for f, m in zip(fs, ms))
        g_m = sum(f * ((m - mbar) / mbar) ** 2 for f, m in zip(fs, ms)) if mbar else 0.0
        rr = [r for r in rs if r == r]
        if len(rr) == len(rs) and rs:
            rbar = sum(f * r for f, r in zip(fs, rs))
            g_r = sum(f * ((r - rbar) / rbar) ** 2 for f, r in zip(fs, rs)) if rbar else 0.0
        else:
            g_r = 0.0
        wgt = (caps[i] / sum(caps)) * (mbar / m_all) ** 2
        gm += wgt * g_m
        gr += wgt * g_r
        out["max_site_disorder"] = max(out["max_site_disorder"], g_m)
    out["gamma_mass"] = float(gm)
    out["gamma_radius"] = float(gr)
    return out


def _props(el):
    m = float(el.atomic_mass)
    try:
        r = float(el.average_ionic_radius) if el.average_ionic_radius else float("nan")
    except Exception:  # noqa: BLE001
        r = float("nan")
    if not r or math.isnan(r):
        try:
            r = float(el.atomic_radius) if el.atomic_radius else float("nan")
        except Exception:  # noqa: BLE001
            r = float("nan")
    x = float(el.X) if el.X else float("nan")
    return m, r, x


def _per_formula_features(formula: str, a_A, prototype, species, coords) -> dict | None:
    """Everything that depends only on the compound, not the temperature."""
    sites = _site_assign(formula)
    if sites is None:
        return None
    f: dict[str, float] = {}
    # site-disorder descriptors -- all exactly 0.0 for a stoichiometric compound, so they are
    # inert on the existing pure-only training set and only carry signal once doped rows enter
    f.update(klemens_gamma(formula))
    M, R, XE = {}, {}, {}
    for site, (el, amt) in sites:
        m, r, x = _props(el)
        key = "Y" if site == "Y2" else site
        if key in M:                      # quaternary: average the two middle elements
            M[key] = (M[key] + m) / 2.0
            R[key] = np.nanmean([R[key], r])
            XE[key] = np.nanmean([XE[key], x])
        else:
            M[key], R[key], XE[key] = m, r, x
    for s in "XYZ":
        f[f"M_{s}"], f[f"r_{s}"], f[f"X_{s}"] = M.get(s, np.nan), R.get(s, np.nan), XE.get(s, np.nan)

    m_vals = [M.get(s, np.nan) for s in "XYZ"]
    r_vals = [R.get(s, np.nan) for s in "XYZ"]
    x_vals = [XE.get(s, np.nan) for s in "XYZ"]
    m_mean, r_mean = np.nanmean(m_vals), np.nanmean(r_vals)
    f["M_mean"], f["r_mean"], f["X_mean"] = m_mean, r_mean, np.nanmean(x_vals)

    # Miyazaki descriptors #2 and #3: MEAN MINUS SITE. One atom being the odd one out is what
    # scatters phonons, and in a stoichiometric compound this is the only mismatch that varies.
    for s in "XYZ":
        f[f"dM_mean_{s}"] = m_mean - M.get(s, np.nan)
        f[f"dr_mean_{s}"] = r_mean - R.get(s, np.nan)
    f["M_sum"] = float(np.nansum(m_vals))                    # his descriptor #4
    f["r_sum"] = float(np.nansum(r_vals))
    f["M_spread"] = float(np.nanmax(m_vals) - np.nanmin(m_vals))
    f["r_spread"] = float(np.nanmax(r_vals) - np.nanmin(r_vals))
    f["M_ratio_XZ"] = M.get("X", np.nan) / M["Z"] if M.get("Z") else np.nan
    f["r_ratio_XZ"] = R.get("X", np.nan) / R["Z"] if R.get("Z") else np.nan
    f["dX_XZ"] = abs(XE.get("X", np.nan) - XE.get("Z", np.nan))     # bond ionicity
    f["X_spread"] = float(np.nanmax(x_vals) - np.nanmin(x_vals))

    # ---- convention-independent structure ----------------------------------------------------
    key = _stoich_key(formula)
    n_conv = _N_CONV.get(key)
    a = float(a_A) if pd.notna(a_A) else np.nan
    if np.isfinite(a) and a > 0 and n_conv:
        vol = a ** 3
        f["volume_per_atom"] = vol / n_conv
        try:
            cell_mass = Composition(str(formula)).weight * (n_conv / sum(key))
            f["density_g_cm3"] = cell_mass / vol / 0.6022140857
        except Exception:  # noqa: BLE001
            f["density_g_cm3"] = np.nan
        f["bond_scale"] = (vol / n_conv) ** (1.0 / 3.0)
    else:
        f["volume_per_atom"] = f["density_g_cm3"] = f["bond_scale"] = np.nan

    f["n_atoms_conv"] = float(n_conv) if n_conv else np.nan
    f["is_full_heusler"] = 1.0 if key == (1, 1, 2) else 0.0
    f["is_quaternary"] = 1.0 if key == (1, 1, 1, 1) else 0.0
    f.update(_bond_stats(species, coords, a))
    # _anharmonic_stats() is NOT wired in -- TESTED AND REJECTED 2026-09-01. See its docstring.
    # It is kept, documented, because the negative result is worth preserving and because the
    # descriptors would become usable if the low-kappa test set ever grows beyond 6 compounds.

    # Slack-flavoured geometric term. The Debye and Gruneisen parts exist for only ~37% of
    # compounds and are joined separately in s20 if they earn their place.
    f["slack_geom"] = (f["M_mean"] * f["bond_scale"]
                       if np.isfinite(f.get("bond_scale", np.nan)) else np.nan)
    return f


def build(csv: str = TRAINING_CSV, tier_max: int = 3, tier_min: int = 0,
          drop_formulas: set[str] | None = None, verbose: bool = True,
          tier_max_by_type: dict[str, int] | None = None,
          extra_csv: str | None = None, group_by: str = "formula") -> dict:
    """Return X, y (log10 kappa_L), groups, weights and metadata for modelling.

    `extra_csv` appends recovered rows (the doped set) AFTER the tier filter, so they are not
    silently removed by it -- they carry their own tier and weight.

    `group_by` selects the cross-validation grouping unit. "formula" is the historical default;
    "element_system" is REQUIRED whenever doped rows are present, because a doped variant and its
    parent share a chemistry and must never straddle a train/test split.
    """
    d = pd.read_csv(csv)
    n0 = len(d)
    _t = pd.to_numeric(d.method_tier, errors="coerce")
    d = d.assign(_kind=d.formula.map(heusler_type))
    if tier_max_by_type:
        # PER-TYPE TIER RULE. Excluding the semi-empirical tier was decided on a GLOBAL test
        # (R2 0.886 -> 0.847), but that test was dominated by half-Heuslers, where 504 clean
        # compounds make noisy extra labels pure cost. Full Heuslers have only 37 -- so few that
        # 38%-error labels may still beat having nothing. Bias-variance trades differently when a
        # class is starved, and one verdict should not be applied to both.
        keep = pd.Series(False, index=d.index)
        for k, tmax in tier_max_by_type.items():
            keep |= (d._kind == k) & (_t <= tmax) & (_t >= tier_min)
        default = tier_max_by_type.get("*", tier_max)
        known = set(tier_max_by_type) - {"*"}
        keep |= (~d._kind.isin(known)) & (_t <= default) & (_t >= tier_min)
        d = d[keep]
    else:
        d = d[(_t <= tier_max) & (_t >= tier_min)]
    if extra_csv:
        try:
            ex = pd.read_csv(extra_csv)
            keep_cols = [c for c in d.columns if c in ex.columns]
            ex = ex[keep_cols]
            for c in d.columns:
                if c not in ex.columns:
                    ex[c] = pd.NA
            d = pd.concat([d, ex[d.columns]], ignore_index=True)
            d["_kind"] = d.formula.map(heusler_type)
            if verbose:
                print(f"  + {len(ex)} extra rows from {extra_csv} "
                      f"({ex.formula.nunique()} compounds)")
        except Exception as exc:  # noqa: BLE001
            print(f"  extra_csv could not be read ({exc}) -- continuing without it")
    if drop_formulas:
        d = d[~d.formula.isin(drop_formulas)]
    d = d[(pd.to_numeric(d.kappa_L, errors="coerce") > 0)
          & (pd.to_numeric(d.temperature_K, errors="coerce") > 0)]

    # DEDUPLICATE EXACT (formula, temperature, kappa) TRIPLES.
    # The same published number arrives twice for 240 compounds -- once via the BTE master file and
    # once via paper mining of the same arXiv preprint -- bit-for-bit identical, both at weight 1.0.
    # That is not two measurements: it double-weights those compounds in the MAE loss and counts
    # them twice in every reported median. Grouped CV keeps duplicates in the same fold, so this was
    # never a train/test leak, only a silent weighting bug. The most authoritative tier is kept.
    n_pre = len(d)
    _sorted = (d.assign(_t_sort=pd.to_numeric(d.method_tier, errors="coerce").fillna(9))
               .sort_values("_t_sort", kind="mergesort"))
    _keep = ~_sorted.duplicated(subset=["formula", "temperature_K", "kappa_L"], keep="first")
    _dropped = _sorted[~_keep].drop(columns=["_t_sort"])
    d = _sorted[_keep].drop(columns=["_t_sort"])
    if verbose and len(_dropped):
        print(f"  deduplicated {len(_dropped)} exact (formula, T, kappa) duplicate rows "
              f"({100 * len(_dropped) / n_pre:.2f}% of {n_pre})")
    if len(_dropped):
        # AUDITABLE, like every other exclusion in this pipeline. The dedup key cannot distinguish
        # "the same published number ingested twice" from "two sources that coincidentally agree to
        # full float precision", so the discarded rows are written out with their source and DOI
        # rather than vanishing behind a count. Every duplicate group inspected so far differs in
        # source_doi, and all of them trace to citation or preprint/journal re-ingestion.
        try:
            Path(DEDUP_LOG).parent.mkdir(parents=True, exist_ok=True)
            _dropped.to_csv(DEDUP_LOG, index=False)
            if verbose:
                print(f"    the discarded rows are listed in {DEDUP_LOG}")
        except Exception as exc:  # noqa: BLE001
            print(f"    (could not write the dedup audit trail: {exc})")

    if verbose:
        print(f"  rows {n0} -> {len(d)}   compounds {d.formula.nunique()}"
              f"   (tier {tier_min}-{tier_max}"
              f"{f', {len(drop_formulas)} formulae held out' if drop_formulas else ''})")

    # per-formula features computed once, then broadcast to every temperature row.
    # PREFER A ROW THAT ACTUALLY HAS COORDINATES. `d` is tier-sorted, so a plain
    # drop_duplicates("formula") took the most authoritative row -- but if that row happens to lack
    # struct_species / struct_frac_coords, `_bond_stats()` returns all-NaN and the compound loses
    # four of its forty-five features silently. Sorting on coordinate availability first, tier
    # second, keeps the authoritative choice among rows that can actually be used.
    _u = d[["formula", "struct_a_A", "struct_prototype",
            "struct_species", "struct_frac_coords", "method_tier"]].copy()
    _u["_xyz"] = _u.struct_species.notna() & _u.struct_frac_coords.notna()
    _u["_tt"] = pd.to_numeric(_u.method_tier, errors="coerce").fillna(9)
    uniq = (_u.sort_values(["_xyz", "_tt"], ascending=[False, True], kind="mergesort")
            .drop_duplicates("formula")
            .drop(columns=["_xyz", "_tt", "method_tier"]))
    recs, failed = {}, []
    for r in uniq.itertuples():
        f = _per_formula_features(r.formula, r.struct_a_A, r.struct_prototype,
                                  r.struct_species, r.struct_frac_coords)
        if f is None:
            failed.append(r.formula)
        else:
            recs[r.formula] = f
    if verbose and failed:
        print(f"  {len(failed)} formulae could not be site-assigned (dropped): {failed[:6]}")
    d = d[d.formula.isin(recs)]
    F = pd.DataFrame.from_dict(recs, orient="index")
    X = d[["formula"]].join(F, on="formula").drop(columns=["formula"])

    # temperature: both forms, because Umklapp gives kappa ~ 1/T and our own data measures the
    # exponent at -0.92 (median over 143 compounds with a real curve)
    T = pd.to_numeric(d.temperature_K, errors="coerce").values
    X = X.reset_index(drop=True)
    X["temperature_K"] = T
    X["inv_T"] = 1000.0 / T
    X["log_T"] = np.log10(T)

    y = np.log10(pd.to_numeric(d.kappa_L, errors="coerce").values)
    if group_by == "element_system":
        groups = d.formula.map(element_system).values
    else:
        groups = d.formula.values
    w = pd.to_numeric(d.weight, errors="coerce").fillna(1.0).values
    tier = pd.to_numeric(d.method_tier, errors="coerce").fillna(3).astype(int).values
    htype = d._kind.values if "_kind" in d.columns else np.array(["?"] * len(d))

    if verbose:
        import collections as _c
        by = _c.Counter(d._kind.map(str))
        cmpd = {k: d[d._kind == k].formula.nunique() for k in by}
        ndop = int((pd.to_numeric(X.get("stoich_deviation", 0), errors="coerce")
                    .fillna(0) > 1e-6).sum())
        print(f"  features {X.shape[1]}   rows {X.shape[0]}   "
              f"groups({group_by}) {len(set(groups))}   by type: {cmpd}")
        if ndop:
            print(f"  doped rows carrying non-zero site disorder: {ndop}")
    return dict(X=X, y=y, groups=groups, weights=w, tier=tier, htype=htype,
                meta=d.reset_index(drop=True), feature_names=list(X.columns))


def validation_formulas(path: str = VALIDATION_XLSX) -> set[str]:
    """The 44 experimentally-measured Heuslers. 40 of them also sit in the training set, so they
    must be removed from training BEFORE any transfer test -- otherwise the model is being asked
    about compounds whose answers it has already seen."""
    p = Path(path)
    if not p.exists():
        return set()
    v = pd.read_excel(p, sheet_name="Compounds")
    out = set()
    for f in v.formula.dropna():
        try:
            out.add(Composition(str(f)).reduced_formula)
        except Exception:  # noqa: BLE001
            pass
    return out


if __name__ == "__main__":
    print("TRAINING SET")
    b = build()
    print(f"\nfeature list ({len(b['feature_names'])}):")
    for i in range(0, len(b["feature_names"]), 4):
        print("   " + "  ".join(f"{n:<20}" for n in b["feature_names"][i:i + 4]))
    val = validation_formulas()
    print(f"\nvalidation formulae: {len(val)}")
    print(f"  of which present in training: {len(val & set(b['groups']))}")
