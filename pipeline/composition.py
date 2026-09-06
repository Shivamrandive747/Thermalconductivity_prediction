"""Composition parsing: formula string -> element fractions + reduced formula.

Used for dedup, cross-DB / structure matching, and ML featurization. Prefers
pymatgen (handles nested groups, fractional subscripts, hydrates); falls back to
a lightweight regex parser if pymatgen isn't installed yet.
"""
from __future__ import annotations

import re
from typing import Optional

try:
    from pymatgen.core import Composition as _PmgComposition  # type: ignore
    _HAVE_PMG = True
except Exception:  # noqa: BLE001
    _HAVE_PMG = False

# element symbol regex (two-letter then one-letter); ordered longest-first below
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_GROUP = re.compile(r"\(([^()]*)\)(\d*\.?\d*)")


def _expand_groups(formula: str) -> str:
    """Expand one level of parenthesized groups, e.g. Ti(Co0.9Ni0.1)Sb."""
    prev = None
    f = formula
    # iterate until no parentheses remain (handles a couple nesting levels)
    while prev != f and "(" in f:
        prev = f
        def repl(m: re.Match) -> str:
            inner, mult = m.group(1), m.group(2)
            k = float(mult) if mult else 1.0
            out = []
            for sym, num in _TOKEN.findall(inner):
                if not sym:
                    continue
                n = float(num) if num else 1.0
                out.append(f"{sym}{_fmt(n * k)}")
            return "".join(out)
        f = _GROUP.sub(repl, f)
    return f


def _fmt(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}"


# subscript arithmetic (1-x, 0.8+x, 2-y) or a series definition (x = 0.1);
# deliberately narrow so Dy (dysprosium) etc. never false-positive
_VARIABLE = re.compile(r"[-+]\s*[xyz](?![a-z])|\b[xyz]\s*=")


def has_unresolved_variable(formula: str) -> bool:
    """True for series formulas with a literal variable (Nb1-xCoSb, V0.8+xCoSb):
    they CANNOT be parsed — the regex fallback would silently drop the x term and
    produce a wrong formula, so callers must flag them instead."""
    return bool(formula) and bool(_VARIABLE.search(formula.strip()))


def parse_fractions(formula: str) -> dict[str, float]:
    """Return {element: atomic_fraction} summing to 1.0, or {} if unparseable."""
    if not formula:
        return {}
    clean = formula.strip().replace(" ", "")
    # strip charge/phase annotations like '(cubic)' handled by group expansion;
    # remove trailing descriptors after a comma/space already stripped.
    if _HAVE_PMG:
        try:
            comp = _PmgComposition(clean)
            frac = comp.fractional_composition.get_el_amt_dict()
            return {el: float(v) for el, v in frac.items()}
        except Exception:  # noqa: BLE001 — fall through to regex
            pass
    # regex fallback
    expanded = _expand_groups(clean)
    counts: dict[str, float] = {}
    for sym, num in _TOKEN.findall(expanded):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {el: v / total for el, v in counts.items()}


def reduced_formula(formula: str) -> Optional[str]:
    """Canonical reduced formula for dedup/matching, e.g. 'Fe2VAl'. None if bad."""
    if not formula:
        return None
    if _HAVE_PMG:
        try:
            return _PmgComposition(formula.strip().replace(" ", "")).reduced_formula
        except Exception:  # noqa: BLE001
            pass
    # fallback: normalize by GCD of integer counts (only when all near-integer)
    expanded = _expand_groups(formula.strip().replace(" ", ""))
    counts: dict[str, float] = {}
    for sym, num in _TOKEN.findall(expanded):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return None
    # emit a stable string (sorted) — not a true Hill/reduced formula, but
    # consistent for matching within our own data when pymatgen is absent.
    return "".join(f"{el}{_fmt(counts[el])}" for el in sorted(counts))


def _amounts(formula: str) -> dict[str, float]:
    """Return {element: absolute amount} (NOT normalized to 1), or {} if bad."""
    if not formula:
        return {}
    clean = formula.strip().replace(" ", "")
    if _HAVE_PMG:
        try:
            return {el: float(v) for el, v in _PmgComposition(clean).get_el_amt_dict().items()}
        except Exception:  # noqa: BLE001
            pass
    expanded = _expand_groups(clean)
    counts: dict[str, float] = {}
    for sym, num in _TOKEN.findall(expanded):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    return counts


def _formula_from_counts(counts: dict[str, float]) -> str:
    return "".join(f"{e}{_fmt(counts[e])}" for e in sorted(counts) if counts[e] > 0)


def _group_sites(frac_els: dict[str, float]) -> list[dict[str, float]]:
    """Greedily group co-doped elements into crystallographic sites whose
    occupancies sum to ~1 (Heusler sites are multiplicity 1 per formula unit)."""
    items = sorted(frac_els.items(), key=lambda x: -x[1])
    sites, cur, s = [], {}, 0.0
    for e, a in items:
        cur[e] = a
        s += a
        if abs(s - round(s)) < 0.15 and round(s) >= 1:
            sites.append(dict(cur))
            cur, s = {}, 0.0
    if cur:                      # leftover occupants -> their own site
        sites.append(dict(cur))
    return sites


# --- Heusler site chemistry (X-early / Y-late / Z-main buckets) --------------
# Standard site assignment: half-Heusler XYZ = (early TM / RE)(late TM)(main group);
# full Heusler X2YZ has the LATE TM on the multiplicity-2 site (Co2MnSi) or,
# less often, Mn (Mn2VAl) — Mn is chemically ambidextrous, so it is tried in both.
_EARLY_TM = {
    "Li", "Mg", "Ca", "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ti", "Zr", "Hf", "V", "Nb",
    "Ta", "Cr", "Mo", "W", "Th", "U",
}
_LATE_TM = {
    "Fe", "Co", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Re", "Os", "Ir", "Pt", "Au",
}
_MAIN = {
    "B", "Al", "Ga", "In", "Tl", "Si", "Ge", "Sn", "Pb",
    "P", "As", "Sb", "Bi", "S", "Se", "Te",
}


def _bucket_parents(fr: dict[str, float], max_candidates: int) -> list[str]:
    """Parent prototypes via Heusler site chemistry, for compositions the generic
    doped-site logic can't reduce: atomic-% formulas (Mn25Al22Ni53 -> Ni2MnAl) and
    equiatomic shared-site alloys (HfZrTiNiSn -> HfNiSn+ZrNiSn+TiNiSn)."""
    from itertools import product

    total = sum(fr.values())
    if total <= 0:
        return []
    frac = {e: a / total for e, a in fr.items()}

    mn_options = ("early", "late") if "Mn" in frac else ("early",)
    out: list[str] = []
    for mn_as in mn_options:
        buckets: dict[str, dict[str, float]] = {"E": {}, "L": {}, "M": {}}
        ok = True
        for e, a in frac.items():
            if e == "Mn":
                buckets["E" if mn_as == "early" else "L"][e] = a
            elif e in _EARLY_TM:
                buckets["E"][e] = a
            elif e in _LATE_TM:
                buckets["L"][e] = a
            elif e in _MAIN:
                buckets["M"][e] = a
            else:
                ok = False       # element outside Heusler chemistry -> no parent
        if not ok or not all(buckets.values()):
            continue             # needs one of each site to be a Heusler

        # Each bucket is either 'literal' (sum as written) or 'shared' (n equal
        # occupants sharing ONE site, e.g. Hf1Zr1Ti1 on the X site -> sum/n).
        def _hypotheses(b: dict[str, float]) -> list[float]:
            s = sum(b.values())
            hyp = [s]
            if len(b) > 1:
                vals = sorted(b.values())
                if vals[-1] / max(vals[0], 1e-12) < 1.3:   # near-equal occupants
                    hyp.append(s / len(b))
            return hyp

        # site-fraction patterns: half-Heusler XYZ = (1,1,1)/3; full X2YZ = (1,2,1)/4
        patterns = {"half": (1 / 3, 1 / 3, 1 / 3), "full": (0.25, 0.5, 0.25)}
        for eh, lh, mh in product(_hypotheses(buckets["E"]), _hypotheses(buckets["L"]),
                                  _hypotheses(buckets["M"])):
            s = eh + lh + mh
            if s <= 0:
                continue
            e_f, l_f, m_f = eh / s, lh / s, mh / s
            for kind, (pe, pl, pm) in patterns.items():
                # 0.09 site-fraction tolerance (~0.36 atom/f.u.): admits the
                # deliberately off-stoichiometric Ni-Mn-Al/Ga SMA series while
                # keeping half- vs full-Heusler patterns (0.33 vs 0.25/0.50) apart
                if max(abs(e_f - pe), abs(l_f - pl), abs(m_f - pm)) > 0.09:
                    continue
                # occupant choices per site, richest first
                es = sorted(buckets["E"], key=lambda x: -buckets["E"][x])
                ls = sorted(buckets["L"], key=lambda x: -buckets["L"][x])
                ms = sorted(buckets["M"], key=lambda x: -buckets["M"][x])
                for xe, xl, xm in product(es, ls, ms):
                    counts = ({xe: 1, xl: 1, xm: 1} if kind == "half"
                              else {xl: 2, xe: 1, xm: 1})
                    rf = reduced_formula(_formula_from_counts(counts))
                    if rf and rf not in out:
                        out.append(rf)
                if out:
                    break        # first matching pattern for this hypothesis wins
            if out:
                break
        if out:
            break
    return out[:max_candidates]


def parent_prototypes(formula: str, max_candidates: int = 6) -> list[str]:
    """Integer-stoichiometry parent prototype formula(s) for a (possibly doped)
    Heusler, for matching against structure databases that only hold end-members.

    Examples: 'Nb0.8Ti0.2FeSb' -> ['NbFeSb','TiFeSb'];
              'Hf0.5Zr0.5NiSn' -> ['HfNiSn','ZrNiSn']; 'Fe2VAl' -> ['Fe2VAl'];
              'Mn25Al22Ni53' (atomic-%) -> ['Ni2MnAl'];
              'HfZrTiNiSn' (shared X site) -> ['HfNiSn','ZrNiSn','TiNiSn'].
    A dopant occupant fully takes its site; occupants < 0.15 are treated as trace
    and dropped. Returns [] if it can't be reduced to a clean prototype.
    """
    fr = _amounts(formula)
    if not fr:
        return []
    major = {e: round(a) for e, a in fr.items() if a >= 0.85}      # fixed / majority sites
    frac = {e: a for e, a in fr.items() if 0.15 <= a < 0.85}       # doped sites
    # (elements with a < 0.15 are trace dopants -> ignored for the prototype)
    sites = _group_sites(frac)
    from itertools import product
    per_site = [list(site.keys()) for site in sites]               # non-empty lists
    combos = list(product(*per_site))[:max_candidates] if per_site else [()]
    candidates: set[str] = set()
    for combo in combos:
        comp = dict(major)
        for e in combo:
            comp[e] = comp.get(e, 0) + 1
        rf = reduced_formula(_formula_from_counts(comp))
        if rf:
            candidates.add(rf)

    # Was the generic path useful? Useful = it produced at least one clean,
    # small-integer prototype (a real end-member a database could hold). If not
    # (atomic-% formulas, equiatomic shared-site alloys, junk like Mn10Al11Ni29),
    # fall back to Heusler site chemistry.
    def _clean_proto(rf_: str) -> bool:
        am = _amounts(rf_)
        return (bool(am) and all(abs(a - round(a)) < 0.02 for a in am.values())
                and sum(round(a) for a in am.values()) <= 4)

    if any(_clean_proto(c) for c in candidates):
        return list(candidates)
    bucket = _bucket_parents(fr, max_candidates)
    return bucket if bucket else list(candidates)


def backend_name() -> str:
    return "pymatgen" if _HAVE_PMG else "regex-fallback"
