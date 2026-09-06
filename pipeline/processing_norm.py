"""Processing standardization: map free-text synthesis/processing descriptions to
controlled vocabularies so preparation becomes a clean categorical feature block
for the attention model (the "processing" leg of processing-structure-property).

The LLM extracts processing as free text (robust); these mappers canonicalize the
categorical fields. Numeric fields (temperatures→K, times→h/min, pressure→MPa,
grain→µm) are already canonical from the schema. Raw text is preserved in the DB;
standardization is applied at export time (s10), so nothing is lost.
"""
from __future__ import annotations

from typing import Optional

# keyword -> canonical term (first match wins; order longest/most-specific first)
_SYNTHESIS = [
    ("arc_melt", ("arc melt", "arc-melt")),
    ("induction_melt", ("induction melt", "induction-melt", "induction")),
    ("levitation_melt", ("levitation", "levitated")),
    ("melt_spin", ("melt spin", "melt-spin", "melt spun", "rapidly quench", "rapid quench",
                   "single roller", "melt quench")),
    ("mechanical_alloy", ("mechanical alloy", "mechanically alloy", "ball mill", "ball-mill",
                          "high energy mill", "high-energy mill", "planetary mill")),
    ("solid_state", ("solid state", "solid-state")),
    ("shs", ("self-propagating", "self propagating", "combustion synth", "shs")),
    ("bridgman", ("bridgman",)),
    ("czochralski", ("czochralski", "float zone", "floating zone")),
    ("sputter", ("sputter", "magnetron")),
    ("pld", ("pulsed laser", "pld")),
    ("mbe", ("molecular beam", "mbe")),
    ("sol_gel", ("sol-gel", "sol gel")),
]
_CONSOLIDATION = [
    ("sps", ("spark plasma", "sps", "pecs", "field assisted", "field-assisted", "plasma activated")),
    ("hip", ("hot isostatic", "hip")),
    ("hot_press", ("hot press", "hot-press", "hot pressing", "uniaxial press")),
    ("cold_press_sinter", ("cold press", "cold-press", "pressureless", "conventional sinter",
                           "cold pressing")),
]
_ATMOSPHERE = [
    ("forming_gas", ("forming gas", "ar-h2", "ar/h2", "h2/ar", "h2-ar", "reducing")),
    ("argon", ("argon", "ar atmos", " ar ", "under ar", "in ar")),
    ("vacuum", ("vacuum", "evacuat", "sealed quartz", "sealed silica")),
    ("nitrogen", ("nitrogen", "n2")),
    ("air", ("air", "ambient", "oxygen", "o2")),
]
_COOLING = [
    ("water_quench", ("water quench", "water-quench", "ice water", "brine")),
    ("quench", ("quench", "rapidly cool")),
    ("furnace_cool", ("furnace cool", "furnace-cool", "cooled in the furnace", "cooled in furnace")),
    ("slow_cool", ("slow cool", "slow-cool", "slowly cool", "gradually cool", "controlled cool")),
]


def _match(text: Optional[str], table) -> Optional[str]:
    if not text:
        return None
    t = f" {text.lower()} "
    for canon, keys in table:
        if any(k in t for k in keys):
            return canon
    return "other"                     # non-empty but unrecognized -> flagged as 'other'


def canon_synthesis_route(text): return _match(text, _SYNTHESIS)
def canon_consolidation(text):   return _match(text, _CONSOLIDATION)
def canon_atmosphere(text):      return _match(text, _ATMOSPHERE)
def canon_cooling(text):         return _match(text, _COOLING)


# --- R12: order-state canonicalization --------------------------------------
# Heusler site-ordering free text -> controlled vocab so ordering is a trainable
# categorical, not prose. Order matters: disorder statements beat the ordered
# prototype ("B2 (antisite disorder)" must map to b2, not l21).
_ORDER = [
    ("b2", ("b2",)),
    ("a2", ("a2",)),
    ("do3", ("do3", "d03", "do_3")),
    ("xa", ("xa", "inverse heusler", "hg2cuti")),
    ("l21", ("l2_1", "l21", "l2 1", "fully order", "fully-order", "chemically order",
             "well order", "highly order")),
    ("c1b", ("c1_b", "c1b", "half-heusler structure", "half heusler structure", "mgagas")),
    ("partial_disorder", ("partial", "some disorder", "antisite", "site disorder",
                          "occupation of", "intermixing", "site exchange")),
    ("disordered", ("disorder",)),
    ("ordered_unspecified", ("order",)),
]

_ORDER_PARAM = None  # compiled lazily


def canon_order_state(text: Optional[str]) -> Optional[str]:
    return _match(text, _ORDER)


def order_parameter(text: Optional[str]) -> Optional[float]:
    """Numeric long-range order parameter S when stated (e.g. 'S = 0.92')."""
    global _ORDER_PARAM
    if not text:
        return None
    import re
    if _ORDER_PARAM is None:
        _ORDER_PARAM = re.compile(r"\bS\s*[=~≈]\s*(1(?:\.0+)?|0?\.\d+)")
    m = _ORDER_PARAM.search(text)
    return float(m.group(1)) if m else None
