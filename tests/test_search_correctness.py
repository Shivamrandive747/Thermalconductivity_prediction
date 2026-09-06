"""Offline correctness tests for the compound search. No network calls.

Each test corresponds to a defect found in the bug review, so a regression re-fails the same way
it originally failed.
"""
import sys

sys.path.insert(0, ".")
from search_compounds_fixed import (classify, extract_formulas, match_kind, norm,
                                    parent_formula, reduced)

FAILS = []


def check(name, got, want):
    ok = got == want
    if not ok:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58} {got!r}")


print("\n[bug A] norm() must not eat the letters 'sub' from ordinary words")
check("substitution survives whole", norm("substitution of Fe"), "substitutionoffe")
check("<sub> tags still removed", norm("Co<sub>2</sub>TiSn"), "co2tisn")
check("unicode subscripts folded", norm("Co\u2082TiSn"), "co2tisn")

print("\n[bug B] formula matching must reject look-alike compounds")
check("CaAgP vs a CaAgPb paper",
      match_kind("we studied CaAgPb thin films thermal", "CaAgP"), None)
check("ZrNiSn vs a ZrNiSn2 paper",
      match_kind("ZrNiSn2 shows thermal conductivity", "ZrNiSn"), None)
check("Fe2VAl vs a doped Fe2VAl0.9Si0.1 paper",
      match_kind("Fe2VAl0.9Si0.1 thermal conductivity", "Fe2VAl"), None)
print("        ...while still matching the real thing, in any element ordering")
check("Fe2VAl matches Fe2VAl", match_kind("Fe2VAl thermal conductivity", "Fe2VAl"), "exact")
check("Fe2VAl matches AlVFe2", match_kind("AlVFe2 lattice thermal conductivity", "Fe2VAl"),
      "exact")
check("ZrNiSn matches ZrNiSn", match_kind("ZrNiSn half-Heusler kappa", "ZrNiSn"), "exact")
check("Co2TiSn via subscripts", match_kind("Co<sub>2</sub>TiSn transport", "Co2TiSn"), "exact")

print("\n[bug B] formula extraction must not hallucinate compounds from English")
for word in ("In this work we", "Cost of the sample", "Bane of the method", "No significant"):
    check(f"no formula in {word!r}", extract_formulas(word), set())
# extract_formulas returns REDUCED formulas, so compare against the reduced form
check("real formula is found",
      reduced("Fe2VAl") in extract_formulas("The Fe2VAl sample"), True)

print("\n[bug C] every material class we hold must be admitted")
for f, want in [("Fe2VAl", "full_heusler"), ("ZrNiSn", "half_heusler"),
                ("Ti2MnAl", "full_heusler"), ("CoFeYGe", "quaternary_heusler"),
                ("Ag3Sn", "binary_Cu3Au"), ("Cu2Sb", "binary_2to1")]:
    check(f"{f} classified", classify(f)[0], want)
check("doped compound flagged", classify("Fe2VAl0.9Si0.1")[1], True)
check("pure compound not flagged", classify("Fe2VAl")[1], False)

print("\n[scope] a doped compound must be searched by its stoichiometric parent")
check("Al0.85V1.15Fe1.85Cu0.15 -> parent", reduced(parent_formula("Al0.85V1.15Fe1.85Cu0.15")),
      reduced("Fe2VAl"))
check("Ta1.05Al1Ru1.95 -> parent", reduced(parent_formula("Ta1.05Al1Ru1.95")),
      reduced("TaAlRu2"))
check("pure formula unchanged", reduced(parent_formula("Fe2VAl")), reduced("Fe2VAl"))

print("\n" + ("=" * 72))
if FAILS:
    print(f"{len(FAILS)} FAILURES")
    for f in FAILS:
        print("  " + f)
    raise SystemExit(1)
print("ALL TESTS PASSED")
