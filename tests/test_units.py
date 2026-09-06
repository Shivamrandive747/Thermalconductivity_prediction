"""Unit-normalization regression tests (run: python tests/test_units.py).

Guards the s06 canonicalizer against the real-world unit variants we saw in the
extracted data: 10^n axis-scale prefixes, K-first orderings, K^-2 power-factor
units, µΩ·m resistivity, and dimensionless-ZT spellings. Plain asserts so it runs
without pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.s06_validate_export import _canonicalize  # noqa: E402

# (property, value, unit, expected_canonical_value or None) — None => must convert,
# value not asserted; use a number to assert the converted value (abs tol 1e-6 rel).
CASES = [
    # electrical conductivity -> S/cm, incl. 10^n prefixes and cm^-1 ordering
    ("electrical_conductivity", 2.0, "10^4 S m^-1", 200.0),
    ("electrical_conductivity", 200.0, "S cm^-1", 200.0),
    ("electrical_conductivity", 1000.0, "S/m", 10.0),
    # resistivity -> ohm.cm, incl. µΩ·m and 10^-3 prefix
    ("electrical_resistivity", 5.0, "10^-3 Ohm cm", 0.005),
    ("electrical_resistivity", 45.0, "µΩ m", 4.5e-3),
    # thermal conductivity -> W/m/K, incl. K-first order variants
    ("thermal_conductivity_total", 3.5, "W/mK", 3.5),
    ("thermal_conductivity_lattice", 3.5, "W K^-1 m^-1", 3.5),
    ("thermal_conductivity_total", 3.5, "W/Km", 3.5),
    # seebeck
    ("seebeck_coefficient", 150.0, "µV/K", 150.0),
    # power factor -> µW/cm/K², incl. orderings + prefixes
    ("power_factor", 15.0, "µW cm⁻¹ K⁻²", 15.0),
    ("power_factor", 2.0, "mWm^-1K^-2", 20.0),
    ("power_factor", 30.0, "μW K-2cm-1", 30.0),
    # carrier concentration -> cm-3, 10^20 prefix
    ("carrier_concentration", 2.0, "10^20 cm^-3", 2e20),
    # zt dimensionless spellings
    ("zt", 1.2, "null", 1.2),
    ("zt", 1.2, "dimensionless", 1.2),
    ("zt", 1.2, "", 1.2),
    # µΩ·m unconvertible negative control removed — these MUST convert
    # --- separator variants: '-' and '.' used as unit separators -------------
    # 446 experimental kappa rows were stranded on exactly these spellings.
    ("thermal_conductivity_total", 3.5, "W/m-K", 3.5),
    ("thermal_conductivity_total", 3.5, "W/m.K", 3.5),
    ("thermal_conductivity_total", 3.5, "W.m-1.K-1", 3.5),
    ("thermal_conductivity_total", 3.5, "W/(m.K)", 3.5),
    ("thermal_conductivity_lattice", 3.5, "W/(m*K)", 3.5),
    # --- cm-based thermal conductivity (no factors existed at all) ----------
    # mW/cm/K = 1e-3 W / 1e-2 m / K = 0.1 W/m/K ; W/cm/K = 100 W/m/K
    ("thermal_conductivity_total", 35.0, "mW/cmK", 3.5),
    ("thermal_conductivity_total", 35.0, "mW/cm-K", 3.5),
    ("thermal_conductivity_total", 35.0, "mW/(cm*K)", 3.5),
    ("thermal_conductivity_total", 0.035, "W/cmK", 3.5),
]

# Units that MUST NOT convert. These are BoltzTraP kappa_e/tau values (~1e14
# W/m/K/s) — a different physical quantity, correctly tagged data_origin=
# 'theoretical' by extraction. If the separator fix above ever normalises the
# trailing per-second away, ~104 DFT rows would silently enter the training set
# as experimental kappa. This list is the guard.
MUST_NOT_CONVERT = [
    ("thermal_conductivity_total", 2.5e14, "W/(m*K*s)"),
    ("thermal_conductivity_electronic", 2.5e14, "W m-1 K-1 s-1"),
    ("thermal_conductivity_total", 2.5e14, "W/mKs"),
    ("thermal_conductivity_electronic", 2.5e14, "W/(K*m*s)"),
    ("thermal_conductivity_electronic", 2.5e14, "W/(K.m.s)"),
    ("thermal_conductivity_electronic", 2.5e14, "W/(m.K.s)"),
]


def main() -> int:
    failures = []
    for prop, val, unit, expected in CASES:
        cval, cunit, flag = _canonicalize(prop, val, unit)
        if cval is None or flag == "unconvertible":
            failures.append(f"{prop} [{unit}] -> UNCONVERTIBLE (expected convertible)")
            continue
        if expected is not None:
            tol = max(1e-6, abs(expected) * 1e-6)
            if abs(cval - expected) > tol:
                failures.append(f"{prop} [{unit}] -> {cval} (expected {expected})")

    for prop, val, unit in MUST_NOT_CONVERT:
        cval, cunit, flag = _canonicalize(prop, val, unit)
        if cval is not None and flag != "unconvertible":
            failures.append(
                f"{prop} [{unit}] -> {cval} but MUST stay unconvertible "
                "(kappa_e/tau is not thermal conductivity)")

    if failures:
        print("UNIT TEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"All {len(CASES)} unit cases + {len(MUST_NOT_CONVERT)} negative controls passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
