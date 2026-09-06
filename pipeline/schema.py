"""The core deliverable: the data schema (sample-centric).

A paper reports one or more physical SAMPLES; each sample has a shared context
(composition, phase, ordering, synthesis, density...) and a list of MEASUREMENTS
(one property value each). Grouping by sample lets us link S, sigma, kappa, ZT of
the same sample at matching temperature — required for physical self-consistency
checks (ZT = S^2 sigma T / kappa) and far more useful for ML than isolated rows.

The same Pydantic models drive both extraction backends (Gemini `response_schema`
and Claude `output_config.format`). extra="forbid" is intentionally NOT set — it
emits additionalProperties:false, which Gemini rejects; the Anthropic backend
injects it itself. Extra keys are ignored, which is safe.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class HeuslerType(str, Enum):
    full = "full"                # X2YZ, L2_1
    half = "half"                # XYZ, C1_b
    inverse = "inverse"          # inverse full-Heusler, XA
    quaternary = "quaternary"    # XX'YZ, equiatomic quaternary
    unknown = "unknown"


class DataOrigin(str, Enum):
    experimental = "experimental"  # measured on a real sample (the priority)
    theoretical = "theoretical"    # computed (DFT / first-principles / model)
    unknown = "unknown"


class ExtractionSource(str, Enum):
    text = "text"        # value stated in running text
    table = "table"      # value read from a table
    figure = "figure"    # value read off a plot/curve (approximate; flag it)
    unknown = "unknown"


class PropertyName(str, Enum):
    # --- Thermoelectric ---
    thermal_conductivity_total = "thermal_conductivity_total"
    thermal_conductivity_lattice = "thermal_conductivity_lattice"
    thermal_conductivity_electronic = "thermal_conductivity_electronic"
    thermal_diffusivity = "thermal_diffusivity"        # LFA primary measurement
    seebeck_coefficient = "seebeck_coefficient"
    electrical_conductivity = "electrical_conductivity"
    electrical_resistivity = "electrical_resistivity"
    zt = "zt"                                  # figure of merit
    power_factor = "power_factor"
    carrier_concentration = "carrier_concentration"
    carrier_mobility = "carrier_mobility"
    hall_coefficient = "hall_coefficient"
    # --- Magnetic ---
    curie_temperature = "curie_temperature"
    neel_temperature = "neel_temperature"
    saturation_magnetization = "saturation_magnetization"
    remanent_magnetization = "remanent_magnetization"
    magnetic_moment = "magnetic_moment"        # per formula unit
    coercivity = "coercivity"
    spin_polarization = "spin_polarization"
    magnetoresistance = "magnetoresistance"
    magnetocaloric_entropy_change = "magnetocaloric_entropy_change"
    exchange_bias_field = "exchange_bias_field"
    # --- Mechanical / elastic ---
    vickers_hardness = "vickers_hardness"
    youngs_modulus = "youngs_modulus"
    bulk_modulus = "bulk_modulus"
    shear_modulus = "shear_modulus"
    poisson_ratio = "poisson_ratio"
    fracture_toughness = "fracture_toughness"
    yield_strength = "yield_strength"
    debye_temperature = "debye_temperature"
    # --- Electronic structure (measurable or DFT) ---
    band_gap = "band_gap"
    half_metallic_gap = "half_metallic_gap"
    dos_at_fermi = "dos_at_fermi"
    effective_mass = "effective_mass"
    work_function = "work_function"
    defect_formation_energy = "defect_formation_energy"
    # --- Structural / thermal ---
    lattice_parameter = "lattice_parameter"
    density = "density"
    thermal_expansion_coeff = "thermal_expansion_coeff"
    specific_heat = "specific_heat"
    sommerfeld_coefficient = "sommerfeld_coefficient"  # electronic C_p gamma
    sound_velocity = "sound_velocity"
    melting_temperature = "melting_temperature"
    superconducting_tc = "superconducting_tc"
    # --- Martensitic / shape-memory (Ni-Mn-Ga/In/Sn family) ---
    martensite_start_temperature = "martensite_start_temperature"   # M_s
    martensite_finish_temperature = "martensite_finish_temperature" # M_f
    austenite_start_temperature = "austenite_start_temperature"     # A_s
    austenite_finish_temperature = "austenite_finish_temperature"   # A_f
    magnetostriction = "magnetostriction"      # field-induced strain
    other = "other"                            # capture value + note when unlisted


# Canonical unit per property + a coarse plausibility range (in canonical unit)
# used by s06 to normalize and to flag likely extraction errors.
# range is (low, high); None means "don't range-check".
CANONICAL_UNITS: dict[str, dict] = {
    "thermal_conductivity_total":      {"unit": "W/m/K",   "range": (0.1, 200)},
    "thermal_conductivity_lattice":    {"unit": "W/m/K",   "range": (0.1, 200)},
    "thermal_conductivity_electronic": {"unit": "W/m/K",   "range": (0.0, 200)},
    "thermal_diffusivity":             {"unit": "mm2/s",   "range": (0.01, 100)},
    "seebeck_coefficient":             {"unit": "uV/K",    "range": (-1500, 1500)},
    "electrical_conductivity":         {"unit": "S/cm",    "range": (0.0, 1e6)},
    "electrical_resistivity":          {"unit": "ohm.cm",  "range": (1e-8, 1e6)},
    "zt":                              {"unit": "1",       "range": (0.0, 5.0)},
    "power_factor":                    {"unit": "uW/cm/K2","range": (0.0, 500)},
    "carrier_concentration":           {"unit": "cm-3",    "range": (1e14, 1e23)},
    "carrier_mobility":                {"unit": "cm2/V/s", "range": (0.0, 1e5)},
    "hall_coefficient":                {"unit": "cm3/C",   "range": None},
    "curie_temperature":               {"unit": "K",       "range": (0.0, 1500)},
    "neel_temperature":                {"unit": "K",       "range": (0.0, 1500)},
    "saturation_magnetization":        {"unit": "emu/g",   "range": (0.0, 500)},
    "remanent_magnetization":          {"unit": "emu/g",   "range": (0.0, 500)},
    "magnetic_moment":                 {"unit": "uB/f.u.", "range": (0.0, 30)},
    "coercivity":                      {"unit": "Oe",      "range": (0.0, 1e5)},
    "spin_polarization":               {"unit": "%",       "range": (0.0, 100)},
    "magnetoresistance":               {"unit": "%",       "range": (-100, 1e6)},
    "magnetocaloric_entropy_change":   {"unit": "J/kg/K",  "range": (0.0, 100)},
    "exchange_bias_field":             {"unit": "Oe",      "range": (0.0, 1e5)},
    "vickers_hardness":                {"unit": "GPa",     "range": (0.1, 100)},
    "youngs_modulus":                  {"unit": "GPa",     "range": (1, 700)},
    "bulk_modulus":                    {"unit": "GPa",     "range": (1, 700)},
    "shear_modulus":                   {"unit": "GPa",     "range": (1, 500)},
    "poisson_ratio":                   {"unit": "1",       "range": (0.0, 0.5)},
    "fracture_toughness":              {"unit": "MPa.m0.5","range": (0.0, 50)},
    "yield_strength":                  {"unit": "MPa",     "range": (0.0, 5000)},
    "debye_temperature":               {"unit": "K",       "range": (0.0, 1500)},
    "band_gap":                        {"unit": "eV",      "range": (0.0, 6.0)},
    "half_metallic_gap":               {"unit": "eV",      "range": (0.0, 3.0)},
    "dos_at_fermi":                    {"unit": "states/eV","range": None},
    "effective_mass":                  {"unit": "m_e",     "range": (0.0, 100)},
    "work_function":                   {"unit": "eV",      "range": (0.0, 10)},
    "defect_formation_energy":         {"unit": "eV",      "range": (-5.0, 20)},
    "lattice_parameter":               {"unit": "angstrom","range": (2.0, 15.0)},
    "density":                         {"unit": "g/cm3",   "range": (1.0, 25.0)},
    "thermal_expansion_coeff":         {"unit": "1e-6/K",  "range": (0.0, 100)},
    "specific_heat":                   {"unit": "J/g/K",   "range": (0.0, 5.0)},
    "sound_velocity":                  {"unit": "m/s",     "range": (500, 12000)},
    "melting_temperature":             {"unit": "K",       "range": (300, 3500)},
    "superconducting_tc":              {"unit": "K",       "range": (0.0, 300)},
    "sommerfeld_coefficient":          {"unit": "mJ/mol/K2", "range": (0.0, 500)},
    "martensite_start_temperature":    {"unit": "K",       "range": (0.0, 1200)},
    "martensite_finish_temperature":   {"unit": "K",       "range": (0.0, 1200)},
    "austenite_start_temperature":     {"unit": "K",       "range": (0.0, 1200)},
    "austenite_finish_temperature":    {"unit": "K",       "range": (0.0, 1200)},
    "magnetostriction":                {"unit": "%",       "range": (-20, 20)},
    "other":                           {"unit": "",        "range": None},
}


class Measurement(BaseModel):
    """One reported physical-property value from a sample."""

    property_name: PropertyName = Field(description="Which physical property this value is.")
    data_origin: DataOrigin = Field(
        default=DataOrigin.experimental,
        description="'experimental' if measured on a real sample, 'theoretical' "
        "if computed (DFT / first-principles / model). The dataset prioritizes "
        "experimental values; never mislabel one as the other.",
    )
    value: float = Field(description="Numeric value as reported (before unit conversion).")
    unit: str = Field(
        description="Unit exactly as reported, e.g. 'W/mK', 'uV/K', 'mOhm cm', "
        "'emu/g', 'GPa', 'A'. Keep the paper's unit; normalization happens later."
    )
    value_error: Optional[float] = Field(
        default=None,
        description="Reported measurement uncertainty (± value) in the SAME unit "
        "as `value`, if the paper gives one. Null if not stated.",
    )
    temperature_k: Optional[float] = Field(
        default=None,
        description="Measurement temperature in Kelvin (convert degC to K). Null "
        "if the value is not tied to a specific temperature.",
    )
    measurement_method: Optional[str] = Field(
        default=None,
        description="Instrument/method, e.g. 'LFA' (laser flash), 'PPMS', "
        "'four-probe', 'VSM', 'steady-state'. Null if not stated.",
    )
    measurement_direction: Optional[str] = Field(
        default=None,
        description="Direction for anisotropic samples, e.g. 'parallel to "
        "pressing', 'in-plane', '[100]'. Null if isotropic / not stated.",
    )
    applied_field_t: Optional[float] = Field(
        default=None,
        description="Applied magnetic field in Tesla, for field-dependent values "
        "(M–H, magnetoresistance, magnetocaloric). Null if not field-dependent.",
    )
    applied_pressure_gpa: Optional[float] = Field(
        default=None,
        description="Applied pressure in GPa, for pressure-dependent values. "
        "Null if ambient pressure / not applicable.",
    )
    extraction_source: ExtractionSource = Field(
        default=ExtractionSource.text,
        description="Where in the paper you read this value: 'table', 'text', or "
        "'figure' (read off a plot — approximate). Be honest; figure values are "
        "weighted/validated differently.",
    )
    location_in_paper: str = Field(
        description="Provenance: e.g. 'Table 2, row 3', 'Fig. 4a at 700 K', "
        "'text, section 3.2'. Required."
    )


class Processing(BaseModel):
    """How the sample was made — the 'processing' leg of the processing-structure-
    property triangle. These parameters strongly change the outcome (ordering,
    grain size, density, defects) even at fixed composition, so an ML model that
    learns structure-property relations needs them. Fill from the Methods /
    Experimental section; leave null if the paper doesn't state a field."""

    synthesis_route: Optional[str] = Field(
        default=None,
        description="Primary synthesis method: 'arc melting', 'induction melting', "
        "'solid-state reaction', 'melt spinning', 'ball milling', 'mechanical "
        "alloying', 'self-propagating HT (SHS)', 'Bridgman', 'Czochralski', "
        "'sputtering', 'PLD', etc. Null if not stated.",
    )
    consolidation_method: Optional[str] = Field(
        default=None,
        description="Densification/consolidation: 'SPS' (spark plasma sintering), "
        "'hot pressing', 'cold press + sinter', 'HIP', 'none'. Null if not stated.",
    )
    atmosphere: Optional[str] = Field(
        default=None,
        description="Processing atmosphere: 'Ar', 'vacuum', 'N2', 'air', "
        "'forming gas'. Null if not stated.",
    )
    annealing_temperature_k: Optional[float] = Field(
        default=None, description="Annealing/homogenization temperature in K "
        "(convert degC). Controls ordering & homogeneity. Null if none.")
    annealing_time_h: Optional[float] = Field(
        default=None, description="Annealing time in hours (convert days->h). Null if none.")
    sintering_temperature_k: Optional[float] = Field(
        default=None, description="Sintering/consolidation temperature in K. Null if none.")
    sintering_time_min: Optional[float] = Field(
        default=None, description="Sintering hold time in minutes. Null if none.")
    sintering_pressure_mpa: Optional[float] = Field(
        default=None, description="Applied pressure during consolidation in MPa. Null if none.")
    cooling: Optional[str] = Field(
        default=None, description="Cooling after heat treatment: 'quenched', "
        "'furnace-cooled', 'water-quenched', 'slow-cooled', or a rate. Null if none.")
    milling_time_h: Optional[float] = Field(
        default=None, description="Ball-milling / mechanical-alloying time in hours. Null if none.")
    grain_size_um: Optional[float] = Field(
        default=None, description="Reported average grain/crystallite size in micrometres "
        "(convert nm->um). Mediates transport & mechanical props. Null if none.")
    starting_purity: Optional[str] = Field(
        default=None, description="Purity of starting elements/precursors, e.g. "
        "'99.99%', '4N'. Null if not stated.")


class XrdPeak(BaseModel):
    """One reported XRD reflection — the reciprocal-space (Fourier) signature."""

    two_theta_deg: Optional[float] = Field(default=None, description="Peak 2θ in degrees, if given.")
    d_spacing_ang: Optional[float] = Field(default=None, description="d-spacing in Å, if given.")
    hkl: Optional[str] = Field(default=None, description="Miller indices, e.g. '220', if indexed.")
    relative_intensity: Optional[float] = Field(
        default=None, description="Relative intensity (0–100) if given.")


class Characterization(BaseModel):
    """A non-property characterization reading (Raman, XPS, Mössbauer, optical,
    Hall, neutron, EXAFS/XANES...) that helps understand the material. Capture the
    salient extracted features, not the whole spectrum."""

    technique: str = Field(
        description="e.g. 'Raman', 'XPS', 'Mossbauer', 'optical absorption', "
        "'photoluminescence', 'neutron diffraction', 'EXAFS', 'XANES', 'Hall'.")
    key_features: Optional[str] = Field(
        default=None, description="Salient reported features: Raman peak positions "
        "(cm⁻¹), XPS binding energies (eV) + assigned states, Mössbauer isomer "
        "shift/quadrupole splitting, absorption edge, etc. — as reported.")
    interpretation: Optional[str] = Field(
        default=None, description="What the paper concludes (phase, ordering, "
        "oxidation/charge state, defect, band gap...). Null if none.")
    location_in_paper: Optional[str] = Field(
        default=None, description="Provenance, e.g. 'Fig. 5', 'Table 3'.")


class Sample(BaseModel):
    """One physical sample and the context every measurement on it shares."""

    composition: str = Field(
        description="Nominal chemical formula exactly as reported, e.g. 'Fe2VAl' "
        "or 'Ti(Co0.9Ni0.1)Sb'."
    )
    measured_composition: Optional[str] = Field(
        default=None,
        description="Actual/measured composition if reported separately "
        "(EDS/EPMA/WDS). Null if not given.",
    )
    sample_label: Optional[str] = Field(
        default=None,
        description="The paper's own label for this sample if any, e.g. 'S1', "
        "'x = 0.1', 'sample A'. Helps disambiguate a doping series. Null if none.",
    )
    is_stoichiometric: Optional[bool] = Field(
        default=None,
        description="TRUE if this sample is the UNDOPED, nominally stoichiometric "
        "parent compound (e.g. ZrNiSn, Fe2VAl with no substitution) — including "
        "when the paper calls it 'pristine', 'undoped', 'pure', 'as-grown' or the "
        "x = 0 member of a doping series. FALSE if any site is substituted/alloyed "
        "or the composition is off-stoichiometric. Null only if genuinely unclear. "
        "IMPORTANT: judge the NOMINAL composition, not small EDS/EPMA deviations.",
    )
    doping_level_x: Optional[float] = Field(
        default=None,
        description="The paper's own doping/substitution variable x for this sample "
        "(0.0 for the undoped end-member of a series). Null if the paper uses no "
        "such variable.",
    )
    series_role: Optional[str] = Field(
        default=None,
        description="One of 'pure_endmember' (the x = 0 / undoped member of a "
        "composition series), 'doped_member' (a substituted member), or "
        "'standalone' (not part of a series). Null if unclear.",
    )
    heusler_type: HeuslerType = Field(default=HeuslerType.unknown)
    structure: Optional[str] = Field(
        default=None,
        description="Crystal structure / space group, e.g. 'L2_1', 'C1_b', "
        "'Fm-3m', 'cubic'. Null if not stated.",
    )
    space_group: Optional[str] = Field(
        default=None,
        description="Space group EXACTLY as reported — symbol and/or number, e.g. "
        "'F-43m', 'Fm-3m (No. 225)', 'Pnma'. ALWAYS read the XRD/structure section "
        "for this. Null only if the paper truly never states it.",
    )
    lattice_a_ang: Optional[float] = Field(
        default=None,
        description="Lattice parameter a in Angstrom, from THIS paper's own "
        "XRD/ND refinement (convert nm->Angstrom). MANDATORY when reported — "
        "check the XRD section, structure tables and figure captions. For cubic "
        "phases this is the single lattice constant. Null only if truly absent.",
    )
    lattice_a_source: Optional[str] = Field(
        default=None,
        description="Where lattice_a_ang came from — provenance decides how much the "
        "dataset trusts it. One of: 'xrd_refined' (this paper's Rietveld/XRD fit), "
        "'xrd_stated' (this paper states it without refinement detail), "
        "'from_cell_volume' (you computed a = V^(1/3) from this paper's reported cubic "
        "cell volume), 'from_xray_density' (you computed it from this paper's THEORETICAL "
        "/X-ray density), 'dft' (this paper's own DFT-relaxed value), or 'cited' (taken "
        "from a reference — then put the value in lattice_a_cited_ang, NOT lattice_a_ang).",
    )
    lattice_a_cited_ang: Optional[float] = Field(
        default=None,
        description="A lattice parameter this paper QUOTES FROM ANOTHER WORK or a database "
        "(e.g. 'in agreement with the reported a = 5.93 A of Ref. 12'). Keep it SEPARATE "
        "from lattice_a_ang, which is reserved for this paper's own determination. Useful "
        "as a fallback, so record it rather than discarding it. Null if none.",
    )
    cell_volume_ang3: Optional[float] = Field(
        default=None,
        description="Unit-cell volume in cubic Angstrom, if the paper reports it (common in "
        "Rietveld tables). For a cubic cell a = V^(1/3), so this recovers the lattice "
        "parameter exactly when a itself is not printed. Null if not reported.",
    )
    theoretical_density_gcm3: Optional[float] = Field(
        default=None,
        description="THEORETICAL / X-ray / crystallographic density in g/cm3 — the density "
        "computed from the unit cell, NOT the measured pellet density. Papers usually give "
        "both ('theoretical density 7.19 g/cm3, measured 6.98, i.e. 97% dense'). Keep the "
        "two apart: this one fixes the unit cell exactly (rho = Z*M/(N_A*a^3)), whereas the "
        "measured value is lowered by porosity. Record the MEASURED one as a `density` "
        "measurement. Null if not reported.",
    )
    lattice_b_ang: Optional[float] = Field(
        default=None,
        description="Lattice parameter b in Angstrom for non-cubic phases "
        "(tetragonal/orthorhombic/hexagonal). Null if cubic or not reported.",
    )
    lattice_c_ang: Optional[float] = Field(
        default=None,
        description="Lattice parameter c in Angstrom for non-cubic phases. "
        "Null if cubic or not reported.",
    )
    site_occupancy: Optional[str] = Field(
        default=None,
        description="Reported Wyckoff site assignment / refined occupancies, "
        "e.g. 'Ni on 4c; Ti/Zr mixed on 4a', 'Rietveld: 4a occ 0.97(2)'. "
        "Null if not reported.",
    )
    order_state: Optional[str] = Field(
        default=None,
        description="Atomic site-ordering — the Heusler signature: 'L2_1' (fully "
        "ordered), 'B2' / 'A2' (antisite disorder), 'DO3', or an order parameter. "
        "Capture whenever discussed; routinely omitted. Null if not stated.",
    )
    phase_purity: Optional[str] = Field(
        default=None,
        description="Single-phase or multiphase, e.g. 'single-phase L2_1', 'main "
        "+ ~5% secondary'. Capture even when the value looks clean. Null if not stated.",
    )
    secondary_phases: Optional[str] = Field(
        default=None,
        description="Named secondary/impurity phases with amount if given. Null if none.",
    )
    sample_form: Optional[str] = Field(
        default=None,
        description="'bulk polycrystalline', 'single crystal', 'thin film', "
        "'melt-spun ribbon', 'nanostructured bulk', 'hot-pressed pellet', "
        "'powder'. Null if not stated.",
    )
    relative_density_pct: Optional[float] = Field(
        default=None,
        description="Relative density as % of theoretical. Porosity strongly "
        "affects transport — capture whenever reported. Null if not stated.",
    )
    carrier_type: Optional[str] = Field(
        default=None,
        description="'n' or 'p' majority carrier. Null if not applicable/stated.",
    )
    synthesis_method: Optional[str] = Field(
        default=None,
        description="One-line free-text summary of how the sample was made, e.g. "
        "'arc melting + SPS'. Put the structured details in `processing`.",
    )
    processing: Optional[Processing] = Field(
        default=None,
        description="Structured preparation/processing parameters (synthesis route, "
        "annealing, sintering, atmosphere, grain size...). Fill from the Methods "
        "section — these affect the measured properties and are needed to learn "
        "processing-structure-property relations.",
    )
    charge_state: Optional[str] = Field(
        default=None,
        description="Reported valence / oxidation / charge state of constituent "
        "ions if given (e.g. 'Fe2+/Fe3+', 'Mn3+'). Null if not stated.",
    )
    defect_notes: Optional[str] = Field(
        default=None,
        description="Point-defect / disorder info beyond order_state: vacancies, "
        "antisites, interstitials, off-stoichiometry, doping site. Null if none.",
    )
    xrd: List[XrdPeak] = Field(
        default_factory=list,
        description="Reported XRD peaks (reciprocal-space signature): 2θ / d-spacing "
        "/ hkl. Read the main reflections when a pattern/table is given. Empty if none.",
    )
    characterization: List[Characterization] = Field(
        default_factory=list,
        description="Other characterization readings (Raman, XPS, Mössbauer, optical, "
        "Hall, neutron, EXAFS...) with their key features. Empty if none.",
    )
    measurements: List[Measurement] = Field(
        default_factory=list,
        description="All property values measured/computed on THIS sample.",
    )


class PaperExtraction(BaseModel):
    """Everything extracted from a single paper."""

    doi: str = Field(description="DOI of the paper being extracted.")
    is_experimental: bool = Field(
        description="True if the paper reports experimental measurements on real "
        "samples. False for pure computational/DFT-only or review papers."
    )
    is_heusler: bool = Field(
        description="True if the paper actually concerns Heusler / half-Heusler "
        "compounds (not just mentioning them)."
    )
    samples: List[Sample] = Field(
        default_factory=list,
        description="One entry per distinct physical sample. Empty list if none.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Brief free-text note on anything ambiguous or noteworthy.",
    )


def json_schema() -> dict:
    """JSON Schema for PaperExtraction, used by both extraction backends."""
    return PaperExtraction.model_json_schema()
