"""SQLite storage. One file at data/heusler.sqlite.

Tables:
  papers       — one row per work found by s01 (DOI is the key).
  fulltext     — one row per acquired full text (PDF or Elsevier XML).
  extractions  — one row per paper the LLM processed (raw JSON + status).
  samples      — one physical sample (composition + phase/synthesis context).
  measurements — one property value, linked to a sample (the exported data).
  structures   — crystallographic structure per composition (CIF + parsed).
  reference    — bootstrapped rows from Starrydata2 / ESTM for cross-validation.

Every stage is resumable: it checks these tables and skips finished work.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doi            TEXT PRIMARY KEY,
    title          TEXT,
    year           INTEGER,
    journal        TEXT,
    publisher      TEXT,
    abstract       TEXT,
    is_oa          INTEGER,
    oa_url         TEXT,
    openalex_id    TEXT,
    screen_label   TEXT,
    screen_reason  TEXT,
    richness_score REAL,               -- Stage-1 rule richness (0..1), Phase S
    richness_llm   REAL,               -- Stage-2 Gemini richness (1..5), Phase S
    llm_gate_json  TEXT,               -- full structured LLM-gate verdict
    gold           INTEGER,            -- 1 if the extracted paper passed the gold gate
    added_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fulltext (
    doi        TEXT PRIMARY KEY,
    kind       TEXT,                 -- 'pdf' | 'xml'
    path       TEXT,
    source     TEXT,                 -- 'oa' | 'elsevier' | 'inbox' | 'paywall'
    sha256     TEXT,
    bytes      INTEGER,
    fetched_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (doi) REFERENCES papers(doi)
);

CREATE TABLE IF NOT EXISTS extractions (
    doi          TEXT PRIMARY KEY,
    status       TEXT,               -- 'ok' | 'failed' | 'refused' | 'oversized' | 'empty'
    backend      TEXT,
    model        TEXT,
    raw_json     TEXT,
    error        TEXT,
    extracted_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (doi) REFERENCES papers(doi)
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doi                  TEXT,
    sample_label         TEXT,
    composition          TEXT,
    measured_composition TEXT,
    reduced_formula      TEXT,        -- filled by composition parser (s05)
    heusler_type         TEXT,
    structure            TEXT,
    order_state          TEXT,
    -- purity / doping-series identification (added 2026-08; the pure end-member
    -- of a doping series is the scarce, high-value data for stoichiometric ML)
    is_stoichiometric    INTEGER,
    doping_level_x       REAL,
    series_role          TEXT,
    phase_purity         TEXT,
    secondary_phases     TEXT,
    sample_form          TEXT,
    relative_density_pct REAL,
    carrier_type         TEXT,
    synthesis_method     TEXT,
    -- structured processing parameters (processing-structure-property modality)
    synthesis_route         TEXT,
    consolidation_method    TEXT,
    atmosphere              TEXT,
    annealing_temperature_k REAL,
    annealing_time_h        REAL,
    sintering_temperature_k REAL,
    sintering_time_min      REAL,
    sintering_pressure_mpa  REAL,
    cooling                 TEXT,
    milling_time_h          REAL,
    grain_size_um           REAL,
    starting_purity         TEXT,
    -- point-defect / charge + spectral modalities
    charge_state            TEXT,
    defect_notes            TEXT,
    xrd_json                TEXT,    -- JSON list of XRD peaks (reciprocal-space)
    characterization_json   TEXT,    -- JSON list of Raman/XPS/Mossbauer/... readings
    -- structure-first fields (Phase ST): the paper's OWN structure determination
    space_group             TEXT,    -- as reported, e.g. 'F-43m (No. 216)'
    lattice_a_ang           REAL,    -- paper's own lattice parameter a (Angstrom)
    lattice_b_ang           REAL,
    lattice_c_ang           REAL,
    site_occupancy          TEXT,    -- reported Wyckoff assignment / occupancies
    structure_tier          TEXT,    -- 'A' paper-derived | 'B' DB-validated | 'C' unverified DB | 'D' none
    structure_flag          TEXT,    -- e.g. 'lattice_mismatch_db', 'ambiguous_site'
    FOREIGN KEY (doi) REFERENCES papers(doi)
);

CREATE TABLE IF NOT EXISTS measurements (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id             INTEGER,
    doi                   TEXT,       -- denormalized for convenience
    property_name         TEXT,
    data_origin           TEXT,       -- 'experimental' | 'theoretical' | 'unknown'
    value                 REAL,
    unit                  TEXT,
    value_error           REAL,
    temperature_k         REAL,
    measurement_method    TEXT,
    measurement_direction TEXT,
    applied_field_t       REAL,       -- measurement magnetic field (Tesla)
    applied_pressure_gpa  REAL,       -- measurement pressure (GPa)
    extraction_source     TEXT,       -- 'text' | 'table' | 'figure'
    location_in_paper     TEXT,
    -- filled by s06:
    canonical_value       REAL,
    canonical_unit        TEXT,
    flag                  TEXT,       -- null | 'out_of_range' | 'unconvertible'
    -- filled by s07:
    physics_flag          TEXT,       -- null | 'zt_inconsistent' | ...
    confidence            REAL,       -- 0..1 per-value trust weight (source + flags + physics)
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
    FOREIGN KEY (doi) REFERENCES papers(doi)
);

CREATE TABLE IF NOT EXISTS structures (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    composition      TEXT,
    reduced_formula  TEXT,            -- the SAMPLE's reduced formula (join key)
    matched_formula  TEXT,            -- the prototype formula that actually matched
    match_type       TEXT,            -- 'exact' | 'parent' | 'constructed'
    source           TEXT,            -- 'cod' | 'aflow' | 'mp' | 'oqmd'
    source_id        TEXT,
    is_experimental  INTEGER,
    space_group      TEXT,
    a REAL, b REAL, c REAL, alpha REAL, beta REAL, gamma REAL,
    formation_energy REAL,
    cif_path         TEXT,
    fetched_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sample_structures (
    sample_id        INTEGER PRIMARY KEY,   -- one structure per sample
    doi              TEXT,
    source           TEXT,                  -- 'paper_derived' | 'db_validated'
    prototype        TEXT,                  -- 'C1b' | 'L21' | 'XA' | 'B2' | 'A2' | ...
    spacegroup       TEXT,
    a REAL, b REAL, c REAL,
    cif              TEXT,                  -- full CIF text (deterministic build)
    notes            TEXT,                  -- site assignment provenance / flags
    built_at         TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    doi        TEXT,
    tier       TEXT,                 -- 'oa' | 'crossref' | 'springer' | 'ieee' | 'elsevier'
    status     TEXT,                 -- 'ok' | 'miss' | 'error'
    detail     TEXT,
    ts         TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (doi, tier)          -- one record per DOI x tier => never re-attempt
);

CREATE TABLE IF NOT EXISTS reference (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT,             -- 'starrydata' | 'estm' | 'tematdb'
    ref_id          TEXT,
    doi             TEXT,
    composition     TEXT,
    reduced_formula TEXT,
    property_name   TEXT,
    value           REAL,
    unit            TEXT,
    temperature_k   REAL
);

-- The canonical 155-material deployment set (loaded by s00c from the two DXMag
-- top-100 CSVs). Identity = (reduced_formula, family, space_group): the 44
-- formula-duplicates across the CSVs are the same material under different
-- DXMag calculation records (UUIDs), NOT polymorphs. Campaign queries (s01),
-- candidate structure linkage (s09) and coverage re-audits all key off this.
CREATE TABLE IF NOT EXISTS deployment_candidates (
    material_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    composition     TEXT NOT NULL,    -- as written in the CSV, e.g. 'Fe2HfSi'
    reduced_formula TEXT,
    family          TEXT,             -- 'full_cubic' | 'half_cubic' | ...
    heusler_type    TEXT,             -- 'full' | 'half' | 'inverse'
    space_group     TEXT,
    elements        TEXT,             -- sorted element set, e.g. 'Fe,Hf,Si'
    uuid_canonical  TEXT,             -- preferred DXMag record (phonon-computed first)
    uuids_all       TEXT,             -- every DXMag UUID seen for this material (JSON)
    phonon_computed INTEGER,          -- 1 if any record has phonon results
    phonon_stable   TEXT,             -- 'Yes' | 'No' | NULL (as in DXMag)
    in_training_csv INTEGER DEFAULT 0,
    in_test_csv     INTEGER DEFAULT 0,
    rank_train      INTEGER,
    rank_test       INTEGER,
    lit_tier        TEXT,             -- family/stability tier 'S'|'A'|'B'|'C'|'D'
    kappa_tier      TEXT,             -- EXPERIMENTAL-kappa mining tier: Fe2 family
                                      -- downgraded to 'C' (no bulk exp kappa exists;
                                      -- theory/thin-film only) while lit_tier stays 'S'
                                      -- for the phonon-stability task (DXMag DFT).
    added_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (reduced_formula, family, space_group)
);

CREATE INDEX IF NOT EXISTS idx_meas_sample ON measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_meas_doi ON measurements(doi);
CREATE INDEX IF NOT EXISTS idx_meas_prop ON measurements(property_name);
CREATE INDEX IF NOT EXISTS idx_meas_origin ON measurements(data_origin);
CREATE INDEX IF NOT EXISTS idx_samples_doi ON samples(doi);
CREATE INDEX IF NOT EXISTS idx_samples_reduced ON samples(reduced_formula);
CREATE INDEX IF NOT EXISTS idx_struct_reduced ON structures(reduced_formula);
CREATE INDEX IF NOT EXISTS idx_ref_reduced ON reference(reduced_formula);
CREATE INDEX IF NOT EXISTS idx_papers_screen ON papers(screen_label);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    # generous busy-timeout: background stages (LLM gate, downloads) share this
    # file and commit frequently — waiting beats crashing with 'database is locked'
    conn = sqlite3.connect(DB_PATH, timeout=300)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# processing columns added after the samples table first shipped; ALTER-in on open
_SAMPLE_MIGRATIONS = [
    ("synthesis_route", "TEXT"), ("consolidation_method", "TEXT"),
    ("atmosphere", "TEXT"), ("annealing_temperature_k", "REAL"),
    ("annealing_time_h", "REAL"), ("sintering_temperature_k", "REAL"),
    ("sintering_time_min", "REAL"), ("sintering_pressure_mpa", "REAL"),
    ("cooling", "TEXT"), ("milling_time_h", "REAL"),
    ("grain_size_um", "REAL"), ("starting_purity", "TEXT"),
    ("charge_state", "TEXT"), ("defect_notes", "TEXT"),
    ("xrd_json", "TEXT"), ("characterization_json", "TEXT"),
    # structure-first fields (Phase ST)
    ("space_group", "TEXT"), ("lattice_a_ang", "REAL"),
    # C8: lattice-parameter provenance + the two exact derivation routes
    ("lattice_a_source", "TEXT"), ("lattice_a_cited_ang", "REAL"),
    ("cell_volume_ang3", "REAL"), ("theoretical_density_gcm3", "REAL"),
    ("lattice_b_ang", "REAL"), ("lattice_c_ang", "REAL"),
    ("site_occupancy", "TEXT"), ("structure_tier", "TEXT"),
    ("structure_flag", "TEXT"),
    # purity / doping-series fields
    ("is_stoichiometric", "INTEGER"), ("doping_level_x", "REAL"),
    ("series_role", "TEXT"),
]


def init_db() -> None:
    with connect() as conn:
        # WAL: concurrent stages (gate/download/extract/build) stop blocking each
        # other — writers no longer starve readers (observed multi-minute stalls).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        # deployment_candidates gained kappa_tier after first ship
        dc_existing = {r[1] for r in conn.execute(
            "PRAGMA table_info(deployment_candidates)").fetchall()}
        if dc_existing and "kappa_tier" not in dc_existing:
            conn.execute("ALTER TABLE deployment_candidates ADD COLUMN kappa_tier TEXT")
        # add any processing columns missing from a pre-existing samples table
        existing = {r[1] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        for col, typ in _SAMPLE_MIGRATIONS:
            if col not in existing:
                conn.execute(f"ALTER TABLE samples ADD COLUMN {col} {typ}")
        # structures gained match provenance columns after first ship
        s_existing = {r[1] for r in conn.execute("PRAGMA table_info(structures)").fetchall()}
        for col, typ in (("matched_formula", "TEXT"), ("match_type", "TEXT")):
            if col not in s_existing:
                conn.execute(f"ALTER TABLE structures ADD COLUMN {col} {typ}")
        # measurements gained a per-value confidence column (s07)
        m_existing = {r[1] for r in conn.execute("PRAGMA table_info(measurements)").fetchall()}
        for col, typ in (("confidence", "REAL"), ("applied_field_t", "REAL"),
                         ("applied_pressure_gpa", "REAL")):
            if col not in m_existing:
                conn.execute(f"ALTER TABLE measurements ADD COLUMN {col} {typ}")
        # extractions gained a verification-pass status (Phase R, s04c)
        e_existing = {r[1] for r in conn.execute("PRAGMA table_info(extractions)").fetchall()}
        if "verify_status" not in e_existing:
            conn.execute("ALTER TABLE extractions ADD COLUMN verify_status TEXT")
        # real token usage / billed cost per paper (from usage_metadata), so spend
        # is measured rather than estimated. NULL on rows extracted before this.
        for col, typ in (("prompt_tokens", "INTEGER"), ("output_tokens", "INTEGER"),
                         ("total_tokens", "INTEGER"), ("cost_usd", "REAL"),
                         ("api_calls", "INTEGER")):
            if col not in e_existing:
                conn.execute(f"ALTER TABLE extractions ADD COLUMN {col} {typ}")
        # papers gained richness/gold columns (Phase S selection funnel);
        # mining_campaign tags which targeted campaign (e.g. 'tierS') matched a
        # paper — attribution for per-tier before/after coverage (roadmap C2)
        p_existing = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
        # gate_class: s03b two-class verdict — 'gold' (experimental & Heusler &
        # rich & reports BOTH kappa and crystal structure) | 'silver' (rich but
        # missing one) | NULL (not gated / failed gate). Queryable by s04 order.
        for col, typ in (("richness_score", "REAL"), ("richness_llm", "REAL"),
                         ("llm_gate_json", "TEXT"), ("gold", "INTEGER"),
                         ("mining_campaign", "TEXT"), ("gate_class", "TEXT")):
            if col not in p_existing:
                conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {typ}")


# --- papers / fulltext / extractions (unchanged shape) ---
def upsert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    # mining_campaign: first campaign to claim a paper keeps it (COALESCE order);
    # the global corpus search passes None and never overwrites a campaign tag.
    # New-vs-preexisting attribution comes from added_at, which the upsert never
    # touches after insert.
    paper.setdefault("mining_campaign", None)
    conn.execute(
        """
        INSERT INTO papers (doi, title, year, journal, publisher, abstract,
                            is_oa, oa_url, openalex_id, mining_campaign)
        VALUES (:doi, :title, :year, :journal, :publisher, :abstract,
                :is_oa, :oa_url, :openalex_id, :mining_campaign)
        ON CONFLICT(doi) DO UPDATE SET
            title=excluded.title, year=excluded.year, journal=excluded.journal,
            publisher=excluded.publisher, abstract=excluded.abstract,
            is_oa=excluded.is_oa, oa_url=excluded.oa_url,
            openalex_id=excluded.openalex_id,
            mining_campaign=COALESCE(papers.mining_campaign, excluded.mining_campaign)
        """,
        paper,
    )


def record_fulltext(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO fulltext (doi, kind, path, source, sha256, bytes)
        VALUES (:doi, :kind, :path, :source, :sha256, :bytes)
        ON CONFLICT(doi) DO UPDATE SET
            kind=excluded.kind, path=excluded.path, source=excluded.source,
            sha256=excluded.sha256, bytes=excluded.bytes,
            fetched_at=datetime('now')
        """,
        row,
    )


def has_fulltext(conn: sqlite3.Connection, doi: str) -> bool:
    cur = conn.execute("SELECT 1 FROM fulltext WHERE doi = ?", (doi,))
    return cur.fetchone() is not None


def record_fetch_attempt(conn: sqlite3.Connection, doi: str, tier: str,
                         status: str, detail: str | None = None) -> None:
    """Log a full-text fetch attempt so re-runs never re-hit the same DOI x tier
    (protects limited free-API quotas from duplicate calls)."""
    conn.execute(
        "INSERT INTO fetch_log (doi, tier, status, detail) VALUES (?,?,?,?) "
        "ON CONFLICT(doi, tier) DO UPDATE SET status=excluded.status, "
        "detail=excluded.detail, ts=datetime('now')",
        (doi, tier, status, detail),
    )


def attempted_tiers(conn: sqlite3.Connection, doi: str) -> set:
    """Tiers already tried for this DOI (skip them to avoid wasted API calls)."""
    return {r[0] for r in conn.execute(
        "SELECT tier FROM fetch_log WHERE doi = ?", (doi,)).fetchall()}


def record_extraction(conn: sqlite3.Connection, row: dict) -> None:
    row = {"prompt_tokens": None, "output_tokens": None, "total_tokens": None,
           "cost_usd": None, "api_calls": None, **row}
    conn.execute(
        """
        INSERT INTO extractions (doi, status, backend, model, raw_json, error,
                                 prompt_tokens, output_tokens, total_tokens,
                                 cost_usd, api_calls)
        VALUES (:doi, :status, :backend, :model, :raw_json, :error,
                :prompt_tokens, :output_tokens, :total_tokens,
                :cost_usd, :api_calls)
        ON CONFLICT(doi) DO UPDATE SET
            status=excluded.status, backend=excluded.backend, model=excluded.model,
            raw_json=excluded.raw_json, error=excluded.error,
            prompt_tokens=excluded.prompt_tokens, output_tokens=excluded.output_tokens,
            total_tokens=excluded.total_tokens, cost_usd=excluded.cost_usd,
            api_calls=excluded.api_calls,
            extracted_at=datetime('now')
        """,
        row,
    )


# --- samples / measurements ---
def clear_paper_samples(conn: sqlite3.Connection, doi: str) -> None:
    """Remove a paper's samples + measurements + built structures (re-run safe).
    Clearing sample_structures too prevents orphaned rows: s05 reinserts samples
    with NEW sample_ids, so old sample_structures rows would otherwise dangle and
    s10 would write dead CIF files for them."""
    conn.execute(
        "DELETE FROM measurements WHERE sample_id IN "
        "(SELECT sample_id FROM samples WHERE doi = ?)", (doi,))
    conn.execute(
        "DELETE FROM sample_structures WHERE sample_id IN "
        "(SELECT sample_id FROM samples WHERE doi = ?)", (doi,))
    conn.execute("DELETE FROM samples WHERE doi = ?", (doi,))


def insert_sample(conn: sqlite3.Connection, row: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO samples
            (doi, sample_label, composition, measured_composition, reduced_formula,
             heusler_type, structure, order_state,
             is_stoichiometric, doping_level_x, series_role,
             phase_purity, secondary_phases,
             sample_form, relative_density_pct, carrier_type, synthesis_method,
             synthesis_route, consolidation_method, atmosphere, annealing_temperature_k,
             annealing_time_h, sintering_temperature_k, sintering_time_min,
             sintering_pressure_mpa, cooling, milling_time_h, grain_size_um, starting_purity,
             charge_state, defect_notes, xrd_json, characterization_json,
             space_group, lattice_a_ang, lattice_b_ang, lattice_c_ang, site_occupancy,
             lattice_a_source, lattice_a_cited_ang, cell_volume_ang3,
             theoretical_density_gcm3)
        VALUES
            (:doi, :sample_label, :composition, :measured_composition, :reduced_formula,
             :heusler_type, :structure, :order_state,
             :is_stoichiometric, :doping_level_x, :series_role,
             :phase_purity, :secondary_phases,
             :sample_form, :relative_density_pct, :carrier_type, :synthesis_method,
             :synthesis_route, :consolidation_method, :atmosphere, :annealing_temperature_k,
             :annealing_time_h, :sintering_temperature_k, :sintering_time_min,
             :sintering_pressure_mpa, :cooling, :milling_time_h, :grain_size_um, :starting_purity,
             :charge_state, :defect_notes, :xrd_json, :characterization_json,
             :space_group, :lattice_a_ang, :lattice_b_ang, :lattice_c_ang, :site_occupancy,
             :lattice_a_source, :lattice_a_cited_ang, :cell_volume_ang3,
             :theoretical_density_gcm3)
        """,
        row,
    )
    return cur.lastrowid


def insert_measurement(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO measurements
            (sample_id, doi, property_name, data_origin, value, unit, value_error,
             temperature_k, measurement_method, measurement_direction,
             applied_field_t, applied_pressure_gpa, extraction_source, location_in_paper)
        VALUES
            (:sample_id, :doi, :property_name, :data_origin, :value, :unit, :value_error,
             :temperature_k, :measurement_method, :measurement_direction,
             :applied_field_t, :applied_pressure_gpa, :extraction_source, :location_in_paper)
        """,
        row,
    )


def insert_structure(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO structures
            (composition, reduced_formula, matched_formula, match_type, source,
             source_id, is_experimental, space_group, a, b, c, alpha, beta, gamma,
             formation_energy, cif_path)
        VALUES
            (:composition, :reduced_formula, :matched_formula, :match_type, :source,
             :source_id, :is_experimental, :space_group, :a, :b, :c, :alpha, :beta, :gamma,
             :formation_energy, :cif_path)
        """,
        row,
    )


def upsert_sample_structure(conn: sqlite3.Connection, row: dict) -> None:
    """Store the per-sample deterministic structure (paper-derived CIF). One per
    sample; rebuilt rows replace the old build (re-run safe)."""
    conn.execute(
        """
        INSERT INTO sample_structures
            (sample_id, doi, source, prototype, spacegroup, a, b, c, cif, notes)
        VALUES
            (:sample_id, :doi, :source, :prototype, :spacegroup, :a, :b, :c, :cif, :notes)
        ON CONFLICT(sample_id) DO UPDATE SET
            source=excluded.source, prototype=excluded.prototype,
            spacegroup=excluded.spacegroup, a=excluded.a, b=excluded.b, c=excluded.c,
            cif=excluded.cif, notes=excluded.notes, built_at=datetime('now')
        """,
        row,
    )


def insert_reference(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO reference
            (source, ref_id, doi, composition, reduced_formula,
             property_name, value, unit, temperature_k)
        VALUES
            (:source, :ref_id, :doi, :composition, :reduced_formula,
             :property_name, :value, :unit, :temperature_k)
        """,
        row,
    )


def upsert_candidate(conn: sqlite3.Connection, row: dict) -> None:
    """One row per deployment material; re-running the s00c loader merges CSV
    membership flags and keeps the phonon-computed record as canonical UUID."""
    conn.execute(
        """
        INSERT INTO deployment_candidates
            (composition, reduced_formula, family, heusler_type, space_group,
             elements, uuid_canonical, uuids_all, phonon_computed, phonon_stable,
             in_training_csv, in_test_csv, rank_train, rank_test, lit_tier, kappa_tier)
        VALUES
            (:composition, :reduced_formula, :family, :heusler_type, :space_group,
             :elements, :uuid_canonical, :uuids_all, :phonon_computed, :phonon_stable,
             :in_training_csv, :in_test_csv, :rank_train, :rank_test, :lit_tier, :kappa_tier)
        ON CONFLICT(reduced_formula, family, space_group) DO UPDATE SET
            uuid_canonical=excluded.uuid_canonical, uuids_all=excluded.uuids_all,
            phonon_computed=excluded.phonon_computed, phonon_stable=excluded.phonon_stable,
            in_training_csv=MAX(deployment_candidates.in_training_csv, excluded.in_training_csv),
            in_test_csv=MAX(deployment_candidates.in_test_csv, excluded.in_test_csv),
            rank_train=COALESCE(deployment_candidates.rank_train, excluded.rank_train),
            rank_test=COALESCE(deployment_candidates.rank_test, excluded.rank_test),
            lit_tier=excluded.lit_tier, kappa_tier=excluded.kappa_tier
        """,
        row,
    )


def extracted_dois(conn: sqlite3.Connection, statuses: Optional[tuple] = None) -> set:
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        cur = conn.execute(
            f"SELECT doi FROM extractions WHERE status IN ({placeholders})", statuses
        )
    else:
        cur = conn.execute("SELECT doi FROM extractions")
    return {r["doi"] for r in cur.fetchall()}
