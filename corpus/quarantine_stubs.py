"""Quarantine every extraction made from an Elsevier entitlement stub — durably.

An earlier attempt flagged the MEASUREMENTS table. That does not survive: s05 calls
db.clear_paper_samples() and rebuilds measurements from extractions.raw_json, so Phase 4
and Phase 5 silently restored 7,611 fabricated rows (1,813 of them kappa). Anything meant
to persist has to sit on `extractions`, which s05 reads, not on `measurements`, which it
regenerates.

Detection is by CONTENT, not size. The previous `_MIN_BODY_BYTES = 3000` heuristic was
calibrated on stubs of 580-2,915 B, but Elsevier returns coredata-only responses up to
130,455 B when an article carries long reference lists — 226 of those cleared the gate.
The reliable signature is structural: a <full-text-retrieval-response> that contains no
<ce:para> or <ce:section> has no body text at all, whatever its size.

Nothing is deleted. raw_json is preserved for audit, and setting status='stub_source'
leaves the paper eligible for re-extraction once a real PDF is downloaded.

    python quarantine_stubs.py --dry-run
    python quarantine_stubs.py
"""
from __future__ import annotations

import argparse
import os
import sqlite3

DB = "data/heusler.sqlite"


def is_stub(path: str | None) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            t = f.read()
    except OSError:
        return False
    if "<full-text-retrieval-response" not in t:
        return False
    return "<ce:para" not in t and "<ce:section" not in t


def main(dry_run: bool = False) -> int:
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT doi, path, bytes FROM fulltext WHERE kind='xml'").fetchall()
    stubs = [doi for doi, path, _ in rows if is_stub(path)]
    print(f"xml fulltexts scanned      : {len(rows)}")
    print(f"entitlement stubs (content): {len(stubs)}")
    if not stubs:
        return 0
    ph = ",".join("?" for _ in stubs)

    n_ext = conn.execute(f"SELECT COUNT(*) FROM extractions WHERE doi IN ({ph}) "
                         f"AND status='ok'", stubs).fetchone()[0]
    n_meas = conn.execute(f"SELECT COUNT(*) FROM measurements WHERE doi IN ({ph})",
                          stubs).fetchone()[0]
    n_kappa = conn.execute(f"SELECT COUNT(*) FROM measurements WHERE doi IN ({ph}) "
                           f"AND property_name LIKE 'thermal_conductivity%'",
                           stubs).fetchone()[0]
    print(f"  extractions from them    : {n_ext}")
    print(f"  measurements they created: {n_meas}  (kappa {n_kappa})")

    if dry_run:
        print("\nDRY RUN — nothing changed.")
        return 0

    # 1. stop s05 regenerating them. raw_json is kept so the decision is auditable.
    conn.execute(f"UPDATE extractions SET status='stub_source' "
                 f"WHERE doi IN ({ph}) AND status='ok'", stubs)
    # 2. remove the rows already in the tables
    conn.execute(f"DELETE FROM measurements WHERE doi IN ({ph})", stubs)
    conn.execute(f"DELETE FROM samples WHERE doi IN ({ph})", stubs)
    # 3. drop the useless fulltext rows so s02d/s04 re-queue the paper for a real fetch
    conn.execute(f"DELETE FROM fulltext WHERE doi IN ({ph})", stubs)
    conn.commit()

    left = conn.execute(f"SELECT COUNT(*) FROM measurements WHERE doi IN ({ph})",
                        stubs).fetchone()[0]
    print(f"\nquarantined {len(stubs)} stub papers")
    print(f"  extractions marked 'stub_source' : {n_ext}")
    print(f"  measurements removed             : {n_meas}")
    print(f"  measurements remaining (must be 0): {left}")
    print("\nNext: rebuild the export ->")
    print("  python -m pipeline.s06_validate_export")
    print("  python -m pipeline.s07_validate_physics")
    print("  python -m pipeline.s09b_build_structures")
    print("  python -m pipeline.s10_featurize --version v11 --include-external --include-thin")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(**vars(ap.parse_args())))
