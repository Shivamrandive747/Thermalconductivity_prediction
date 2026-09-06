"""Register a hand-placed PDF against its DOI and make it extractable.

A PDF dropped into data/pdfs/ under the publisher's own filename is invisible to the
pipeline: s04 finds work by joining `papers` to `fulltext`, so a file with no fulltext
row simply does not exist as far as extraction is concerned. This copies it to the
doi_slug convention and records it with size and hash.

`bytes` matters specifically: s04 gates on `f.bytes >= 3000` to keep Elsevier entitlement
stubs out, and a NULL there silently makes a perfectly good paper unextractable — the
bug that hid 272 already-downloaded PDFs.

    python add_manual_pdf.py --file "data/pdfs/Advanced Materials - ....pdf" \
                             --doi 10.1002/adma.73841
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

DB = "data/heusler.sqlite"
PDF_DIR = Path("data/pdfs")


def doi_slug(doi: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doi)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(file: str, doi: str, dry_run: bool = False) -> int:
    src = Path(file)
    if not src.exists():
        sys.exit(f"not found: {src}")
    if not open(src, "rb").read(5).startswith(b"%PDF"):
        sys.exit(f"not a PDF: {src}")

    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT title, mining_campaign FROM papers WHERE doi=?",
                       (doi,)).fetchone()
    if not row:
        sys.exit(f"DOI not in papers table: {doi}")
    print(f"doi     : {doi}")
    print(f"title   : {row[0][:78]}")
    print(f"campaign: {row[1]}")
    print(f"size    : {src.stat().st_size:,} B")

    dest = PDF_DIR / f"{doi_slug(doi)}.pdf"
    already = conn.execute("SELECT path, bytes FROM fulltext WHERE doi=?", (doi,)).fetchone()
    if already:
        print(f"NOTE: fulltext row already exists ({already[1]} B) - it will be replaced")

    if dry_run:
        print(f"\nwould copy -> {dest}\nDRY RUN - nothing written.")
        return 0

    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    h, n = sha256(dest), dest.stat().st_size
    conn.execute(
        "INSERT OR REPLACE INTO fulltext (doi, kind, path, source, sha256, bytes) "
        "VALUES (?, 'pdf', ?, 'manual_drop', ?, ?)", (doi, str(dest.resolve()), h, n))
    conn.commit()

    # confirm it is now visible to the extractor's own query
    ok = conn.execute(
        "SELECT COUNT(*) FROM papers p JOIN fulltext f ON f.doi=p.doi "
        "WHERE p.doi=? AND f.bytes >= 3000", (doi,)).fetchone()[0]
    print(f"\nregistered -> {dest.name}")
    print(f"  bytes={n:,}  sha256={h[:16]}...")
    print(f"  visible to s04: {bool(ok)}")
    print(f"\nNext: python -m pipeline.s04_extract --dois {doi} --include-maybe")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--doi", required=True)
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(**vars(ap.parse_args())))
