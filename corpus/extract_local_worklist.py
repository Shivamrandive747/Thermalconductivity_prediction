"""Extract the screened local PDFs, highest-promise first.

`run_extraction.py` builds its worklist from the fetch LOGS, so 1,304 PDFs that were added by hand
had never been offered to it. Screening found 630 that mention both a Heusler term and thermal
conductivity, 280 of them naming a FULL Heusler -- the class the model is starved of.

The DOI is recovered from the filename, which is how every other stage of this pipeline names its
files: `10.1002_adfm.201300663` -> `10.1002/adfm.201300663`. Only the FIRST underscore is a slash;
the rest belong to the suffix. Files whose names are not DOI-shaped (`1-s2.0-...-main`) are still
extracted, tagged with a `local:` identifier, so a real table is never thrown away for want of a
tidy filename -- but they are marked so provenance stays honest.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

WORKLIST = "data/external/LOCAL_PDF_WORKLIST.csv"
OUT_DIR = Path("data/external")
DOI_LIKE = re.compile(r"^(10\.\d{4,9})[_.](.+)$")


def doi_of(stem: str) -> tuple[str, bool]:
    m = DOI_LIKE.match(stem)
    if m:
        return f"{m.group(1)}/{m.group(2)}", True
    return f"local:{stem}", False


def already_extracted() -> set:
    """DOIs the measurement DB already holds.

    THE CHECK THAT WAS MISSING. An earlier session extracted most of these PDFs and wrote the
    results straight into `heusler.sqlite`, not to `pdfx_*.csv`. Screening for missing pdfx files
    therefore reported 1,304 PDFs as "never extracted" when 67% of the relevant ones were already
    done -- and re-extracting them would have cost about $50 to re-derive rows we hold.
    Deduplicate against the DATABASE, which is where the extractions actually live.
    """
    import glob as _glob
    import sqlite3
    try:
        con = sqlite3.connect("file:data/heusler.sqlite?mode=ro", uri=True)
        done = {str(x[0]).lower() for x in con.execute(
            "select distinct doi from measurements where doi is not null").fetchall()}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read the measurement DB ({exc}).")
        print("  Refusing to run: without it every already-extracted paper would be paid for again.")
        raise SystemExit(1)

    # The DB is not the only place extractions live. Anything extracted by the CURRENT session
    # sits in data/external/pdfx_*.csv and has not been ingested yet, so a DB-only check still
    # lets today's work be paid for twice -- 10.1016/j.mtcomm.2023.106289 was extracted this
    # morning and was still showing as new.
    n0 = len(done)
    for f in _glob.glob("data/external/pdfx_*.csv"):
        try:
            c = pd.read_csv(f)
            if "source_doi" in c.columns:
                done |= {str(x).lower().strip() for x in c.source_doi.dropna()}
        except Exception:  # noqa: BLE001
            continue
    print(f"  known-extracted DOIs: {n0} from the DB + {len(done) - n0} from this session's CSVs")
    return done


def main(limit: int | None, min_score: int, budget_min: float, dry_run: bool) -> int:
    d = pd.read_csv(WORKLIST)
    d = d[d.score >= min_score].sort_values(["score", "lattice_kappa"], ascending=False)

    db = already_extracted()

    # SECOND NAMING CONVENTION. Elsewhere this pipeline names files with
    # re.sub(r"[^a-zA-Z0-9]+", "_", doi), which turns 10.1016/j.mtcomm.2023.106289 into
    # 10_1016_j_mtcomm_2023_106289 -- no dots at all, so the DOI regex cannot parse it back.
    # 59 of 182 queued files looked "new" purely because of that, and at least one had been
    # extracted the same day. Reversing the substitution is ambiguous, but matching FORWARD is
    # exact: apply the same transform to every DOI the database knows and compare stems.
    safe_map = {re.sub(r"[^a-zA-Z0-9]+", "_", x): x for x in db}

    def resolve(stem: str):
        s2 = str(stem)
        if s2 in safe_map:
            return safe_map[s2]
        doi, ok = doi_of(s2)
        return doi.lower() if ok else None

    d["doi_guess"] = d.stem.map(resolve)
    before = len(d)
    dup = d.doi_guess.notna() & d.doi_guess.isin(db)
    d = d[~dup]
    print(f"  already in the measurement DB, skipped: {int(dup.sum())} of {before}")
    if limit:
        d = d.head(limit)
    print(f"worklist: {len(d)} PDFs (score >= {min_score})")
    print(f"  full-Heusler mentions : {int(d.full_heusler.sum())}")
    print(f"  lattice kappa         : {int(d.lattice_kappa.sum())}")
    print(f"  grain size            : {int(d.grain.sum())}")
    if dry_run:
        for _, r in d.head(20).iterrows():
            doi, ok = doi_of(r.stem)
            print(f"  {r.score:>2} {'DOI ' if ok else 'LOCAL'} {doi[:44]:<46}{r.stem[:34]}")
        return 0

    deadline = time.time() + budget_min * 60
    ok = fail = 0
    for i, (_, r) in enumerate(d.iterrows(), 1):
        if time.time() > deadline:
            print(f"\n  WALL CLOCK reached -- {len(d) - i + 1} left for the next run")
            break
        doi, is_doi = doi_of(r.stem)
        out = OUT_DIR / f"pdfx_{re.sub(r'[^a-zA-Z0-9]+', '_', doi)}__pdf_local.csv"
        if out.exists():
            continue
        cmd = [sys.executable, "extract_kappa_pdf.py", "--pdf", str(r.path),
               "--doi", doi, "--out", str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=min(600, max(60, int(deadline - time.time()))),
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            print(f"  [{i}/{len(d)}] {doi[:40]:<42} TIMEOUT", flush=True)
            fail += 1
            continue
        txt = (p.stdout or "") + (p.stderr or "")
        m = re.search(r"kept (\d+) rows, (\d+) distinct", txt)
        if m:
            ok += 1
            print(f"  [{i}/{len(d)}] {doi[:40]:<42} {m.group(1):>4} rows  "
                  f"{m.group(2)} compounds", flush=True)
        elif "no lattice thermal conductivity table" in txt:
            print(f"  [{i}/{len(d)}] {doi[:40]:<42} no kappa table", flush=True)
        else:
            fail += 1
            err = [ln for ln in txt.splitlines() if ln.strip()][-1:] or ["failed"]
            print(f"  [{i}/{len(d)}] {doi[:40]:<42} FAIL {err[0][:52]}", flush=True)
    print(f"\n=== extracted {ok},  failed {fail} ===")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-score", type=int, default=6)
    ap.add_argument("--budget-min", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(**vars(ap.parse_args())))
