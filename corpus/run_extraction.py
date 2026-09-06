"""Step 4 — run kappa extraction over every fetched paper and supplementary file.

WHY A RUNNER AND NOT A LOOP. Three things have gone wrong with unattended extraction on this
project, and each is guarded here:

1. **Runaway retries.** A past incident spent 4 h 33 m on ONE paper because three independent retry
   budgets multiplied (3 x 900 s). So the bound here is a WALL CLOCK over the whole run, checked
   before every call. Attempt counts are not a bound; time is.

2. **Vertex 429 RESOURCE_EXHAUSTED.** Hit this session. A 429 is a rate signal, not a failure: back
   off, and if the wall clock is nearly spent, stop cleanly with work saved rather than thrash.

3. **Losing completed work on a rerun.** Every input writes its own `pdfx_*.csv` and an existing one
   is skipped, so a re-run resumes instead of re-paying. Extraction costs roughly $0.24 per
   document, so re-doing 200 of them is a real amount of money.

WHAT IT FEEDS ON. Both article PDFs (`data/pdfs/`) and supplementary spreadsheets
(`data/external/si/`). The SI files matter more: the hundred-row tables live there, and
`extract_kappa_pdf.py` sends a spreadsheet as text rather than as an uploaded document, which is
cheaper and shows the model real cell values instead of a picture of them.

Run `--dry-run` first. It prints exactly what would be processed and the estimated cost, and calls
nothing.
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

QUEUE = "data/external/mining_queue_ranked.csv"
LOG = "data/external/paper_fetch_log.csv"
PDF_DIR = Path("data/pdfs")
SI_DIR = Path("data/external/si")
OUT_DIR = Path("data/external")
COST_PER_DOC = 0.24        # USD, observed


def safe(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(doi))


def discover() -> list[tuple[str, Path, str]]:
    """(doi, file, kind) for everything fetched, SI before PDF -- SI carries the bigger tables."""
    jobs: list[tuple[str, Path, str]] = []
    dois: dict[str, str] = {}
    for f in (QUEUE, LOG):
        p = Path(f)
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if "doi" in d.columns:
            for x in d.doi.dropna().unique():
                dois[safe(x)] = str(x)

    for stem, doi in dois.items():
        for si in sorted(SI_DIR.glob(f"{stem}_SI*.*")):
            if si.suffix.lower() in (".xlsx", ".xls", ".csv", ".pdf"):
                jobs.append((doi, si, "si"))
        pdf = PDF_DIR / f"{stem}.pdf"
        if pdf.exists() and pdf.stat().st_size > 25_000:
            jobs.append((doi, pdf, "pdf"))
    return jobs


def out_path(doi: str, f: Path, kind: str) -> Path:
    return OUT_DIR / f"pdfx_{safe(doi)}__{kind}_{f.stem[-6:]}.csv"


# A document only earns a model call if its own text mentions lattice thermal conductivity.
KAPPA_WORDS = re.compile(
    r"lattice thermal conductivity|thermal conductivity|kappa_?[lL]\b|\bk_?[lL]\b|"
    r"W\s*m\s*[-−]?1\s*K\s*[-−]?1|W/mK|W m-1 K-1", re.I)
HEUSLER_WORDS = re.compile(r"heusler|half.?heusler|C1b|L2\s?1|MgAgAs|F-?43m|Fm-?3m", re.I)


def prescreen(f: Path) -> tuple[bool, str]:
    """Cheap local check before paying for a model call.

    The first SI batch cost $1.44 and returned data from ONE of six documents: the supplementary
    files of *experimental* discovery papers hold synthesis routes and XRD patterns, not kappa
    tables. `pdftotext` and pandas can see that for free, so they screen first.

    Deliberately generous -- it only has to reject documents that never mention thermal conductivity
    at all. Anything ambiguous still goes to the model."""
    try:
        if f.suffix.lower() in (".xlsx", ".xls", ".csv"):
            if f.suffix.lower() == ".csv":
                txt = f.read_text(errors="replace")[:400_000]
            else:
                xl = pd.ExcelFile(f)
                parts = []
                for n in xl.sheet_names[:12]:
                    d = xl.parse(n, nrows=60)
                    parts.append(" ".join(map(str, d.columns)) + " " + d.to_csv(index=False)[:8000])
                txt = " ".join(parts)
        else:
            r = subprocess.run(["pdftotext", "-layout", "-l", "40", str(f), "-"],
                               capture_output=True, timeout=120)
            txt = (r.stdout or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return True, "unreadable locally -> send to model"

    if not txt.strip():
        return True, "no text layer -> send to model (may be scanned tables)"
    n_kappa = len(KAPPA_WORDS.findall(txt))
    if n_kappa == 0:
        return False, "no thermal-conductivity mention anywhere"
    if not HEUSLER_WORDS.search(txt) and n_kappa < 3:
        return False, f"only {n_kappa} kappa mention(s) and no Heusler mention"
    return True, f"{n_kappa} kappa mentions"


def main(limit: int | None, budget_min: float, dry_run: bool, model: str | None,
         si_first: bool) -> int:
    jobs = discover()
    todo = [(d, f, k) for d, f, k in jobs if not out_path(d, f, k).exists()]
    if si_first:
        todo.sort(key=lambda t: 0 if t[2] == "si" else 1)
    done_already = len(jobs) - len(todo)
    if limit:
        todo = todo[:limit]

    print(f"inputs discovered : {len(jobs)}  ({sum(1 for j in jobs if j[2] == 'si')} SI, "
          f"{sum(1 for j in jobs if j[2] == 'pdf')} PDF)")
    print(f"already extracted : {done_already}  (skipped)")
    print(f"to process now    : {len(todo)}   est. cost ${len(todo) * COST_PER_DOC:,.2f}")
    print(f"wall-clock budget : {budget_min:.0f} min")
    if dry_run:
        for d, f, k in todo[:25]:
            print(f"    {k:<3} {d:<34}{f.name}")
        if len(todo) > 25:
            print(f"    ... and {len(todo) - 25} more")
        print("\ndry run -- nothing called")
        return 0

    deadline = time.time() + budget_min * 60
    ok = fail = skipped = 0
    rejected: list[dict] = []
    backoff = 20.0
    for i, (doi, f, kind) in enumerate(todo, 1):
        if time.time() > deadline:
            print(f"\n  WALL CLOCK REACHED -- stopping cleanly with {ok} extracted, "
                  f"{len(todo) - i + 1} left for the next run")
            skipped = len(todo) - i + 1
            break
        o = out_path(doi, f, kind)
        keep, why = prescreen(f)
        if not keep:
            rejected.append(dict(doi=doi, file=f.name, kind=kind, reason=why))
            print(f"  [{i}/{len(todo)}] {doi:<34}{kind:<4} SKIP ({why})", flush=True)
            continue
        # sys.executable, not a hard-coded venv path: CreateProcess on Windows will not resolve
        # ".venv/Scripts/python.exe" as a relative forward-slash path.
        cmd = [sys.executable, "extract_kappa_pdf.py",
               "--pdf", str(f), "--doi", doi, "--out", str(o)]
        if model:
            cmd += ["--model", model]
        remaining = max(60, int(deadline - time.time()))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=min(600, remaining), encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            print(f"  [{i}/{len(todo)}] {doi:<34}{kind:<4} TIMEOUT")
            fail += 1
            continue
        blob = (r.stdout or "") + (r.stderr or "")
        if "RESOURCE_EXHAUSTED" in blob or "429" in blob:
            print(f"  [{i}/{len(todo)}] {doi:<34}{kind:<4} 429 -- backing off {backoff:.0f}s")
            time.sleep(min(backoff, max(0, deadline - time.time())))
            backoff = min(backoff * 2, 300)
            fail += 1
            continue
        backoff = 20.0
        if o.exists():
            try:
                n = len(pd.read_csv(o))
            except Exception:  # noqa: BLE001
                n = 0
            ok += 1
            print(f"  [{i}/{len(todo)}] {doi:<34}{kind:<4} {n:>4} rows", flush=True)
        else:
            m = re.search(r"model returned (\d+) rows|no lattice thermal conductivity", blob)
            print(f"  [{i}/{len(todo)}] {doi:<34}{kind:<4} "
                  f"{'no kappa table' if m else 'failed'}", flush=True)
            fail += 1

    print(f"\n=== EXTRACTION ===")
    print(f"  extracted : {ok}")
    print(f"  no data / failed : {fail}")
    if rejected:
        print(f"  pre-screened out (FREE, no model call): {len(rejected)}"
              f"   -> saved ~${len(rejected) * COST_PER_DOC:,.2f}")
        pd.DataFrame(rejected).to_csv(OUT_DIR / "extraction_prescreen_rejects.csv", index=False)
    if skipped:
        print(f"  left for next run : {skipped}  (rerun the same command to resume)")
    print(f"  spent ~${(ok + fail) * COST_PER_DOC:,.2f}")
    print("\nnext: python merge_pdf_extractions.py && python build_kappa_pool.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--budget-min", type=float, default=45.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--si-first", action="store_true", default=True)
    raise SystemExit(main(**vars(ap.parse_args())))
