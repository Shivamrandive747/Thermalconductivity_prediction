"""Screen the PDFs already on disk that no extraction run has ever seen.

`run_extraction.py` discovers work from the fetch LOGS, so every PDF added by hand is invisible to
it: 1,368 files on disk, 66 ever extracted. Those were downloaded for this project and never read.

Screening is free -- `pdftotext` plus two regexes -- so all of them can be checked before any model
call is paid for. A document only earns extraction if its own text mentions BOTH a Heusler-family
term and a thermal-conductivity term. That is deliberately generous: it only has to reject papers
that never discuss the property at all.

Writes a ranked worklist so extraction can run on the survivors in order of promise, with the ones
that also mention grain size first -- per-sample grain size is the last reducible error source in
the transfer model.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

HEUSLER = re.compile(r"heusler|half.?heusler|C1b|L2\s?1|MgAgAs|F-?43m|Fm-?3m", re.I)
KAPPA = re.compile(r"lattice thermal conductivity|thermal conductivity|kappa_?[lL]\b|"
                   r"W\s*m\s*[-\u2212]?1\s*K\s*[-\u2212]?1|W/mK", re.I)
LATTICE = re.compile(r"lattice thermal conductivity|\bkappa_?[lL]\b|\bk_?lat", re.I)
FULL = re.compile(r"full.?heusler|X2YZ|\bL2\s?1\b", re.I)
GRAIN = re.compile(r"grain size|crystallite size|scherrer|spark plasma|ball.?mill|hot.?press",
                   re.I)
TABLE = re.compile(r"table\s+\d|table\s+s\d", re.I)
OUT = "data/external/LOCAL_PDF_WORKLIST.csv"


def text_of(p: Path, pages: int = 12) -> str:
    try:
        r = subprocess.run(["pdftotext", "-l", str(pages), str(p), "-"],
                           capture_output=True, timeout=60)
        return (r.stdout or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def main(limit: int | None = None) -> int:
    done = set()
    for f in glob.glob("data/external/pdfx_*.csv"):
        m = re.match(r"pdfx_(.+?)__(si|pdf)_", os.path.basename(f))
        if m:
            done.add(m.group(1))
    pdfs = [Path(p) for p in sorted(glob.glob("data/pdfs/*.pdf"))
            if Path(p).stem not in done]
    if limit:
        pdfs = pdfs[:limit]
    print(f"PDFs never extracted: {len(pdfs)}   screening (free, no model calls)\n", flush=True)

    rows, unreadable = [], 0
    for i, p in enumerate(pdfs, 1):
        t = text_of(p)
        if len(t) < 400:
            unreadable += 1
            continue
        h, k = bool(HEUSLER.search(t)), bool(KAPPA.search(t))
        if not (h and k):
            continue
        rows.append(dict(path=str(p), stem=p.stem,
                         heusler=h, kappa=k,
                         lattice_kappa=bool(LATTICE.search(t)),
                         full_heusler=bool(FULL.search(t)),
                         grain=bool(GRAIN.search(t)),
                         has_table=bool(TABLE.search(t)),
                         size_mb=round(p.stat().st_size / 1e6, 2)))
        if i % 100 == 0:
            print(f"  {i}/{len(pdfs)} screened -> {len(rows)} relevant", flush=True)

    d = pd.DataFrame(rows)
    if not len(d):
        print("nothing relevant found")
        return 0
    d["score"] = (d.lattice_kappa.astype(int) * 4 + d.full_heusler.astype(int) * 3
                  + d.has_table.astype(int) * 2 + d.grain.astype(int) * 2)
    d = d.sort_values(["score", "lattice_kappa"], ascending=False)
    d.to_csv(OUT, index=False)

    print(f"\n=== screened {len(pdfs)} PDFs ===")
    print(f"  unreadable (no text layer)      : {unreadable}")
    print(f"  mention Heusler AND thermal cond: {len(d)}")
    print(f"  ... of which LATTICE kappa      : {int(d.lattice_kappa.sum())}")
    print(f"  ... mention a FULL Heusler      : {int(d.full_heusler.sum())}")
    print(f"  ... contain a numbered table    : {int(d.has_table.sum())}")
    print(f"  ... mention grain size          : {int(d.grain.sum())}")
    print(f"\n  worth a model call (score >= 6) : {int((d.score >= 6).sum())}")
    print(f"  estimated cost at $0.24 each    : ${0.24 * int((d.score >= 6).sum()):.2f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    raise SystemExit(main(**vars(ap.parse_args())))
