"""Open queued papers in YOUR normal Chrome, then import whatever you downloaded.

WHY NOT FULL AUTOMATION -- this was tried first and it does not work.
Selenium can now launch Chrome against a copy of your profile (Chrome 136+ blocks automating the
real one), and that part works. But publisher portals sit behind Cloudflare, which serves a
"Just a moment..." challenge to automated browsers and never clears it: measured, 45 s of waiting
on pubs.aip.org with the title unchanged. Getting past that would mean installing detection-evasion
tooling, which breaches publisher terms and puts the institution's subscription at risk. Not worth
five PDFs.

WHAT WORKS INSTEAD. Your ordinary Chrome passes the challenge because it is ordinary. So:

    python get_papers.py open --limit 5      opens 5 article pages as tabs in your real Chrome
    ...you click "Download PDF" on each tab (one click; your login already applies)
    python get_papers.py import              files them into data/pdfs and names them by DOI

One click per paper, and no fight with bot detection. `import` reads your Downloads folder, matches
each new PDF to the DOI whose page produced it, renames it and records the mapping so the
extraction step can cite it.

MATCHING. A downloaded file rarely carries the DOI in its name (`1-s2.0-S0925838824…pdf`), so the
match is made on the PDF's own text: the DOI string almost always appears on the first page. Files
that cannot be matched are left alone and reported, never guessed at.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

LIST_CSV = "data/external/MANUAL_DOWNLOAD_LIST.csv"
PDF_DIR = Path("data/pdfs")
LOG = Path("data/external/manual_fetch_log.csv")


def downloads_dir() -> Path:
    if sys.platform == "win32":
        p = Path(os.path.expandvars(r"%USERPROFILE%\Downloads"))
        if p.exists():
            return p
    return Path.home() / "Downloads"


def safe(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(doi))


def already_have() -> set[str]:
    have = {p.stem for p in PDF_DIR.glob("*.pdf")}
    have |= {p.stem for p in PDF_DIR.glob("**/*.pdf")}
    return have


def cmd_open(limit: int, publisher: str | None, delay: float) -> int:
    d = pd.read_csv(LIST_CSV)
    rank = "priority" if "priority" in d.columns else "score"
    if rank in d.columns:
        d = d.sort_values(rank, ascending=False)
    if publisher:
        d = d[d.publisher.astype(str).str.contains(publisher, case=False, na=False)]
    have = already_have()
    d = d[~d.doi.map(lambda x: safe(x) in have)]
    d = d.head(limit)
    if not len(d):
        print("nothing left to open -- everything queued is already in data/pdfs")
        return 0

    print(f"opening {len(d)} article pages in your normal Chrome, {delay:.0f}s apart\n")
    for i, (_, r) in enumerate(d.iterrows(), 1):
        url = f"https://doi.org/{r.doi}"
        print(f"  [{i}/{len(d)}] {str(r.publisher)[:10]:<12}{r.doi}")
        print(f"           {str(r.title)[:82]}")
        try:
            if sys.platform == "win32":
                os.startfile(url)                      # noqa: S606  (opens default browser)
            else:
                subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", url],
                               check=False)
        except Exception as exc:  # noqa: BLE001
            print(f"           could not open: {str(exc)[:60]}")
        time.sleep(delay)

    print("\nNow, in each tab: click the publisher's 'Download PDF' button.")
    print("Your institutional login already applies, so it should just download.")
    print(f"Then run:  python get_papers.py import")
    return 0


def pdf_text_head(p: Path, pages: int = 2) -> str:
    try:
        r = subprocess.run(["pdftotext", "-l", str(pages), str(p), "-"],
                           capture_output=True, timeout=90)
        return (r.stdout or b"").decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def cmd_import(since_min: float) -> int:
    dl = downloads_dir()
    if not dl.exists():
        print(f"Downloads folder not found: {dl}")
        return 1
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(LIST_CSV)
    dois = [str(x) for x in d.doi.dropna()]

    cutoff = time.time() - since_min * 60
    # Chrome drops files either in Downloads or in data/pdfs/inbox (the folder the selenium route
    # configures), so both are scanned -- a download that landed in the "wrong" one is still yours.
    inbox = PDF_DIR / "inbox"
    sources = [dl] + ([inbox] if inbox.exists() else [])
    cands = sorted({p for src in sources for p in src.glob("*.pdf")
                    if p.stat().st_mtime >= cutoff})
    for src in sources:
        print(f"scanning {src}")
    print(f"  PDFs modified in the last {since_min:.0f} min: {len(cands)}\n")
    if not cands:
        print("  nothing new. Did the downloads finish?")
        return 0

    moved, unmatched = [], []
    for p in cands:
        txt = pdf_text_head(p)
        hit = None
        for doi in dois:
            # DOIs appear on the first page of virtually every publisher PDF
            if doi.lower() in txt.lower().replace(" ", ""):
                hit = doi
                break
            tail = doi.split("/")[-1]
            if len(tail) > 8 and tail.lower() in txt.lower():
                hit = doi
                break
        if not hit:
            unmatched.append(p)
            continue
        dest = PDF_DIR / f"{safe(hit)}.pdf"
        if dest.exists():
            print(f"  already have {hit}")
            continue
        shutil.copy2(p, dest)
        moved.append({"doi": hit, "file": dest.name, "src": p.name,
                      "bytes": dest.stat().st_size})
        print(f"  OK   {hit:<34}-> {dest.name}")

    if unmatched:
        print(f"\n  {len(unmatched)} PDF(s) could not be matched to a queued DOI "
              f"(left in Downloads, nothing guessed):")
        for p in unmatched[:8]:
            print(f"       {p.name[:70]}")

    if moved:
        m = pd.DataFrame(moved)
        if LOG.exists():
            m = pd.concat([pd.read_csv(LOG), m], ignore_index=True).drop_duplicates("doi")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(LOG, index=False)
        print(f"\n  imported {len(moved)} papers -> {PDF_DIR}")
        print(f"  log: {LOG}")
        print(f"\nNext: python run_extraction.py --budget-min 30")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open", help="open article pages in your normal browser")
    o.add_argument("--limit", type=int, default=5)
    o.add_argument("--publisher")
    o.add_argument("--delay", type=float, default=2.0)
    i = sub.add_parser("import", help="file downloaded PDFs into data/pdfs, named by DOI")
    i.add_argument("--since-min", type=float, default=120.0,
                   help="only consider PDFs downloaded in the last N minutes")
    a = ap.parse_args()
    if a.cmd == "open":
        raise SystemExit(cmd_open(a.limit, a.publisher, a.delay))
    raise SystemExit(cmd_import(a.since_min))
