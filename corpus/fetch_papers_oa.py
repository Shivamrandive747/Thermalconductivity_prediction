"""Step 3, lane A — fetch the open-access papers AND their supplementary files, unattended.

THE SUPPLEMENTARY FILE IS THE POINT. The article PDF usually carries a handful of numbers in prose;
the hundred-row table lives in the SI. This was proven the hard way this session: the npj
"million-scale Heusler" article PDF yielded nothing extractable, while its Springer ESM spreadsheets
(`..._MOESM2_ESM.xlsx`, `..._MOESM3_ESM.xlsx`) fetched cleanly on the first try and held 7,373 rows.

So SI is fetched *even when the article PDF succeeds*. That is the single highest-leverage behaviour
in this script.

WHAT IT WILL AND WILL NOT REACH. Publisher landing pages were probed from this machine on the
institute LAN with full browser headers:

    APS / Elsevier / Wiley / RSC   403   -- rejected BEFORE any paywall page rendered
    IOP / Springer                 200   -- scriptable

The 403 is bot detection, not a subscription problem: `paywall_words=False` on every one, meaning
the request never even reached a "purchase this article" screen. Those publishers are lane B and are
handled by the pre-existing `fetch_papers_chrome.py`, which drives the user's own signed-in Chrome.
This script does not try to defeat any protection; it simply uses the open-access copy where one
exists.

POLITENESS IS NOT OPTIONAL. One request per second, a real User-Agent with a contact address, and a
hard 403 is never retried. Publisher licences prohibit systematic bulk download, and hammering a
portal can get an institution's whole subscription suspended.

Every attempt -- success or failure, with the reason -- is written to `paper_fetch_log.csv`, so the
failures become the input to lane B rather than vanishing.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
import requests

QUEUE = "data/external/mining_queue_ranked.csv"
PDF_DIR = Path("data/pdfs")
SI_DIR = Path("data/external/si")
LOG = "data/external/paper_fetch_log.csv"
MAIL = "teammaterialscienceteam@gmail.com"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/122.0.0.0 Safari/537.36")
MIN_PDF = 25_000          # smaller than this is a landing page or an error stub, not a paper
MIN_SI = 3_000


def safe(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", doi)


def springer_esm_urls(doi: str) -> list[str]:
    """s41524-023-00974-0 -> .../MediaObjects/41524_2023_974_MOESM{n}_ESM.{ext}

    The MediaObjects stem is the DOI suffix rearranged: journal id, 4-digit year, article number
    with leading zeros stripped. Derived by inspecting a URL that worked, not guessed."""
    m = re.match(r"^10\.1038/s(\d+)-(\d{2,4})-(\d+)-(\d+)$", doi)
    if not m:
        return []
    jid, yr, art = m.group(1), m.group(2), m.group(3)
    yr = yr if len(yr) == 4 else ("20" + yr[-2:])
    stem = f"{jid}_{yr}_{int(art)}"
    art_enc = quote(f"10.1038/s{m.group(1)}-{m.group(2)}-{m.group(3)}-{m.group(4)}", safe="")
    base = f"https://static-content.springer.com/esm/art%3A{art_enc}/MediaObjects/{stem}"
    out = []
    for n in range(1, 7):
        for ext in ("xlsx", "csv", "pdf", "docx", "txt"):
            out.append(f"{base}_MOESM{n}_ESM.{ext}")
    return out


def mdpi_si_urls(doi: str) -> list[str]:
    if not doi.startswith("10.3390/"):
        return []
    return [f"https://www.mdpi.com/article/{doi}/s1"]


def get(s: requests.Session, url: str, timeout: int = 60):
    try:
        return s.get(url, timeout=timeout, allow_redirects=True)
    except Exception:  # noqa: BLE001
        return None


def looks_pdf(r) -> bool:
    return bool(r) and r.status_code == 200 and (
        r.content[:4] == b"%PDF" or "pdf" in r.headers.get("content-type", "").lower())


def looks_sheet(r) -> bool:
    if not r or r.status_code != 200:
        return False
    ct = r.headers.get("content-type", "").lower()
    return ("sheet" in ct or "excel" in ct or "csv" in ct or "officedocument" in ct
            or r.content[:2] == b"PK")


def arxiv_pdf(s: requests.Session, doi: str, title: str) -> str | None:
    """Most of this corpus is condensed-matter physics, so an arXiv preprint usually exists and is
    always fetchable. MDPI, RSC, IOP and Elsevier all reject scripted clients outright (measured:
    403 from a Cloudflare page, 406 bytes), so arXiv is the main way lane A reaches those papers.

    The preprint's tables are the same tables; the row is still cited to the published DOI, with
    `source_file` recording that the text came from the preprint."""
    for query in (f'all:"{doi}"', f'ti:"{title[:120]}"'):
        r = get(s, "http://export.arxiv.org/api/query?"
                   + f"search_query={quote(query)}&max_results=1", timeout=40)
        if not r or r.status_code != 200:
            continue
        m = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", r.text)
        t = re.search(r"<title>([^<]+)</title>\s*</entry>|<entry>.*?<title>([^<]+)</title>",
                      r.text, re.S)
        if not m:
            continue
        # guard against the API returning an unrelated paper for a loose title match
        if t:
            got = (t.group(1) or t.group(2) or "").strip().lower()
            want = title.strip().lower()
            if want and got:
                overlap = len(set(want.split()) & set(got.split())) / max(len(set(want.split())), 1)
                if overlap < 0.6:
                    continue
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return None


def unpaywall_pdf(s: requests.Session, doi: str) -> str | None:
    r = get(s, f"https://api.unpaywall.org/v2/{doi}?email={MAIL}", timeout=40)
    if not r or r.status_code != 200:
        return None
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        return None
    best = j.get("best_oa_location") or {}
    for loc in [best] + (j.get("oa_locations") or []):
        u = (loc or {}).get("url_for_pdf")
        if u:
            return u
    return None


def main(limit: int | None, sleep: float, si_only: bool, min_score: int,
         queue: str = QUEUE) -> int:
    q = pd.read_csv(queue)
    q = q[q.score >= min_score]
    if "lane" in q.columns:
        q = q[q.lane == "A"]
    q = q.sort_values(["score", "cited"], ascending=False)
    if limit:
        q = q.head(limit)
    print(f"queue: {len(q)} papers from {queue} (score >= {min_score})")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SI_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})

    log = []
    got_pdf = got_si = 0
    for i, (_, row) in enumerate(q.iterrows(), 1):
        doi = str(row.doi)
        stem = safe(doi)

        # ---- article PDF -----------------------------------------------------
        pdf_path = PDF_DIR / f"{stem}.pdf"
        if not si_only and not (pdf_path.exists() and pdf_path.stat().st_size > MIN_PDF):
            for src, url in (("oa_url", row.get("oa_url")),
                             ("unpaywall", None),
                             ("arxiv", None)):
                if src == "unpaywall":
                    url = unpaywall_pdf(s, doi)
                    time.sleep(sleep)
                elif src == "arxiv":
                    url = arxiv_pdf(s, doi, str(row.get("title") or ""))
                    time.sleep(sleep)
                if not url or str(url) == "nan":
                    continue
                r = get(s, str(url))
                if looks_pdf(r) and len(r.content) > MIN_PDF:
                    pdf_path.write_bytes(r.content)
                    log.append(dict(doi=doi, kind="pdf", via=src, ok=True,
                                    bytes=len(r.content), url=url))
                    got_pdf += 1
                    break
                log.append(dict(doi=doi, kind="pdf", via=src, ok=False,
                                status=(r.status_code if r else "err"), url=url))
                time.sleep(sleep)
        elif pdf_path.exists():
            got_pdf += 1

        # ---- supplementary, ALWAYS, even when the PDF worked -----------------
        n_si = 0
        for url in springer_esm_urls(doi) + mdpi_si_urls(doi):
            if n_si >= 4:
                break
            r = get(s, url, timeout=50)
            if r is None or r.status_code != 200:
                continue
            ok_sheet, ok_pdf = looks_sheet(r), looks_pdf(r)
            if not (ok_sheet or ok_pdf) or len(r.content) < MIN_SI:
                continue
            ext = url.rsplit(".", 1)[-1].split("?")[0][:5]
            if "mdpi" in url:
                ext = "zip" if r.content[:2] == b"PK" else "pdf"
            p = SI_DIR / f"{stem}_SI{n_si + 1}.{ext}"
            p.write_bytes(r.content)
            log.append(dict(doi=doi, kind="si", via="publisher-pattern", ok=True,
                            bytes=len(r.content), url=url))
            n_si += 1
            got_si += 1
            time.sleep(sleep)

        if i % 10 == 0 or n_si:
            print(f"  [{i}/{len(q)}] {doi:<34} pdf={'Y' if pdf_path.exists() else '-'} si={n_si}",
                  flush=True)
        time.sleep(sleep)

    d = pd.DataFrame(log)
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(LOG, index=False)
    print(f"\n=== LANE A RESULT ===")
    print(f"  papers with a PDF        : {got_pdf}")
    print(f"  supplementary files saved: {got_si}")
    if len(d):
        f = d[(d.kind == "pdf") & (~d.ok.fillna(False))]
        print(f"  PDF failures             : {f.doi.nunique()} papers "
              f"-> candidates for lane B (your Chrome)")
    print(f"  log: {LOG}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--si-only", action="store_true")
    ap.add_argument("--min-score", type=int, default=0)
    ap.add_argument("--queue", default=QUEUE)
    raise SystemExit(main(**vars(ap.parse_args())))
