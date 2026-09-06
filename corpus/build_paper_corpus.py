"""Step 2 — build the ranked corpus of papers that might carry Heusler kappa_L tables.

WHY A WIDE SWEEP. A single query set once returned 6 usable papers and led me to conclude the
literature was exhausted. That was a sizing error, not a finding. Fifteen query variants against
OpenAlex return ~1,049 distinct DOIs, of which ~250 name both a Heusler and a thermal property in
the title and ~195 are open access. The corpus is real; the first search was just too narrow.

WHAT THIS SCRIPT DOES NOT DO. It does not decide whether a paper contains usable data -- only a
human or the extractor can. It produces a RANKED QUEUE, so the expensive step (fetching and
model-extraction at ~$0.24/paper) is spent on the papers most likely to carry a big table.

THE RANKING IS ABOUT TABLE SIZE, NOT QUALITY. One high-throughput screening paper with 100
compounds is worth thirty single-compound DFT studies, and costs the same to extract. So the score
rewards words that signal a many-row table -- "high-throughput", "screening", "database",
"descriptor", "systematic" -- and open access, because a paper we cannot fetch is worth nothing.

TWO LANES, DECIDED BY MEASUREMENT. Publisher landing pages were probed from this machine on the
institute LAN with full browser headers:

    APS / Elsevier / Wiley / RSC  -> 403 before any paywall page rendered  (bot detection)
    IOP / Springer                -> 200                                   (scriptable)

The 403s are NOT a subscription problem -- the request never reaches a paywall. So each row is
tagged `lane`: A for anything we can fetch unattended, B for the publishers that need the user's own
signed-in Chrome via the existing `fetch_papers_chrome.py`.

DOIs already mined are excluded, so a rerun does not re-queue work already done.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
import requests

from pipeline.config import cfg  # noqa: F401  (loads .env -> OPENALEX_API)

OUT = "data/external/mining_queue_ranked.csv"
MANUAL = "data/external/MANUAL_DOWNLOAD_LIST.csv"
MAIL = "teammaterialscienceteam@gmail.com"

# Files whose `source_doi` column records what we have already extracted.
ALREADY = ["data/external/kappa_papers_mined.csv", "data/external/kappa_repos_mined.csv",
           "data/external/kappa_bte_heusler_MASTER.csv", "data/external/kappa_pdf_batch2.csv",
           "data/external/KAPPA_POOL_MASTER.csv"]

QUERIES = [
    'half-Heusler "lattice thermal conductivity" high-throughput screening',
    'Heusler thermoelectric "lattice thermal conductivity" first-principles screening',
    '"half-Heusler" phonon "thermal conductivity" ShengBTE',
    '"half-Heusler" Phono3py "thermal conductivity"',
    'quaternary Heusler "lattice thermal conductivity" DFT',
    '"full Heusler" phonon thermal transport ab initio',
    'Heusler "thermal conductivity" Slack model Debye',
    'half-Heusler thermoelectric "figure of merit" "thermal conductivity" first principles',
    '"XYZ half-Heusler" thermoelectric screening thermal',
    'Heusler alloys "phonon transport" anharmonic',
    '18-electron half-Heusler semiconductor thermal conductivity',
    'Heusler "lattice thermal conductivity" machine learning dataset',
    'half-Heusler "thermal conductivity" Boltzmann transport equation compounds',
    'Heusler C1b "thermal conductivity" ab initio',
    'Heusler L21 "thermal conductivity" first-principles',
]

HEUSLER = re.compile(r"heusler|half.?heusler|\bC1b\b|\bL2\s?1\b|MgAgAs", re.I)
THERMAL = re.compile(r"thermal conduct|phonon|thermal transport|lattice dynam|thermoelectric", re.I)
BIGTABLE = re.compile(r"high.?throughput|screening|database|dataset|descriptor|machine learn|"
                      r"systematic|survey|discovery|catalog|compounds|series|family", re.I)
COMPUTED = re.compile(r"first.?principles|ab initio|dft|computation|phonon|boltzmann|shengbte|"
                      r"phono3py|anharmonic|lattice dynam", re.I)

# Publishers whose sites reject scripted clients outright -> user's Chrome (lane B).
LANE_B = re.compile(r"^10\.(1103|1016|1002|1039|1021|1063|1149)/", re.I)


def already_mined() -> set[str]:
    have = set()
    for f in ALREADY:
        p = Path(f)
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=lambda c: c == "source_doi")
        except Exception:  # noqa: BLE001
            continue
        if "source_doi" in d.columns:
            have |= {str(x).lower().strip() for x in d.source_doi.dropna()}
    return have


def main(pages: int = 2, per_page: int = 100) -> int:
    key = os.getenv("OPENALEX_API")
    s = requests.Session()
    s.headers["User-Agent"] = f"heusler-kappa-mining/1.0 (mailto:{MAIL})"
    print(f"OpenAlex key: {'yes' if key else 'no (anonymous, will be slower)'}")

    rows = []
    for q in QUERIES:
        for page in range(1, pages + 1):
            p = {"search": q, "per-page": per_page, "page": page,
                 "filter": "from_publication_date:2010-01-01,type:article"}
            if key:
                p["api_key"] = key
            try:
                r = s.get("https://api.openalex.org/works", params=p, timeout=60)
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code} on '{q[:34]}' page {page}")
                    break
                res = r.json().get("results", [])
            except Exception as exc:  # noqa: BLE001
                print(f"    ERR {str(exc)[:50]}")
                break
            if not res:
                break
            for w in res:
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower().strip()
                if not doi:
                    continue
                oa = w.get("open_access") or {}
                loc = w.get("primary_location") or {}
                src = (loc.get("source") or {})
                rows.append(dict(
                    doi=doi, title=w.get("title") or "", year=w.get("publication_year"),
                    cited=w.get("cited_by_count") or 0,
                    is_oa=bool(oa.get("is_oa")), oa_url=oa.get("oa_url"),
                    journal=src.get("display_name"), query=q))
            time.sleep(0.3)
        print(f"  {q[:52]:<54} cum {len(rows)}")

    d = pd.DataFrame(rows).drop_duplicates("doi")
    print(f"\n  unique DOIs: {len(d)}")

    have = already_mined()
    d["already_mined"] = d.doi.isin(have)
    print(f"  already mined, excluded: {int(d.already_mined.sum())}")
    d = d[~d.already_mined]

    d["heusler"] = d.title.str.contains(HEUSLER, na=False)
    d["thermal"] = d.title.str.contains(THERMAL, na=False)
    d["bigtable"] = d.title.str.contains(BIGTABLE, na=False)
    d["computed"] = d.title.str.contains(COMPUTED, na=False)
    d["relevant"] = d.heusler & d.thermal

    d["score"] = (d.bigtable.astype(int) * 3 + d.computed.astype(int) * 2
                  + d.is_oa.astype(int) * 2 + (d.cited > 50).astype(int))
    d["lane"] = ["A" if (o or not LANE_B.match(x)) else "B"
                 for o, x in zip(d.is_oa, d.doi)]

    rel = d[d.relevant].sort_values(["score", "cited"], ascending=False)
    print(f"\n  Heusler AND thermal in title : {len(rel)}")
    print(f"    open access                : {int(rel.is_oa.sum())}")
    print(f"    likely big tables          : {int(rel.bigtable.sum())}")
    print(f"    lane A (scriptable)        : {int((rel.lane == 'A').sum())}")
    print(f"    lane B (needs your Chrome) : {int((rel.lane == 'B').sum())}")

    print(f"\n  TOP 15 BY EXPECTED YIELD:")
    for _, r in rel.head(15).iterrows():
        print(f"    [{r.score}] {r.lane}  {r.doi:<34}{str(r.title)[:62]}")

    Path("data/external").mkdir(parents=True, exist_ok=True)
    rel.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(rel)} papers)")

    # The blocked ones, in the shape `fetch_papers_chrome.py` already reads.
    b = rel[rel.lane == "B"].copy()
    if len(b):
        b["url"] = "https://doi.org/" + b.doi
        b[["doi", "url", "title", "journal", "year", "score"]].to_csv(MANUAL, index=False)
        print(f"wrote {MANUAL}  ({len(b)} papers for the browser lane)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--per-page", type=int, default=100)
    raise SystemExit(main(**vars(ap.parse_args())))
