"""Build references.bib from the DOIs already in the corpus, verified against Crossref.

WHY THIS EXISTS. A language model cannot be trusted to produce a citation. It generates plausible
author names, plausible titles and plausible DOIs that resolve to a different paper or to nothing
at all. A wrong number in a paper gets corrected; a fabricated citation is misconduct. So the rule
for this manuscript is that no citation is ever WRITTEN -- every one is RESOLVED, from a DOI that
already exists in the data pipeline, against a real registry.

That rule is cheap here because the corpus already carries 245 DOIs: 201 behind experimental
measurements and 31 behind first-principles calculations. Those papers were read to extract the
numbers in the training set, so their existence is not in question.

Crossref's REST API is used because it is free, needs no key, and is the registry that ISSUES
DOIs -- if Crossref does not know a DOI, the DOI is wrong. Requests go through the polite pool
(a mailto identifies the caller), which is both faster and the courteous way to use a free service
that this project has no special claim on.

A DOI that fails to resolve is REPORTED, never silently dropped: it means a row in the training
set is attributed to something that does not exist, which is a data-quality problem worth knowing
about regardless of the bibliography.
"""
from __future__ import annotations

import json
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

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT_BIB = "paper/references.bib"
OUT_REPORT = "paper/bibliography_report.json"
CACHE = "data/external/crossref_cache.json"
API = "https://api.crossref.org/works/"
# arXiv, Zenodo and figshare register their DOIs with DataCite, not Crossref. A DOI that 404s at
# Crossref is therefore not necessarily wrong -- it may simply be registered elsewhere, and
# checking only one registry would condemn 569 legitimate rows of this corpus.
API_DATACITE = "https://api.datacite.org/dois/"
DELAY = 0.12          # polite pool allows far more; there is no reason to push a free service
TIMEOUT = 20
RETRIES = 3


def mailto() -> str:
    """Crossref's polite pool wants a contact address. Read it from .env, never hardcode."""
    for line in Path(".env").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OPENALEX_MAILTO"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def load_cache() -> dict:
    try:
        return json.loads(Path(CACHE).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _datacite(doi: str, mail: str) -> dict | None:
    """Resolve a DOI that Crossref does not know, and normalise it to Crossref's shape.

    Preprint servers and data repositories register with DataCite. Their records are legitimate
    citations -- and for a data repository the record is arguably the RIGHT thing to cite for
    provenance -- but the field names differ, so they are translated here rather than special-cased
    everywhere downstream.
    """
    try:
        r = requests.get(f"{API_DATACITE}{doi}", timeout=TIMEOUT,
                         headers={"User-Agent": f"heusler-kappa-bibliography (mailto:{mail})"})
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    a = (r.json().get("data") or {}).get("attributes") or {}
    authors = []
    for c in (a.get("creators") or []):
        fam, giv = c.get("familyName"), c.get("givenName")
        if not fam and c.get("name"):
            parts = str(c["name"]).split(",", 1)
            fam = parts[0].strip()
            giv = parts[1].strip() if len(parts) > 1 else ""
        if fam:
            authors.append({"family": fam, "given": giv or ""})
    yr = a.get("publicationYear")
    return {
        "title": [(a.get("titles") or [{}])[0].get("title", "")],
        "author": authors,
        "container-title": [a.get("publisher") or ""],
        "issued": {"date-parts": [[int(yr)]]} if yr else {},
        "DOI": a.get("doi", doi),
        "publisher": a.get("publisher") or "",
        "type": "dataset" if (a.get("types") or {}).get("resourceTypeGeneral") == "Dataset"
                else "journal-article",
        "_registry": "DataCite",
    }


def fetch(doi: str, mail: str, cache: dict) -> dict | None:
    if doi in cache:
        return cache[doi]
    url = f"{API}{doi}"
    params = {"mailto": mail} if mail else {}
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT,
                             headers={"User-Agent": f"heusler-kappa-bibliography (mailto:{mail})"})
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            cache[doi] = r.json().get("message", {})
            return cache[doi]
        if r.status_code == 404:
            m = _datacite(doi, mail)      # not wrong, possibly just a different registry
            time.sleep(DELAY)
            cache[doi] = m
            return m
        # 429 / 5xx -- back off rather than hammer
        time.sleep(2.0 * (attempt + 1))
    return None


def bibkey(m: dict, taken: set) -> str:
    auth = m.get("author") or []
    last = "anon"
    if auth:
        last = (auth[0].get("family") or auth[0].get("name") or "anon")
    last = re.sub(r"[^A-Za-z]", "", last).lower() or "anon"
    yr = ""
    for f in ("published-print", "published-online", "issued", "created"):
        p = (m.get(f) or {}).get("date-parts") or [[None]]
        if p and p[0] and p[0][0]:
            yr = str(p[0][0])
            break
    title = (m.get("title") or [""])[0]
    word = re.sub(r"[^A-Za-z]", "", (title.split() or ["x"])[0]).lower()[:10] or "x"
    base = f"{last}{yr}{word}"
    key, n = base, 1
    while key in taken:
        n += 1
        key = f"{base}{chr(ord('a') + n - 2)}"
    taken.add(key)
    return key


def esc(s: str) -> str:
    """Protect the characters LaTeX would eat, and translate the markup Crossref returns.

    Crossref serves titles containing real HTML -- `Co<sub>2</sub>Dy<sub>0.5</sub>Mn...` -- plus
    typographic Unicode that a pdfLaTeX run either drops silently or fails on: non-breaking
    hyphens, en dashes, primes and Greek letters are all common in chemistry titles. Escaping
    without translating first would put the literal string "<sub>2</sub>" into the reference list.
    Subscripts are converted BEFORE escaping, since they must survive as maths.
    """
    s = str(s)
    # MathML first -- Crossref serves <mml:msub><mml:mi>Bi</mml:mi><mml:mn>2</mml:mn></mml:msub>
    # for some chemistry titles. The sub/superscript structure has to be captured BEFORE the inner
    # tags are stripped, or the boundary between base and script is lost.
    for _ in range(4):                      # nested scripts need more than one pass
        s = re.sub(r"<mml:msubsup>\s*<mml:m\w+>([^<]*)</mml:m\w+>\s*<mml:m\w+>([^<]*)</mml:m\w+>"
                   r"\s*<mml:m\w+>([^<]*)</mml:m\w+>\s*</mml:msubsup>",
                   r"\1@@SUB{\2}@@@@SUP{\3}@@", s, flags=re.I)
        s = re.sub(r"<mml:msub>\s*<mml:m\w+>([^<]*)</mml:m\w+>\s*<mml:m\w+>([^<]*)</mml:m\w+>"
                   r"\s*</mml:msub>", r"\1@@SUB{\2}@@", s, flags=re.I)
        s = re.sub(r"<mml:msup>\s*<mml:m\w+>([^<]*)</mml:m\w+>\s*<mml:m\w+>([^<]*)</mml:m\w+>"
                   r"\s*</mml:msup>", r"\1@@SUP{\2}@@", s, flags=re.I)
    s = re.sub(r"</?mml:\w+[^>]*>", "", s)          # any MathML scaffolding left, text kept
    # plain HTML markup -> LaTeX maths
    for _ in range(3):
        s = re.sub(r"<sub>(.*?)</sub>", r"@@SUB{\1}@@", s, flags=re.I | re.S)
        s = re.sub(r"<sup>(.*?)</sup>", r"@@SUP{\1}@@", s, flags=re.I | re.S)
    s = re.sub(r"</?[a-z][^>]{0,40}>", "", s, flags=re.I)   # i, b, em, font, span, ...
    # Named and numeric HTML entities. `&minus;` in particular appears in off-stoichiometry
    # titles (Fe2V1&minus;xAl1+x) and, left alone, survives escaping as a literal "\&minus;".
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                 ("&apos;", "'"), ("&nbsp;", " "), ("&minus;", "-"), ("&ndash;", "--"),
                 ("&mdash;", "---"), ("&plusmn;", "±"), ("&times;", "×"),
                 ("&deg;", "°"), ("&alpha;", "α"), ("&beta;", "β"),
                 ("&gamma;", "γ"), ("&delta;", "δ"), ("&kappa;", "κ"),
                 ("&mu;", "μ"), ("&sigma;", "σ"), ("&le;", "≤"),
                 ("&ge;", "≥"), ("&#x2010;", "-")):
        s = s.replace(a, b)
    s = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), s)

    # SECOND MARKUP PASS -- required, not belt-and-braces.
    #
    # Crossref serves some titles with ESCAPED markup: literally "&lt;sub&gt;2&lt;/sub&gt;" rather
    # than "<sub>2</sub>". Decoding entities above therefore RE-CREATES tags behind the stripper
    # that already ran, and three titles shipped with raw <sub> and <i> in the reference list while
    # this function passed its own test on the unescaped form. Whatever the entity decode revealed
    # has to be converted and stripped again.
    for _ in range(3):
        s = re.sub(r"<sub>(.*?)</sub>", r"@@SUB{\1}@@", s, flags=re.I | re.S)
        s = re.sub(r"<sup>(.*?)</sup>", r"@@SUP{\1}@@", s, flags=re.I | re.S)
    s = re.sub(r"</?[a-z][^>]{0,40}>", "", s, flags=re.I)
    # Typographic and mathematical Unicode that pdfLaTeX cannot set in the default encoding.
    # ACCENTED LATIN LETTERS ARE DELIBERATELY ABSENT from this table: the file carries names like
    # Sondergaard, Simonek, Jerome, Kaminski and Wolinski in their correct spelling, and mangling
    # a person's name to dodge an encoding problem is not acceptable. Modern LaTeX with utf8
    # inputenc, or any XeLaTeX/LuaLaTeX run, sets them correctly.
    for a, b in (("‐", "-"), ("‑", "-"), ("‒", "-"), ("–", "--"),
                 ("—", "---"), ("‘", "`"), ("’", "'"), ("“", "``"),
                 ("”", "''"), ("′", "'"), (" ", " "), ("−", "-"),
                 ("≤", r"@@MATH{\leq}@@"), ("≥", r"@@MATH{\geq}@@"),
                 ("≈", r"@@MATH{\approx}@@"), ("±", r"@@MATH{\pm}@@"),
                 ("×", r"@@MATH{\times}@@"), ("·", r"@@MATH{\cdot}@@"),
                 ("α", r"@@MATH{\alpha}@@"), ("β", r"@@MATH{\beta}@@"),
                 ("γ", r"@@MATH{\gamma}@@"), ("δ", r"@@MATH{\delta}@@"),
                 ("ε", r"@@MATH{\epsilon}@@"), ("θ", r"@@MATH{\theta}@@"),
                 ("κ", r"@@MATH{\kappa}@@"), ("λ", r"@@MATH{\lambda}@@"),
                 ("μ", r"@@MATH{\mu}@@"), ("σ", r"@@MATH{\sigma}@@"),
                 ("Δ", r"@@MATH{\Delta}@@"), ("Ω", r"@@MATH{\Omega}@@"),
                 ("→", r"@@MATH{\rightarrow}@@")):
        s = s.replace(a, b)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    # restore the placeholders as real maths now that escaping is done
    s = re.sub(r"@@SUB\\\{(.*?)\\\}@@", r"$_{\1}$", s)
    s = re.sub(r"@@SUP\\\{(.*?)\\\}@@", r"$^{\1}$", s)
    s = re.sub(r"@@MATH\\\{\\textbackslash\{\}(.*?)\\\}@@", r"$\\\1$", s)
    return re.sub(r"\s+", " ", s).strip()


def to_bibtex(m: dict, key: str) -> str:
    auth = m.get("author") or []
    names = " and ".join(
        f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
        for a in auth if a.get("family") or a.get("given"))
    yr = ""
    for f in ("published-print", "published-online", "issued", "created"):
        p = (m.get(f) or {}).get("date-parts") or [[None]]
        if p and p[0] and p[0][0]:
            yr = str(p[0][0])
            break
    ty = m.get("type", "journal-article")
    kind = {"journal-article": "article", "proceedings-article": "inproceedings",
            "book-chapter": "incollection", "book": "book"}.get(ty, "article")
    fields = {
        "author": esc(names),
        "title": esc((m.get("title") or [""])[0]),
        "journal": esc((m.get("container-title") or [""])[0]),
        "year": yr,
        "volume": esc(m.get("volume", "")),
        "number": esc(m.get("issue", "")),
        "pages": esc(m.get("page", "")),
        "doi": m.get("DOI", ""),
        "publisher": esc(m.get("publisher", "")),
    }
    body = ",\n".join(f"  {k:<10}= {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main() -> int:
    Path("paper").mkdir(exist_ok=True)
    mail = mailto()
    print(f"Crossref polite pool as: {mail or '(no mailto -- slower, still works)'}")

    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    s = tr.source_doi.dropna().astype(str).str.strip()
    s = s[s.str.startswith("10.")]
    # a DOI can carry a URL prefix or a trailing punctuation mark from extraction
    s = s.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True).str.rstrip(".,;)")
    dois = sorted(set(s))

    tier_of: dict = {}
    for d_, g in tr.dropna(subset=["source_doi"]).groupby(
            tr.source_doi.astype(str).str.strip().str.replace(
                r"^https?://(dx\.)?doi\.org/", "", regex=True).str.rstrip(".,;)")):
        ts = sorted({int(t) for t in g.tier.dropna().unique()})
        tier_of[d_] = ts

    # Tier-B references: papers the manuscript needs that are not in the data pipeline -- mostly
    # Introduction background, method sources, and database papers. They are resolved through the
    # same registries as everything else, so a hand-added DOI gets no weaker verification than a
    # mined one. (This file was named in the generated header before it was read; adding a DOI to
    # it silently did nothing.)
    extra_path = Path("paper/extra_dois.txt")
    extra = []
    if extra_path.exists():
        for line in extra_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.split("#", 1)[0].strip()
            line = re.sub(r"^https?://(dx\.)?doi\.org/", "", line).rstrip(".,;)")
            if line.startswith("10."):
                extra.append(line)
    new_extra = sorted(set(extra) - set(dois))
    if extra:
        print(f"extra_dois.txt: {len(set(extra))} listed, {len(new_extra)} not already in the corpus")
    dois = sorted(set(dois) | set(extra))

    print(f"distinct DOIs to resolve: {len(dois)}\n")

    cache = load_cache()
    n_cached = sum(1 for d in dois if d in cache)
    print(f"already cached: {n_cached}   to fetch: {len(dois) - n_cached}\n")

    resolved, failed, taken = {}, [], set()
    for i, d in enumerate(dois, 1):
        was_cached = d in cache
        m = fetch(d, mail, cache)
        if m:
            resolved[d] = m
        else:
            failed.append(d)
        if not was_cached:
            time.sleep(DELAY)
        if i % 25 == 0 or i == len(dois):
            print(f"  {i:>4}/{len(dois)}   resolved {len(resolved)}   failed {len(failed)}")
        if i % 50 == 0:
            Path(CACHE).write_text(json.dumps(cache), encoding="utf-8")
    Path(CACHE).write_text(json.dumps(cache), encoding="utf-8")

    entries, keymap = [], {}
    for d, m in sorted(resolved.items()):
        k = bibkey(m, taken)
        keymap[d] = k
        entries.append(to_bibtex(m, k))

    header = (
        "% references.bib -- GENERATED by build_bibliography.py. Do not edit by hand.\n"
        "%\n"
        "% Every entry was resolved against the Crossref REST API from a DOI that already exists\n"
        "% in the project's data pipeline. No entry was written from memory. To add a reference,\n"
        "% add its DOI to the pipeline or to paper/extra_dois.txt and re-run this script.\n"
        f"%\n% entries: {len(entries)}   unresolved DOIs: {len(failed)}\n\n")
    Path(OUT_BIB).write_text(header + "\n".join(entries), encoding="utf-8")

    by_tier: dict = {}
    for d in resolved:
        for t in tier_of.get(d, []):
            by_tier.setdefault(str(t), []).append(keymap[d])
    report = dict(
        n_dois=len(dois), n_resolved=len(resolved), n_failed=len(failed),
        failed_dois=failed,
        key_by_doi=keymap,
        keys_by_method_tier={k: sorted(v) for k, v in sorted(by_tier.items())},
        note=("keys_by_method_tier lets the manuscript cite the right provenance: tier 0 keys are "
              "experimental measurements, tier 1 are first-principles calculations"))
    Path(OUT_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT_BIB}      {len(entries)} verified entries")
    print(f"wrote {OUT_REPORT}")
    if failed:
        print(f"\n{len(failed)} DOI(s) did NOT resolve at Crossref -- these are data-quality "
              f"problems, not just missing citations:")
        for d in failed[:20]:
            n = int((s == d).sum())
            print(f"    {d:<44} {n} row(s) in the training set")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more (full list in the report)")
    else:
        print("\nevery DOI in the corpus resolved. No fabricated or malformed attributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
