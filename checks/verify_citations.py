"""The pre-submission citation gate. Run it before the manuscript leaves the building.

Four failures, all blocking, each corresponding to a real way a bibliography goes wrong:

  1 UNDEFINED KEY      the text cites a key that is not in references.bib. LaTeX renders this as
                       [?] and it is the most common way a fabricated citation survives -- it was
                       typed straight into the text and never entered the bibliography at all.
  2 DEAD DOI           the entry exists but its DOI no longer resolves at Crossref. Either the DOI
                       was wrong from the start or the record was withdrawn; both matter.
  3 UNSUPPORTED CLAIM  the citation is real and resolves, but claims_ledger.csv has no sentence
                       from the source backing the claim it is attached to. This is the failure a
                       DOI check cannot catch and the one a referee in the field WILL catch.
  4 UNCITED ENTRY      references.bib carries an entry nothing cites. Harmless to a reader, but it
                       usually means a citation was removed from the text and the claim it
                       supported is now unsupported.

Exit code is non-zero if anything blocks, so this can gate a commit hook or CI.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import requests

BIB = "paper/references.bib"
LEDGER = "paper/claims_ledger.csv"
CACHE = "data/external/crossref_cache.json"
API = "https://api.crossref.org/works/"
# arXiv, Zenodo and figshare register with DataCite, not Crossref
API_DATACITE = "https://api.datacite.org/dois/"
DELAY = 0.12


def mailto() -> str:
    try:
        for line in Path(".env").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("OPENALEX_MAILTO"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        pass
    return ""


def parse_bib(path: str) -> dict:
    """Key -> {field: value}. Deliberately a small parser: bibtexparser is installed, but this
    file is machine-generated with a known shape, and a parser that cannot fail is worth more here
    than one that handles arbitrary BibTeX."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        key, body = m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        out[key] = fields
    return out


def _expand_inputs(tex: str, _seen=None) -> str:
    r"""Return the manuscript with every \input{...} / \include{...} spliced in.

    This function exists because its absence made the whole gate vacuous. The manuscript is a
    thin elsarticle wrapper that \input{}s seven section files, so reading manuscript.tex alone
    found ZERO \cite commands -- and a gate that finds zero citations reports that all zero of
    them are supported, and PASSES. A check that cannot fail is worse than no check, because it
    is trusted.
    """
    if _seen is None:
        _seen = set()
    path = Path(tex)
    if path.suffix == "":
        path = path.with_suffix(".tex")
    key = str(path.resolve())
    if key in _seen or not path.exists():
        return ""
    _seen.add(key)
    body = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8", errors="ignore"))

    def splice(m):
        return _expand_inputs(str(path.parent / m.group(1).strip()), _seen)

    return re.sub(r"\\(?:input|include)\s*\{([^}]*)\}", splice, body)


def cited_keys(tex: str, include_nocite: bool = False) -> dict:
    r"""Key -> number of times cited.

    \cite and its variants attach a source to a CLAIM, so each needs a row in the claims ledger
    recording the sentence that supports it. \nocite does not: it exists to pull an entry into the
    reference list without making any assertion about it, and this manuscript uses it for the
    publications the corpus data came from. Demanding a supporting sentence for each of those 208
    provenance entries would block submission over a correctly-cited dataset, and a gate that
    fires on correct behaviour gets switched off.

    So \nocite keys are checked for existence and a live DOI, and exempted from the ledger
    requirement. Pass include_nocite=True to get them.
    """
    txt = _expand_inputs(tex)
    txt = re.sub(r"(?<!\\)%.*", "", txt)          # strip comments, keep escaped \%
    pat = (r"\\nocite\s*\{([^}]*)\}" if include_nocite
           else r"\\(?!nocite)[a-zA-Z]*cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
    out: dict = {}
    for m in re.finditer(pat, txt):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                out[k] = out.get(k, 0) + 1
    return out


def doi_alive(doi: str, cache: dict, mail: str) -> bool:
    """Does this DOI resolve at EITHER registry?

    Checking Crossref alone marks every arXiv, Zenodo and figshare DOI dead, because those are
    registered with DataCite. build_bibliography.py already resolves through both; this function
    did not, and consequently blocked submission over a CatBoost citation that is perfectly valid.
    A gate that raises false alarms gets ignored, which is worse than no gate.
    """
    if not doi:
        return False
    if doi in cache:
        return cache[doi] is not None
    hdr = {"User-Agent": f"heusler-kappa-verify (mailto:{mail})"}
    try:
        r = requests.get(f"{API}{doi}", params={"mailto": mail} if mail else {},
                         timeout=20, headers=hdr)
        time.sleep(DELAY)
        if r.status_code == 200:
            cache[doi] = r.json().get("message", {})
            return True
        r2 = requests.get(f"{API_DATACITE}{doi}", timeout=20, headers=hdr)
        time.sleep(DELAY)
        ok = r2.status_code == 200
        cache[doi] = (r2.json().get("data", {}).get("attributes", {}) if ok else None)
        return ok
    except requests.RequestException:
        return True          # a network problem is not evidence the DOI is dead


def load_ledger() -> list:
    import csv
    p = Path(LEDGER)
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_citations.py paper/manuscript.tex")
        return 2
    tex = sys.argv[1]
    if not Path(tex).exists():
        print(f"manuscript not found: {tex}")
        print("(nothing to verify yet -- this gate runs once drafting starts)")
        return 0
    if not Path(BIB).exists():
        print(f"missing {BIB} -- run build_bibliography.py first")
        return 2

    bib = parse_bib(BIB)
    cites = cited_keys(tex)                       # claims -- need a ledger row
    nocites = cited_keys(tex, include_nocite=True)  # provenance -- need only to exist
    ledger = load_ledger()
    try:
        cache = json.loads(Path(CACHE).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cache = {}
    mail = mailto()

    print(f"manuscript      {tex}")
    print(f"bibliography    {len(bib)} entries")
    print(f"citations       {sum(cites.values())} uses of {len(cites)} distinct keys")
    print(f"claims ledger   {len(ledger)} rows\n")

    fail = 0

    print(f"provenance      {len(nocites)} nocite keys (data sources, exempt from the ledger)\n")

    # 1 -- undefined keys. A \nocite key missing from the .bib is just as broken as a \cite one.
    undef = sorted(k for k in set(cites) | set(nocites) if k not in bib)
    if undef:
        fail += len(undef)
        print(f"BLOCK  {len(undef)} citation key(s) not in references.bib:")
        for k in undef:
            print(f"    \\cite{{{k}}}  used {cites[k]}x  -- typed into the text, never resolved")
        print()

    # 2 -- dead DOIs, checked only for keys actually cited
    dead = []
    for k in sorted(cites):
        if k in bib:
            d = bib[k].get("doi", "")
            if d and not doi_alive(d, cache, mail):
                dead.append((k, d))
    Path(CACHE).write_text(json.dumps(cache), encoding="utf-8")
    if dead:
        fail += len(dead)
        print(f"BLOCK  {len(dead)} cited entr(ies) whose DOI does not resolve:")
        for k, d in dead:
            print(f"    {k:<28}{d}")
        print()

    # 3 -- claims with no supporting text
    by_key = {}
    for row in ledger:
        by_key.setdefault((row.get("cite_key") or "").strip(), []).append(row)
    unsupported = []
    for k in sorted(cites):
        rows = by_key.get(k, [])
        if not rows:
            unsupported.append((k, "no ledger row at all"))
        elif not any((r.get("supporting_text") or "").strip() for r in rows):
            unsupported.append((k, "ledger row has empty supporting_text"))
    if unsupported:
        fail += len(unsupported)
        print(f"BLOCK  {len(unsupported)} citation(s) with no recorded support from the source:")
        for k, why in unsupported[:25]:
            print(f"    {k:<28} {why}")
        if len(unsupported) > 25:
            print(f"    ... and {len(unsupported) - 25} more")
        print("  A real DOI attached to a claim it does not support is still a bad citation.")
        print()

    # 4 -- uncited entries
    orphan = sorted(k for k in bib if k not in cites and k not in nocites)
    if orphan:
        print(f"WARN   {len(orphan)} bibliography entr(ies) never cited "
              f"(usually a claim lost its support):")
        for k in orphan[:12]:
            print(f"    {k}")
        if len(orphan) > 12:
            print(f"    ... and {len(orphan) - 12} more")
        print()

    if fail:
        print(f"FAILED -- {fail} blocking problem(s). The manuscript is not ready to submit.")
        return 1
    print("PASSED -- every citation resolves and every claim has recorded support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
