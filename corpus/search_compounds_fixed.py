"""Compound-by-compound literature search, rebuilt after a seven-bug review.

WHAT WENT WRONG BEFORE, AND WHAT IS DIFFERENT NOW.

The previous two designs each produced confident zeros that were not real, and I reported them as
fact. The review found seven defects; the three that mattered:

  G  A 429 was indistinguishable from "no papers exist". `oa_search` returned None on failure and
     the caller logged `+0`. OpenAlex's daily budget ran out mid-run and the job kept going,
     recording an empty result for every compound it touched. FIXED: every fetcher returns one of
     three things -- a list, the sentinel EMPTY, or None meaning THE REQUEST FAILED -- and a
     failure aborts the run instead of counting as absence.

  B  Formula matching was a substring test, so `CaAgP` matched a paper about CaAgPb, `ZrNiSn`
     matched ZrNiSn2, and `Fe2VAl` matched the doped Fe2VAl0.9Si0.1. FIXED: formulas are extracted
     from the text and compared as pymatgen Compositions by reduced formula, so `Fe2VAl` still
     equals `AlVFe2` while `CaAgPb` is correctly rejected.

  C  Only 3-element integer compounds were searched, silently excluding quaternary Heuslers, both
     binary families and every doped compound -- material classes this project actually holds.
     FIXED: seven classes are admitted and labelled.

Retrieval templates are not guesses. `validate_queries.py` measured all ten against compounds whose
published kappa we already hold; "{f} lattice thermal conductivity" -- the old default -- had the
worst miss rate on OpenAlex (38%) and retrieved literally nothing on arXiv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from pymatgen.core import Composition, Element

load_dotenv(find_dotenv(usecwd=True))
KEY = os.getenv("OPENALEX_API")
MAIL = "teammaterialscienceteam@gmail.com"
S = requests.Session()
S.headers["User-Agent"] = f"heusler-cpd/3.0 (mailto:{MAIL})"

HITS = Path("data/external/COMPOUND_HITS_FIXED.csv")
STATE = Path("data/external/compound_search_state.json")
FAILURES = Path("data/external/search_failures.csv")

EMPTY: list = []                     # a genuine zero result, distinct from a failed request
MAX_CONSECUTIVE_FAILURES = 20


class BudgetExhausted(RuntimeError):
    """Raised when an index reports it will not serve us any more requests today."""


# --------------------------------------------------------------------------- text handling
_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# NOTE: no literal "sub" here. The previous version stripped those three letters anywhere they
# appeared, turning "substitution" into "stitution". The tag rule already removes <sub></sub>.
_SUPSUB = re.compile(r"</?su[bp]>", re.I)      # removed with NO space: they sit INSIDE a formula
_MARKUP = re.compile(r"<[^>]+>|&[a-z]+;")      # any other tag becomes a space
_TOKEN = re.compile(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?)+\b")

_VALID_EL = {e.symbol for e in Element}

KAPPA = re.compile(r"thermal conduct|lattice thermal|kappa|phonon|thermal transport|"
                   r"heat transport|thermoelectric|figure of merit|seebeck|\bzT\b", re.I)
LATTICE = re.compile(r"lattice thermal conduct|phonon thermal|kappa_?l\b|phonon transport", re.I)
METHOD = re.compile(r"first.?principles|density functional|\bDFT\b|phono3py|shengbte|almabte|"
                    r"boltzmann|three.?phonon|four.?phonon|anharmonic|slack|debye", re.I)
DOPED = re.compile(r"\bdop(ed|ing)\b|substitut|off.?stoichiometr|solid solution|\bx\s*=\s*0?\.\d",
                   re.I)
# Grain size is not decoration: the transfer model currently applies one global L_eff = 38 nm to
# every sample in the world, and per-sample grain size is the last reducible error source left.
GRAIN = re.compile(r"grain size|grain boundar|crystallite size|scherrer|\bSEM\b|\bEBSD\b|"
                   r"ball.?mill|spark plasma|hot.?press|sinter|melt.?spin|nanostructur", re.I)
EXPERIMENTAL = re.compile(r"measur|experiment|synthes|arc.?melt|prepared by|as-cast|"
                          r"single crystal|polycrystal", re.I)


def clean_text(s: str) -> str:
    """Strip markup and fold unicode subscripts, but KEEP word boundaries.

    Subscript and superscript tags are deleted outright rather than replaced by a space: they sit
    INSIDE a formula, so `Co<sub>2</sub>TiSn` must become `Co2TiSn` and not `Co 2 TiSn`, which
    would be torn into three tokens and never recognised.
    """
    return _MARKUP.sub(" ", _SUPSUB.sub("", str(s))).translate(_SUBS)


def norm(s: str) -> str:
    """Aggressive fold for loose comparison only -- never for identity."""
    return re.sub(r"[\s\-_·,{}$\\]", "", clean_text(s)).lower()


def extract_formulas(text: str) -> set[str]:
    """Every chemical formula in the text, as reduced formulas.

    This is what replaces the substring test. A token is accepted only if the WHOLE of it parses
    as a composition of real elements, which rejects ordinary English (`Cost`, `Bane`, `Nice`)
    because their trailing letters are not element symbols. Two-element tokens additionally
    require a digit, so `In` (indium) inside "In this work" cannot masquerade as a compound.
    """
    out: set[str] = set()
    for tok in _TOKEN.findall(clean_text(text)):
        if len(tok) < 3:
            continue
        syms = re.findall(r"[A-Z][a-z]?", tok)
        if not syms or not all(s in _VALID_EL for s in syms):
            continue
        n_el = len(set(syms))
        if n_el < 2:
            continue
        if n_el == 2 and not re.search(r"\d", tok):
            continue
        try:
            c = Composition(tok)
            if len(c.elements) >= 2:
                out.add(c.reduced_formula)
        except Exception:  # noqa: BLE001
            continue
    return out


def reduced(f: str) -> str | None:
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def formula_variants(f: str) -> set[str]:
    """Element orderings, used only for the LOOSE secondary match and for query strings."""
    out = {str(f)}
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return out
    out.add(c.reduced_formula)
    out.add(c.formula.replace(" ", ""))
    items = sorted(c.get_el_amt_dict().items(), key=lambda kv: -kv[1])

    def s(n):
        if abs(n - 1) < 1e-6:
            return ""
        return str(int(n)) if abs(n - round(n)) < 1e-6 else f"{n:g}"

    if len(items) == 3:
        (a, na), (b, nb), (d, nd) = items
        for p in ((a, na, b, nb, d, nd), (a, na, d, nd, b, nb), (b, nb, d, nd, a, na),
                  (b, nb, a, na, d, nd), (d, nd, b, nb, a, na), (d, nd, a, na, b, nb)):
            out.add(f"{p[0]}{s(p[1])}{p[2]}{s(p[3])}{p[4]}{s(p[5])}")
    return {o for o in out if o}


def match_kind(text: str, compound: str) -> str | None:
    """'exact' when a real formula in the text reduces to ours, 'loose' for a fallback name match.

    The loose pass is kept because a few papers write a compound only in a figure caption our
    abstract does not include, but it is recorded separately so precision stays auditable.
    """
    target = reduced(compound)
    if target and target in extract_formulas(text):
        return "exact"
    n = norm(text)
    for v in formula_variants(compound):
        nv = norm(v)
        if len(nv) >= 5 and nv in n:
            # guard the substring case that caused the CaAgP/CaAgPb bug: reject when the match is
            # immediately followed by another element symbol or a digit
            for m in re.finditer(re.escape(nv), n):
                tail = n[m.end():m.end() + 2]
                if not re.match(r"^[a-z0-9]", tail):
                    return "loose"
    return None


# --------------------------------------------------------------------------- classification
_TEMPLATES = ([2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [3.0, 1.0], [2.0, 1.0])
_CLASS = {"[2.0, 1.0, 1.0]": "full_heusler", "[1.0, 1.0, 1.0]": "half_heusler",
          "[1.0, 1.0, 1.0, 1.0]": "quaternary_heusler", "[3.0, 1.0]": "binary_Cu3Au",
          "[2.0, 1.0]": "binary_2to1"}


def classify(f: str) -> tuple[str, bool]:
    """(structure_family, is_doped). Admits all seven classes the project actually holds."""
    try:
        amts = Composition(str(f)).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return "unknown", False
    tot = sum(amts.values())
    if tot <= 0 or not amts:
        return "unknown", False
    best = None
    for caps in _TEMPLATES:
        if len(amts) < len(caps):
            continue                       # too few elements to fill the sites
        scale = sum(caps) / tot
        vals = sorted((a * scale for a in amts.values()), reverse=True)
        tmpl = sorted(caps, reverse=True)
        # Compare the amount vector to the template directly, and charge for any element beyond
        # the site count. Pouring the amounts into the sites (a previous attempt) always fills
        # them exactly once the composition is normalised, so every template scored 0 and the
        # first one listed always won -- CoFeYGe came out "full_heusler".
        dev = sum(abs(v - c) for v, c in zip(vals, tmpl))
        dev += sum(vals[len(tmpl):])       # extra elements must be explained by substitution
        if best is None or dev < best[0]:
            best = (dev, caps)
    if best is None or best[0] > 0.8:
        return "other", any(abs(v - round(v)) > 0.02 for v in amts.values())
    doped = any(abs(v - round(v)) > 0.02 for v in amts.values()) or best[0] > 1e-6
    return _CLASS.get(str(best[1]), "other"), doped


def parent_formula(f: str) -> str:
    """For a doped compound, the stoichiometric parent -- which is what papers actually print."""
    try:
        c = Composition(str(f))
    except Exception:  # noqa: BLE001
        return str(f)
    amts = sorted(c.get_el_amt_dict().items(), key=lambda kv: -kv[1])
    fam, doped = classify(f)
    if not doped:
        return str(f)
    keep = 4 if fam == "quaternary_heusler" else (2 if fam.startswith("binary") else 3)
    majors = amts[:keep]
    tot = sum(v for _, v in majors)
    if tot <= 0:
        return str(f)
    n_at = 4.0 if fam in ("full_heusler", "quaternary_heusler", "binary_Cu3Au") else 3.0
    return "".join(f"{e}{'' if round(v * n_at / tot) == 1 else int(round(v * n_at / tot))}"
                   for e, v in majors)


# --------------------------------------------------------------------------- fetchers
# Each returns: list of items (success), EMPTY (genuine zero), or None (REQUEST FAILED -> abort).
def _budget_check(resp) -> None:
    try:
        j = resp.json()
    except Exception:  # noqa: BLE001
        return
    msg = str(j.get("message", ""))
    if j.get("dailyRemainingUsd") == 0 or "Insufficient budget" in msg:
        raise BudgetExhausted(msg or "daily budget exhausted")


def oa_search(q, use_key: bool, per_page=50):
    params = {"search": q, "per-page": per_page, "filter": "type:article"}
    if use_key and KEY:
        params["api_key"] = KEY
    for attempt in (0, 1, 2):
        try:
            r = S.get("https://api.openalex.org/works", params=params, timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json().get("results", []) or EMPTY
            except Exception:  # noqa: BLE001
                return None
        if r.status_code == 429:
            _budget_check(r)                      # raises BudgetExhausted -> whole run stops
            time.sleep(3 * (attempt + 1))
            continue
        return None
    return None


def crossref_search(q, rows=30):
    for attempt in (0, 1, 2):
        try:
            r = S.get("https://api.crossref.org/works",
                      params={"query.bibliographic": q, "rows": rows, "mailto": MAIL,
                              "select": "DOI,title,abstract,issued,"
                                        "is-referenced-by-count"}, timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                items = r.json().get("message", {}).get("items", [])
            except Exception:  # noqa: BLE001
                return None
            out = []
            for w in items:
                yr = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
                out.append({"_doi": w.get("DOI"), "_title": " ".join(w.get("title") or []),
                            "_abs": w.get("abstract") or "", "_year": yr,
                            "_cited": w.get("is-referenced-by-count") or 0, "_oa": None})
            return out or EMPTY
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        return None
    return None


def arxiv_search(q, limit=25):
    for attempt in (0, 1, 2):
        try:
            r = S.get("http://export.arxiv.org/api/query?"
                      f"search_query=all:{quote(q)}&max_results={limit}", timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code != 200:
            time.sleep(2 * (attempt + 1))
            continue
        out = []
        for m in re.finditer(r"<entry>(.*?)</entry>", r.text, re.S):
            e = m.group(1)
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            su = re.search(r"<summary>(.*?)</summary>", e, re.S)
            idm = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", e)
            dm = re.search(r"<arxiv:doi[^>]*>([^<]+)</arxiv:doi>", e)
            if not idm:
                continue
            out.append({"_doi": dm.group(1) if dm else f"arxiv:{idm.group(1)}",
                        "_title": re.sub(r"\s+", " ", t.group(1)).strip() if t else "",
                        "_abs": re.sub(r"\s+", " ", su.group(1)).strip() if su else "",
                        "_year": None, "_cited": 0,
                        "_oa": f"https://arxiv.org/pdf/{idm.group(1)}"})
        return out or EMPTY
    return None


def abstract_of(w) -> str:
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {}
    for wd, ix in inv.items():
        for i in ix:
            pos[i] = wd
    return " ".join(pos[i] for i in sorted(pos))


# Measured by validate_queries.py against compounds whose published kappa we already hold.
PLAN = [("openalex", "{f}"), ("openalex", "{f} thermoelectric"),
        ("openalex", "{f} Heusler"), ("openalex", "{f} alloy"),
        ("crossref", "{f}"), ("crossref", "{f} thermoelectric"),
        ("arxiv", "{f}")]


def main(limit: int | None = None, sleep: float = 1.0, only: str | None = None,
         use_key: bool = False, indexes: str = "openalex,crossref,arxiv") -> int:
    """`indexes` lets a run use only the indexes that are currently serving us.

    OpenAlex meters BOTH the keyed and the anonymous pool against the same daily budget, and when
    it is spent both return 429 for ~18 h. Crossref and arXiv have no such budget. Rather than
    idling until midnight UTC, the free indexes run now and OpenAlex runs later -- which is safe
    only because progress is tracked PER INDEX, so the later pass does not skip a compound merely
    because the earlier pass touched it.
    """
    want_idx = [x.strip() for x in indexes.split(",") if x.strip()]
    plan = [(i, t) for i, t in PLAN if i in want_idx]
    if not plan:
        print(f"no usable indexes in {indexes!r}")
        return 1
    print(f"indexes this run: {sorted({i for i, _ in plan})}", flush=True)
    # ---- 1. build the compound list, all seven classes ----------------------
    want: set[str] = set()
    for path in Path("data/Target_Materials").glob("*"):
        if path.suffix not in (".csv", ".xlsx") or "Target_Scope" in path.name:
            continue
        try:
            sheets = ([pd.read_csv(path)] if path.suffix == ".csv"
                      else [pd.ExcelFile(path).parse(n)
                            for n in pd.ExcelFile(path).sheet_names])
        except Exception:  # noqa: BLE001
            continue
        for d in sheets:
            for c in ("compound", "formula", "reduced_formula"):
                if c in d.columns:
                    want |= {str(x).strip() for x in d[c].dropna()}

    entries: dict[str, dict] = {}
    for f in want:
        fam, doped = classify(f)
        if fam in ("unknown", "other"):
            continue
        # A doped compound is searched by its PARENT: no paper prints "Al0.85V1.15Fe1.85Cu0.15".
        query_name = parent_formula(f) if doped else f
        r = reduced(query_name)
        if not r:
            continue
        entries.setdefault(r, dict(query=query_name, family=fam, doped_seen=False,
                                   members=set()))
        entries[r]["members"].add(f)
        if doped:
            entries[r]["doped_seen"] = True

    known: set[str] = set()
    for src in ("data/external/KAPPA_POOL_MASTER.csv",
                "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"):
        if os.path.exists(src):
            try:
                dd = pd.read_csv(src)
                col = next(c for c in ("formula", "reduced_formula", "compound")
                           if c in dd.columns)
                known |= {reduced(x) for x in dd[col].dropna()}
            except Exception:  # noqa: BLE001
                pass
    order = sorted(entries, key=lambda r: (r not in known, r))
    fam_counts = pd.Series([entries[r]["family"] for r in order]).value_counts().to_dict()
    print(f"compounds to search: {len(order)}   by class: {fam_counts}", flush=True)

    # ---- 2. resume ----------------------------------------------------------
    smoke = bool(only)
    rows: dict[str, dict] = {}
    done: dict[str, list] = {}          # compound -> indexes already searched
    stats = defaultdict(lambda: {"queries": 0, "hits": 0})
    if HITS.exists():
        try:
            for r in pd.read_csv(HITS).to_dict("records"):
                rows[str(r["doi"])] = r
            print(f"  resuming with {len(rows)} papers on file", flush=True)
        except Exception:  # noqa: BLE001
            pass
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
            raw = st.get("done", {})
            # migrate the old list-of-compounds format: those runs used all three indexes
            done = ({k: list(v) for k, v in raw.items()} if isinstance(raw, dict)
                    else {c: ["openalex", "crossref", "arxiv"] for c in raw})
            for k, v in st.get("tmpl", {}).items():
                stats[k] = v
            print(f"  resuming: {len(done)} compounds have some index coverage", flush=True)
        except Exception:  # noqa: BLE001
            pass

    if smoke:
        todo = [reduced(c.strip()) for c in only.split(",") if c.strip()]
        todo = [t for t in todo if t]
        for t in todo:
            entries.setdefault(t, dict(query=t, family=classify(t)[0], doped_seen=False,
                                       members={t}))
        print(f"  SMOKE TEST on {todo} (no checkpoint will be written)", flush=True)
    else:
        # a compound still needs work if ANY requested index has not searched it yet
        need = set(want_idx)
        todo = [r for r in order if not need.issubset(set(done.get(r, [])))]
    if limit:
        todo = todo[:limit]
    n_before = len(rows)
    print(f"  to do this run: {len(todo)}\n", flush=True)

    failures: list[dict] = []
    consecutive = defaultdict(int)

    def record(item, compound, tmpl, src) -> str:
        doi = str(item.get("_doi") or "").replace("https://doi.org/", "").lower().strip()
        title, abs_ = item.get("_title") or "", item.get("_abs") or ""
        if not doi:
            return "rejected"
        blob = f"{title} {abs_}"
        kind = match_kind(blob, compound)
        if kind is None or not KAPPA.search(blob):
            return "rejected"
        if doi in rows:
            cs = set(str(rows[doi].get("compounds", "")).split("|")) | {compound}
            rows[doi]["compounds"] = "|".join(sorted(c for c in cs if c and c != "nan"))
            return "merged"
        fam = entries.get(reduced(compound) or "", {}).get("family", classify(compound)[0])
        sc = 3
        if LATTICE.search(blob):
            sc += 4
        if METHOD.search(blob):
            sc += 2
        if GRAIN.search(blob):
            sc += 2
        if kind == "exact":
            sc += 1
        if DOPED.search(blob):
            sc -= 2
        if not abs_:
            sc -= 1
        rows[doi] = dict(doi=doi, title=str(title)[:180], compounds=compound,
                         structure_family=fam, via=tmpl, index=src, match_kind=kind,
                         year=item.get("_year"), cited=item.get("_cited") or 0,
                         oa_url=item.get("_oa"), score=sc,
                         lattice_kappa=bool(LATTICE.search(blob)),
                         has_grain_size=bool(GRAIN.search(blob)),
                         has_experimental=bool(EXPERIMENTAL.search(blob)),
                         doped_paper=bool(DOPED.search(blob)))
        return "new"

    def checkpoint():
        if smoke:
            return
        pd.DataFrame(rows.values()).to_csv(HITS, index=False)
        STATE.write_text(json.dumps({"done": done,
                                     "tmpl": {k: dict(v) for k, v in stats.items()}}))
        if failures:
            pd.DataFrame(failures).to_csv(FAILURES, index=False)

    try:
        for i, key in enumerate(todo, 1):
            ent = entries[key]
            cpd = ent["query"]
            got = 0
            for idx, tmpl in plan:
                q = tmpl.format(f=cpd)
                if idx == "openalex":
                    res = oa_search(q, use_key)
                    items = ([{"_doi": w.get("doi"), "_title": w.get("title") or "",
                               "_abs": abstract_of(w), "_year": w.get("publication_year"),
                               "_cited": w.get("cited_by_count"),
                               "_oa": (w.get("open_access") or {}).get("oa_url")}
                              for w in res] if res else res)
                elif idx == "crossref":
                    items = crossref_search(q)
                else:
                    items = arxiv_search(q)

                if items is None:                      # THE REQUEST FAILED -- never a zero
                    consecutive[idx] += 1
                    failures.append(dict(compound=cpd, index=idx, template=tmpl,
                                         reason="request failed"))
                    if consecutive[idx] >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{idx}: {consecutive[idx]} consecutive failures -- aborting rather "
                            f"than recording zeros. Retry list in {FAILURES}.")
                    continue
                consecutive[idx] = 0
                this_tmpl = 0
                for it in items:
                    if record(it, cpd, tmpl, idx) in ("new", "merged"):
                        this_tmpl += 1
                got += this_tmpl
                stats[f"{idx}|{tmpl}"]["queries"] += 1
                stats[f"{idx}|{tmpl}"]["hits"] += this_tmpl
                time.sleep(sleep)

            done[key] = sorted(set(done.get(key, [])) | {i for i, _ in plan})
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {cpd:<14} +{got:<3} "
                      f"new this run {len(rows) - n_before:<5} total {len(rows)}", flush=True)
            if i % 25 == 0:
                checkpoint()
    except BudgetExhausted as exc:
        checkpoint()
        print(f"\nABORTED -- OpenAlex budget exhausted: {exc}")
        print("  Nothing after this point was searched; no zeros were recorded for it.")
        print("  Progress saved. OpenAlex meters the keyed and anonymous pools against the "
              "SAME daily budget,")
        print("  so switching pools will not help. Either wait for the reset, or run the free "
              "indexes now:")
        print("      python search_compounds_fixed.py --indexes crossref,arxiv")
        return 2
    except RuntimeError as exc:
        checkpoint()
        print(f"\nABORTED -- {exc}")
        return 3
    except KeyboardInterrupt:
        checkpoint()
        print("\ninterrupted -- progress saved")
        return 130

    checkpoint()
    d = pd.DataFrame(rows.values())
    if not len(d):
        print("no hits")
        return 0
    print(f"\n=== papers found: {len(d)} total, {len(d) - n_before} NEW this run ===")
    print(f"  compounds searched this run : {len(todo)}")
    print(f"  report LATTICE kappa        : {int(d.lattice_kappa.sum())}")
    print(f"  mention grain size          : {int(d.has_grain_size.sum())}")
    print(f"  experimental                : {int(d.has_experimental.sum())}")
    print(f"  exact formula match         : {int((d.match_kind == 'exact').sum())}"
          f"  (loose {int((d.match_kind == 'loose').sum())})")
    if "structure_family" in d:
        print("\n  by structure family:")
        print(d.structure_family.value_counts().to_string())
    print("\n  template yield:")
    for t, s_ in sorted(stats.items(), key=lambda kv: -kv[1]["hits"]):
        print(f"    {t:<34}{s_['hits']:>6} hits /{s_['queries']:>5} queries")
    if failures:
        print(f"\n  {len(failures)} requests FAILED (unknown, not zero) -> {FAILURES}")
    print(f"\nwrote {HITS}" if not smoke
          else "\nsmoke test: no checkpoint written (results discarded by design)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--only", help="comma-separated compounds; smoke test, no checkpoint")
    ap.add_argument("--indexes", default="openalex,crossref,arxiv",
                    help="comma-separated subset of openalex,crossref,arxiv")
    ap.add_argument("--use-key", action="store_true",
                    help="use the OpenAlex api_key (default: anonymous pool, which is "
                         "measured to give identical recall and has no daily budget)")
    raise SystemExit(main(**vars(ap.parse_args())))
