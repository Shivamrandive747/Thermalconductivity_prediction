"""Re-verify every "novel" compound against CURRENT data, not the August snapshot.

WHY. The novelty list was settled 2026-08-24. Since then this project has mined papers, recovered
1,765 Starrydata rows and fetched structures -- so "nobody has measured this" may simply have
stopped being true. Spot-checking found `NbGaNi` in training; a second check found `FeGeMo` and
`NbInNi` too. Spot checks are not good enough for a claim that carries the paper.

Four independent places a compound can turn out to be known:
  1. the training set          (it has a kappa label we use)
  2. the kappa pool            (a kappa exists but was filtered out)
  3. the measurement database  (extracted from a paper at some point)
  4. this session's searches   (a paper names it, even if we never extracted the value)

All four are checked, and the DOI is reported so the verdict is auditable rather than asserted.
"""
from __future__ import annotations

import glob
import sqlite3
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from pymatgen.core import Composition

NOVEL = ("HfFe2Ge GaCoMo ZrGeRu2 FeGeMo ZrGaCu NbGaNi NbInNi HfGeRu2 ScBiRu2 TaMn2Sb HfGaCu "
         "ScSbRu2 NbInRu2 HfInCu Mn2NbSb ScZn2In YZn2In ZnGaCu2 ZnCuGe MnSbMo AgBiPd Cu2GePd "
         "ZnSbPd Zn2GaCu AlZnAg2").split()
OUT = "data/Target_Materials/NOVELTY_REVERIFIED.csv"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    tr = pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv")
    tr["red"] = tr.formula.map(red)
    pool = pd.read_csv("data/external/KAPPA_POOL_MASTER.csv")
    pcol = next(c for c in ("formula", "reduced_formula", "compound") if c in pool.columns)
    pool["red"] = pool[pcol].map(red)

    db = pd.DataFrame()
    try:
        con = sqlite3.connect("file:data/heusler.sqlite?mode=ro", uri=True)
        db = pd.DataFrame(con.execute(
            "select s.reduced_formula f, m.doi, m.property_name p from samples s "
            "join measurements m on m.sample_id=s.sample_id "
            "where m.canonical_value is not null").fetchall(),
            columns=["f", "doi", "p"])
        db["red"] = db.f.map(red)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: measurement DB unreadable ({exc}) -- verdicts will be incomplete")

    hits = pd.DataFrame()
    try:
        h = pd.read_csv("data/external/COMPOUND_HITS_FIXED.csv")
        h["cl"] = h.compounds.astype(str).str.split("|")
        hits = h.explode("cl")
        hits["red"] = hits.cl.map(red)
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for n in NOVEL:
        r = red(n)
        t = tr[tr.red == r]
        p = pool[pool.red == r]
        d = db[db.red == r] if len(db) else db
        dk = d[d.p.astype(str).str.contains("thermal_conductivity", na=False)] if len(d) else d
        hh = hits[(hits.red == r) & hits.lattice_kappa.fillna(False)] if len(hits) else hits
        why = []
        if len(t):
            why.append(f"training({t.method_tier.min():.0f}+)")
        if len(p):
            why.append("pool")
        if len(dk):
            why.append(f"DB-kappa({dk.doi.nunique()} DOI)")
        if len(hh):
            why.append(f"paper-names-kappa({len(hh)})")
        rows.append(dict(
            compound=n, reduced=r,
            in_training=len(t) > 0, in_pool=len(p) > 0,
            db_kappa_rows=int(len(dk)), paper_hits_with_kappa=int(len(hh)),
            example_doi=(dk.doi.iloc[0] if len(dk) else
                         (hh.doi.iloc[0] if len(hh) else "")),
            verdict=("NOT NOVEL" if why else "novel"),
            evidence="; ".join(why)))
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False)

    novel = D[D.verdict == "novel"]
    notn = D[D.verdict == "NOT NOVEL"]
    print(f"=== RE-VERIFIED {len(D)} claimed-novel compounds against CURRENT data ===\n")
    print(f"  {'compound':<11}{'verdict':<11}evidence")
    for _, r in D.sort_values("verdict").iterrows():
        print(f"  {r.compound:<11}{r.verdict:<11}{r.evidence}"
              f"{('  ' + str(r.example_doi)[:40]) if r.example_doi else ''}")
    print(f"\n  STILL NOVEL : {len(novel)}")
    print(f"  {sorted(novel.compound)}")
    print(f"\n  NO LONGER NOVEL: {len(notn)}")
    print(f"  {sorted(notn.compound)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
