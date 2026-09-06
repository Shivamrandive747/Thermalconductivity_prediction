"""Merge the per-paper Gemini extractions into one file, separating PRIMARY from QUOTED values.

THE PROBLEM THIS SOLVES. A paper's kappa table routinely contains three kinds of number:

  1. what THIS paper computed or measured            -- primary, usable
  2. what this paper QUOTES from earlier papers      -- secondary, must not be counted again
  3. reference materials for comparison (PbTe, ...)  -- not Heuslers at all

Phys. Rev. B 95, 045202 Table I is the clean example: 26 rows, but only 3 are "RTA-BTE (this work)".
The other 23 are labelled `Theory [53]`, `Experimental [18]` and so on -- values lifted from the
bibliography. Ingesting those would double-count the originals we already hold and would make
independent sources look like they agree with each other, which is the worst possible failure for a
cross-check.

DETECTION. A method string is secondary if it names a bracketed reference (`[18]`, `[53]`) or says
"literature". This is a syntactic rule over what the model transcribed, not a judgement call.

Non-Heusler comparison materials are left alone here -- `build_kappa_pool.py` already drops anything
that is not a Heusler stoichiometry, so PbTe and Bi2Te3 fall out there.
"""
from __future__ import annotations

import glob
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

OUT = "data/external/kappa_pdf_batch2.csv"
# "[18]", "[3]" -> a value taken from reference 18. "literature" -> same thing said in words.
SECONDARY = re.compile(r"\[\s*\d+\s*\]|literature|ref\.?\s*\d|from ref", re.I)
# A value read off a PLOT is an eyeball estimate, not a transcribed number. The extraction prompt
# forbids it ("Do not convert from a figure by eye") but the model does it anyway -- measured, 28
# of 44 rows in one batch came from "Figure 4e" and "Figure 11d". Digitising a curve to two
# significant figures is guesswork, and guessed numbers are exactly what this pipeline exists to
# keep out. They are kept in the file, flagged, and dropped before training.
FIGURE = "fig"        # plain substring: "Figure 4e", "Fig. 3", "figure 11d"


def main() -> int:
    frames = []
    for f in sorted(glob.glob("data/external/pdfx_*.csv")):
        d = pd.read_csv(f)
        if not len(d):
            continue
        d["source"] = "paper mining (Gemini PDF extraction)"
        frames.append(d)
        print(f"  {Path(f).name:<50}{len(d):>5} rows")
    if not frames:
        print("nothing to merge")
        return 0

    d = pd.concat(frames, ignore_index=True)

    # NORMALISE UNICODE SUBSCRIPTS BEFORE ANYTHING ELSE READS THE FORMULA.
    # The model transcribes formulas exactly as the paper prints them, so it returns "Ru₂TiGe"
    # and "Fe₂VAl" with U+2082-style digits. Every downstream parser -- pymatgen included --
    # expects ASCII, so those rows would silently fail to match the pool and be dropped. That
    # would have lost Ru2TiGe, the one experimental measurement in the target chemistry that
    # this whole campaign produced.
    _SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    if "formula" in d.columns:
        before = d.formula.astype(str)
        d["formula"] = before.map(lambda x: str(x).translate(_SUBS).strip())
        fixed = int((before != d.formula).sum())
        if fixed:
            print(f"\n  normalised unicode subscripts in {fixed} formula(e), e.g. "
                  f"{sorted(set(before[before != d.formula]))[:3]}")

    meth = d.method.astype(str)
    sec = meth.str.contains(SECONDARY, na=False)
    loc = d.location.astype(str) if "location" in d.columns else pd.Series("", index=d.index)
    fig = loc.str.contains(FIGURE, case=False, na=False, regex=False)

    # A provenance already recorded in a per-paper file was set by inspecting that paper's own
    # text -- better evidence than anything inferred here -- so it is preserved. This is how the
    # RhBTi rows stay flagged: the source says RhBiTi 34 times and RhBTi twice with an identical
    # kappa, so RhBTi is a mis-transcription, and RhBTi is a boride rather than a bismuthide.
    preset = (d["provenance"].astype(str) if "provenance" in d.columns
              else pd.Series("", index=d.index))
    keep_preset = preset.str.contains("transcription_error|suspect|invalid", case=False, na=False)

    d["provenance"] = ["secondary (quoted from another paper)" if x else "primary" for x in sec]
    d.loc[fig & ~sec, "provenance"] = "figure_estimate (read off a plot, not a table)"
    d.loc[keep_preset, "provenance"] = preset[keep_preset]
    if int(keep_preset.sum()):
        print(f"  kept {int(keep_preset.sum())} upstream provenance flag(s): "
              f"{sorted(set(preset[keep_preset]))[:2]}")
    d["source_kind"] = ["figure" if f else "table/text" for f in fig]
    prim = d.provenance == "primary"

    print(f"\n  total rows            : {len(d)}")
    print(f"  PRIMARY (table/text)  : {int(prim.sum())}   {d[prim].formula.nunique()} formulae")
    print(f"  SECONDARY (quoted)    : {int(sec.sum())}  -> excluded downstream")
    print(f"  FIGURE estimates      : {int((fig & ~sec).sum())}  -> excluded downstream "
          f"(a value read off a plot is a guess, not a transcription)")
    print(f"  TRANSCRIPTION errors  : {int(keep_preset.sum())}  -> excluded downstream")
    if int(sec.sum()):
        ex = sorted(set(meth[sec]))[:6]
        print(f"    quoted e.g. {ex}")
    print(f"\n  by paper (primary only):")
    for doi, g in d[prim].groupby("source_doi"):
        print(f"    {str(doi):<34}{len(g):>4} rows  {g.formula.nunique():>3} formulae")

    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
