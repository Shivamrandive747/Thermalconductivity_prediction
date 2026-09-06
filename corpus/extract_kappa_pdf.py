"""Extract Heusler lattice thermal conductivity tables from a PDF, via Gemini.

Built for papers whose kappa_L values live in a table -- often in the SI -- rather than in prose.
`d4dd00240g.pdf` (HH130, Digital Discovery 2024) is the first target: it reports kappa_L for 80
half-Heuslers INCLUDING four-phonon scattering, which is a higher level of theory than anything
else in our pool.

WHY A MODEL AND NOT A PDF PARSER. We already learned this the expensive way on Carrete's PRX
tables: `pdftotext -layout` MISPAIRED compound names with values, because the tables are typeset as
images and the text layer interleaves columns. A recovery by word coordinates worked but was
laborious and paper-specific. A model reads the table as a table.

WHAT IT MUST NOT DO. The single failure mode that matters here is invention. This project has
already had 1,571 fabricated measurements enter the database from title-only stubs, which took a
quarantine pass to remove. So the prompt forbids inference explicitly, the schema requires a stated
table/page for every row, and anything without a location is dropped before writing.

Every row carries source_doi, source_file and the table it came from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd

from pipeline.config import cfg

PROMPT = """You are extracting a data table from a scientific paper about Heusler thermoelectrics.

TASK: find EVERY table or figure-table in this document that reports LATTICE THERMAL CONDUCTIVITY
(kappa_L, k_L, kappa_lattice, or "lattice thermal conductivity") for specific named compounds, and
return every row.

For each compound-value pair return:
  formula                 exactly as written in the paper (e.g. "TiNiSn", "Fe2VAl", "ZrCoBi")
  kappa_L                 the numeric value
  kappa_units             the units as printed (usually W/mK or W m^-1 K^-1)
  temperature_K           the temperature. If the table header says 300 K, use 300. If a
                          temperature is genuinely not stated anywhere, use null - do NOT guess.
  method                  how it was computed, as the paper describes it: e.g. "three-phonon BTE",
                          "four-phonon", "ShengBTE", "Phono3py", "MLIP", "Slack model", "RTA", "SCP".
                          If the paper reports several levels of theory for the same compound,
                          RETURN ONE ROW PER LEVEL and name each.
  lattice_parameter_A     the lattice constant in Angstrom if the paper gives one for that
                          compound; null otherwise
  space_group             if stated; null otherwise
  location                WHERE you found it - "Table 2", "Table S1", "Figure 4", "page 7".
                          This is mandatory. A row without a location will be discarded.
  grain_size_nm           the average grain / crystallite size of the MEASURED sample, in nm,
                          if the paper states one (SEM grain size, or an XRD Scherrer crystallite
                          size). This is not decoration: a real polycrystal conducts less heat than
                          a perfect crystal because phonons scatter at grain boundaries, and the
                          grain size is what sets how much less. Convert um to nm (1 um = 1000 nm).
                          null if not stated - do NOT estimate it from micrographs.
  synthesis_method        how the measured sample was made, as the paper says it: e.g.
                          "arc melting + SPS", "ball milling + hot press", "single crystal",
                          "melt spinning". null if not stated.
  provenance              "primary" if THIS paper computed or measured the value.
                          "secondary" if the paper is QUOTING it from another paper - a value
                          carrying a bibliography marker like [18], "Ref. 12", "from Ref. [3]",
                          or a column headed "previous work" / "literature". This is mandatory
                          and it matters: a table that looks like 26 measurements is often 3
                          results plus 23 quotations, and counting the quotations again would
                          double-count work we already hold.

RULES, IN ORDER OF IMPORTANCE:
1. NEVER invent, infer, estimate or interpolate a value. Only transcribe numbers that are printed
   in the document. If you cannot read a cell, omit that row entirely.
2. Do not compute kappa_L from other quantities. Do not convert from a figure by eye.
3. Do NOT return electronic thermal conductivity (kappa_e) or total thermal conductivity
   (kappa_total) - only the LATTICE component. If a table gives total and electronic separately,
   you may return the lattice column only.
4. Keep the compound formula exactly as printed. Do not reorder elements or normalise subscripts.
5. If the same compound appears at several temperatures, return one row per temperature.
6. If the document contains NO lattice thermal conductivity table at all, return an empty list.
   That is a valid and useful answer - do not pad it.
7. Mark `provenance` honestly. When in doubt between primary and secondary, say secondary - an
   over-cautious exclusion costs one row, while a wrongly-included quotation corrupts a
   cross-check between sources that are supposed to be independent.

Return JSON: {"rows": [ ... ]}"""


def spreadsheet_as_text(path: Path, max_rows: int = 4000, max_chars: int = 700_000) -> str | None:
    """Render every sheet as delimited text, headers included, so the model reads real values.

    Truncation is by ROWS and announced in the text, never silent: a table cut off mid-way without
    saying so would look like a complete table with missing compounds."""
    try:
        if path.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(path)}
        else:
            xl = pd.ExcelFile(path)
            sheets = {n: xl.parse(n) for n in xl.sheet_names}
    except Exception as exc:  # noqa: BLE001
        print(f"  spreadsheet read failed: {exc}")
        return None
    out = [f"SUPPLEMENTARY SPREADSHEET: {path.name}"]
    for name, d in sheets.items():
        if d is None or not len(d):
            continue
        note = ""
        if len(d) > max_rows:
            note = f"  [TRUNCATED: showing first {max_rows} of {len(d)} rows]"
            d = d.head(max_rows)
        out.append(f"\n=== sheet: {name}  ({len(d)} rows x {len(d.columns)} cols){note}")
        out.append(d.to_csv(index=False))
    txt = "\n".join(out)
    return txt[:max_chars]


def main(pdf: str, doi: str, out: str, model: str | None = None) -> int:
    from google.genai import types

    # Use the pipeline's own client, which routes to VERTEX + Application Default
    # Credentials whenever GCP_PROJECT is set. The bare AI Studio key is NOT usable:
    # its free tier reports "limit: 0" for the current models, and the backend's own
    # comment records why -- "the AI Studio key, whose India prepaid wallet is
    # depleted -> every paper dies on a 429 after burning retry backoff".
    from pipeline.backends.gemini import _get_client

    mdl = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
    p = Path(pdf)
    if not p.exists():
        print(f"not found: {pdf}")
        return 1
    print(f"reading {p.name}  ({p.stat().st_size/1e6:.1f} MB)  model={mdl}  via Vertex "
          f"(project={cfg.gcp_project}, region={cfg.gcp_region})")

    client = _get_client()

    # A supplementary spreadsheet is where the hundred-row tables actually live, so those are
    # accepted too. Sending the sheet as TEXT rather than as an uploaded document is both cheaper
    # and more faithful -- the model sees the real cell values instead of a rendered picture of them.
    if p.suffix.lower() in (".xlsx", ".xls", ".csv"):
        payload = spreadsheet_as_text(p)
        if payload is None:
            print("  could not read the spreadsheet")
            return 1
        contents = [payload, PROMPT]
    else:
        contents = [types.Part.from_bytes(data=p.read_bytes(), mime_type="application/pdf"), PROMPT]

    resp = client.models.generate_content(
        model=mdl, contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0))

    txt = (resp.text or "").strip()
    try:
        rows = json.loads(txt).get("rows", [])
    except Exception as exc:  # noqa: BLE001
        print(f"could not parse response: {exc}\n{txt[:400]}")
        return 1

    print(f"model returned {len(rows)} rows")
    if not rows:
        print("  no lattice thermal conductivity table found in this document")
        return 0

    d = pd.DataFrame(rows)
    before = len(d)
    d = d[d.get("location").notna() & (d.get("location").astype(str).str.len() > 0)]
    d = d[pd.to_numeric(d.get("kappa_L"), errors="coerce").notna()]
    if len(d) < before:
        print(f"  dropped {before - len(d)} rows with no location or no numeric value")

    for opt in ("grain_size_nm", "synthesis_method"):
        if opt not in d.columns:
            d[opt] = None
    d["source"] = "paper"
    d["source_doi"] = doi
    d["source_file"] = p.name
    d["data_origin"] = "theoretical"
    d["method_class"] = ["BTE" if any(k in str(m).lower() for k in
                                      ("phonon", "bte", "shengbte", "phono3py", "scp"))
                         else "other" for m in d.get("method", "")]

    print(f"\n  kept {len(d)} rows, {d.formula.nunique()} distinct compounds")
    print(f"  methods : {dict(d.method.value_counts())}")
    if "temperature_K" in d:
        t = pd.to_numeric(d.temperature_K, errors="coerce")
        print(f"  T       : {t.min()} - {t.max()} K  ({t.notna().sum()} rows have one)")
    print(f"  locations: {sorted(set(d.location.astype(str)))[:6]}")
    k = pd.to_numeric(d.kappa_L, errors="coerce")
    print(f"  kappa_L : {k.min():.2f} - {k.max():.2f}")
    print(f"\n  {'formula':<14}{'kappa_L':>9}{'T':>7}  {'method':<22}location")
    for _, r in d.head(15).iterrows():
        print(f"  {str(r.formula):<14}{float(r.kappa_L):>9.2f}{str(r.get('temperature_K')):>7}  "
              f"{str(r.get('method'))[:20]:<22}{r.location}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/pdfs/d4dd00240g.pdf")
    ap.add_argument("--doi", default="10.1039/d4dd00240g")
    ap.add_argument("--out", default="data/external/kappa_from_pdf_hh130.csv")
    ap.add_argument("--model")
    raise SystemExit(main(**vars(ap.parse_args())))
