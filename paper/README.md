# Manuscript

**Calibrating first-principles lattice thermal conductivity to experiment: a curated half-Heusler
dataset and a validated transfer function** — targeted at *Computational Materials Science*.

## Compiling

Upload this folder to Overleaf and set `manuscript.tex` as the main document. There is no LaTeX
toolchain in the repository, so the first compile happens there.

```
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

## Before you submit — run all four

From the **repository root**, not this folder:

```bash
./.venv/Scripts/python.exe make_paper_numbers.py                  # regenerate every quoted figure
./.venv/Scripts/python.exe compute_results_indomain.py            # regenerate every statistic
./.venv/Scripts/python.exe verify_citations.py paper/manuscript.tex
./.venv/Scripts/python.exe check_tex_structure.py                  # braces, refs, corrupted macros
```

The citation gate is **blocking**. It fails on an undefined `\cite` key, a DOI that no longer
resolves at Crossref or DataCite, or a citation with no supporting sentence recorded in
`claims_ledger.csv`.

## Files

| file | what it is |
|---|---|
| `manuscript.tex` | wrapper: preamble, title, front matter, conclusions, back matter |
| `section0_abstract.tex` … `section6_limitations.tex` | the body, one file per section; section 7 holds the provenance nocite blocks |
| `references.bib` | **GENERATED.** Never edit by hand — `build_bibliography.py` overwrites it |
| `extra_dois.txt` | hand-added DOIs for references not in the data pipeline |
| `claims_ledger.csv` | claim ↔ source mapping, with the sentence from each source |
| `figures/fig0*.pdf` | the eight figures; `.png` copies are for viewing, the PDFs go to the journal |
| `fig0*.py`, `figlib.py`, `paperstyle.mplstyle` | the figure code and shared style |

## Outstanding before submission

1. **Author block is a placeholder.** `manuscript.tex` carries `FIRST AUTHOR NAME` and an
   affiliation inferred from the corresponding e-mail domain. Fill both in.
2. **Three citations have metadata verified but abstracts unread** — Crossref carries none and the
   publishers return HTTP 403 to automated requests. `claims_ledger.csv` records this per row:
   - `graf2011simple` — **flagged NEEDS HUMAN CHECK.** The VEC = 18 domain restriction, on which
     the headline accuracy depends, rests on this citation. Open the paper and confirm the
     18-electron semiconducting statement is in it.
   - `katsura2025starrydata`, `curtarolo2012aflow` — confirm each supports what it is cited for.
3. **Data availability statement** needs the repository URL and DOI (deposit on acceptance).
4. **Acknowledgements** are empty: funding, computing resources, database maintainers.

## Rules that are enforced in code, not by memory

- No number is typed into the text. Everything traces to `paper_numbers.json` or
  `results_indomain.json`.
- No citation is written from model memory. Every entry in `references.bib` was resolved from a
  DOI through Crossref or DataCite.
- Withdrawn figures — 13.6, 41.4, 36.5, 79.5, 74.5, 55.7 — must not reappear. They are
  earlier headline numbers computed under scopes the paper no longer uses. The enforced list lives
  in `check_tex_structure.py`; keep the two in step.
- **33.7 is NOT withdrawn** — it is the current headline, the median over five model seeds. It was
  listed here in error, which caused a review to report the paper's own headline as a blocklisted
  number. 31.0 is seed 0, the most flattering of the five, and is the figure to avoid quoting as
  the result.
- Every statistic is computed under **one** scope: the declared domain of applicability. The
  project previously carried results from several scopes side by side, which is how a figure from
  the wrong population ends up next to one from the right one.

See `.claude/skills/manuscript/SKILL.md` for the full claims ledger and blocklist, and
`.claude/skills/citations/SKILL.md` for the citation policy.
