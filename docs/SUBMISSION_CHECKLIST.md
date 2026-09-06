# From here to submission: the whole sequence

Six stages, in this order. Stages 1 and 2 are people-and-accounts work; 3 to 6 are the paper.

| # | stage | who | how long |
|---|---|---|---|
| 1 | ORCID for all three authors | each author | 10 min each |
| 2 | Fill the last two placeholders | Shivam | 2 min |
| 3 | GitHub repository | Shivam | 15 min |
| 4 | Zenodo release → **DOI** | Shivam | 10 min |
| 5 | Overleaf compile → PDF | Shivam | 30 min |
| 6 | Submit to the journal | Shivam | 1 hour |

---

## Stage 1 — ORCID (do this first)

**ORCID is the only researcher ID you need before submitting.** It is a permanent 16-digit number,
like `0000-0002-1825-0097`, that identifies *you* rather than your current university. Elsevier asks
for it during submission.

It matters more than usual for this paper for two specific reasons:

- **Two authors share a surname.** "Randive" appears twice on this paper. ORCID is precisely the
  mechanism that keeps your publication records from merging with each other's, permanently.
- **Both of you will move.** A student email stops working after graduation, and every automated
  index that tracked you by that address loses you. ORCID follows you instead.

**How, for each of the three of you:**

1. Go to **orcid.org** → *Sign in / Register*.
2. Register with your **institutional email** (`@svnit.ac.in`, `@nitrr.ac.in`) — it is the strongest
   signal that you are who you say you are. Then add a **personal email as a second address**, so
   the account survives graduation. ORCID allows several.
3. Fill in **Employment** or **Education**: SVNIT Surat for Shivam and Prof. Pandey, NIT Raipur for
   Ojaswini. This is the "verification" part — an ORCID with no institution attached looks empty and
   carries little weight. Pick the institution from ORCID's drop-down list rather than typing it
   free-hand, so it links to the registered organisation record.
4. Set the record's visibility to **Everyone** for name and employment, or the journal cannot read it.
5. Copy the 16-digit iD and send it to Shivam.

**Prof. Pandey almost certainly already has one** — ask him rather than making a new one. Duplicate
ORCID records are a nuisance to merge later.

### What about Google Scholar and ResearchGate?

| | what it is | when |
|---|---|---|
| **ORCID** | an identity number, requested at submission | **now, before submitting** |
| **Google Scholar** | an *index* of work you have already published | **after the paper is out** |
| **ResearchGate** | a social network | optional, and honestly skip it |
| **Scopus Author ID**, **Web of Science ResearcherID** | assigned automatically when you publish | nothing to do |

A Google Scholar profile created today would be empty — it collects papers that already exist, so
there is nothing for it to collect. Make it once the paper is published; it will then find the
paper and start counting citations by itself, and you can link it to your ORCID.

ResearchGate asks nothing of you and no journal requests it. It also has a long history of authors
uploading publisher PDFs they have no right to distribute, which creates problems rather than
solving them. It is not part of this process.

## Stage 2 — the last two placeholders

```bash
python check_tex_structure.py
```

It currently names two: **Prof. Pandey's email** and the **DOI** (Stage 4 produces that). Add each
author's ORCID to `CITATION.cff` at the same time. Then re-run until it prints
`ALL STRUCTURAL CHECKS PASSED`.

---

# Stages 3 and 4 — publishing the code and getting the DOI

The paper cannot be submitted until `\repourl` in `paper/manuscript.tex` holds a real DOI.
This is how to get one. It is free and takes about twenty minutes.

Two separate jobs, and you need both:

- **GitHub** hosts the code so people can read and run it. A GitHub link is *not* permanent —
  rename or delete the repository and the link dies — so journals will not accept it alone.
- **Zenodo** (run by CERN) takes a frozen snapshot and issues a **DOI**, a permanent identifier
  that still resolves in twenty years even if the GitHub repository disappears.

---

## Before you push anything

**1. Check nothing private or copyrighted is about to go public.**

```bash
git status --short          # anything unexpected staged?
cat .gitignore              # data/ and .env must be listed
git ls-files | grep -i "\.env\|api\|key\|token\|password"   # must return nothing
```

`data/pdfs/` holds publisher PDFs downloaded under institutional subscription. **These must never
be published.** `.gitignore` already excludes `data/`, which keeps them out — but confirm it,
because a subscription breach is a serious matter for the institution, not just for you.

**2. Confirm no API keys are in the history**, not just the current files:

```bash
git log -p | grep -iE "api[_-]?key|secret|bearer|OPENALEX_API|GEMINI" | head
```

If anything appears, stop and ask before pushing — a key committed once stays in the history
even after you delete the file, and it must be rotated.

**3. Fill in the two remaining placeholders** in `paper/manuscript.tex` and `CITATION.cff`:
Ojaswini's affiliation, and (after step 3 below) the DOI. Then:

```bash
python check_tex_structure.py     # must print ALL STRUCTURAL CHECKS PASSED
```

---

## Step 1 — GitHub

1. On github.com: **New repository**. Name it something durable, e.g. `half-heusler-kappa-calibration`.
   Public. Do **not** let GitHub add a README or licence — this repository already has them.
2. From the project folder:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

3. Update the two `github.com/SHIVAM-REPO-OWNER/REPO-NAME` placeholders in `CITATION.cff` to the
   real URL, commit and push again.

GitHub will show a **"Cite this repository"** button once it sees `CITATION.cff`.

## Step 2 — connect Zenodo

1. Go to **zenodo.org** and **Log in with GitHub**. Authorise it.
2. Open <https://zenodo.org/account/settings/github/>.
3. Find the repository in the list and flip its switch **ON**.

Nothing is archived yet. Zenodo is now watching for a release.

## Step 3 — make the release, get the DOI

1. On GitHub: **Releases → Create a new release**.
2. Tag `v1.0.0`, title "Version 1.0.0 — accompanying the manuscript".
3. **Publish release.**
4. Wait a minute, then reload the Zenodo GitHub page. The repository now shows a DOI badge like
   `10.5281/zenodo.1234567`.
5. Open the Zenodo record and **Edit** it, because two things need adding by hand:
   - **ORCIDs** for all three authors. If you do not have one, get it at orcid.org — it takes two
     minutes and permanently distinguishes you from every other S. Randive who ever publishes.
   - Confirm the licence reads **CC BY 4.0** and the authors match the manuscript.
   Then **Publish** the edit.

Zenodo issues **two** DOIs. The *concept* DOI always points at the newest version; the *version*
DOI points at exactly this snapshot. **Put the version DOI in the paper**, so a referee gets
precisely the code that produced the results.

## Step 4 — put the DOI in the paper

In `paper/manuscript.tex`, replace the placeholder:

```latex
\newcommand{\repourl}{\href{https://doi.org/10.5281/zenodo.1234567}{10.5281/zenodo.1234567}}
```

Uncomment and fill the `doi:` line in `CITATION.cff` too. Then:

```bash
python check_tex_structure.py       # ALL STRUCTURAL CHECKS PASSED
python verify_citations.py paper/manuscript.tex
python check_supplementary.py
```

All three must pass before the manuscript goes to Overleaf.

---

## What is in the release, and what is deliberately not

See `LICENSE-DATA`. In short: our curated values, the prediction lists, the five model seeds
behind every seed-averaged number, and all the code — released. Bulk copies of Starrydata2, AFLOW,
Materials Project, JARVIS, OQMD and COD — **not** released; the scripts fetch them from their
public APIs instead. Publisher PDFs — never released.

Code is MIT, data is CC BY 4.0.

## Order of operations

Do the Zenodo release **before** submitting, so the DOI is live when a referee clicks it. If the
code changes during peer review, make a `v1.1.0` release; Zenodo mints a new version DOI under the
same concept DOI, and you update the paper at proof stage.

---

## Stage 5 — Overleaf

There is no LaTeX installed on this machine, so the PDF has to be built on Overleaf.

1. **overleaf.com** → *New Project* → *Upload Project* → upload the `paper/` folder as a zip.
2. *Menu* → set **Main document** to `manuscript.tex`, and **Compiler** to **pdfLaTeX**.
3. Compile. The first run will show unresolved-reference warnings; that is normal. Compile
   **three times** (pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX) so the numbering and bibliography
   settle. Overleaf's *Recompile* button usually handles this, but if `[?]` marks appear where
   citation numbers should be, run it again.
4. Upload `supplementary.tex` as a **second project** and compile it the same way. It is a
   standalone document and shares `references.bib`, so include that file in both projects.

**What to check in the built PDF, in this order:**

- No literal `TO BE ADDED` text anywhere. (`check_tex_structure.py` guards this, but look anyway.)
- No `[?]` in place of a citation number and no `??` in place of a figure or section number.
- All 8 figures appear and are legible at print size.
- Chemical formulae render as subscripts: `ZrNiSn`, not `\ce{ZrNiSn}`.
- The author list, affiliations and both corresponding-author asterisks are right.
- Tables 1, 2 and 3 have not overflowed the page margin.

## Stage 6 — submission

Computational Materials Science submits through Elsevier Editorial Manager. Have ready:

- the manuscript PDF and the source files (Elsevier wants the `.tex`, `.bib` and figures)
- the supplementary PDF, uploaded as *Supplementary Material*
- all three ORCIDs
- a **cover letter** — one page: what the paper does, why it suits this journal, and a statement
  that the work is original and not under consideration elsewhere
- **suggested reviewers** if asked. Pick authors of papers you cite who work on half-Heusler
  transport or ML for thermal conductivity, and never anyone from your own institutions.

Both declarations — competing interest, and generative AI in writing — are already in the
manuscript. Editorial Manager will also ask you to confirm them in its own form; the answers must
match what the paper says.
