# Calibrating first-principles lattice thermal conductivity to experiment

**A curated half-Heusler dataset and a validated transfer function.**

Published first-principles transport calculations of lattice thermal conductivity exceed what
laboratories measure for the same compound by a **median factor of 1.55**. That offset is regular
enough to be removed in closed form:

$$\kappa_L^{\mathrm{expt}}(T) \;=\; \kappa_L^{\mathrm{BTE}}(T)\;\min\!\left[c\left(\tfrac{T}{300}\right)^{p},\,1\right],
\qquad c = 0.51,\quad p = 0.80$$

Inside a domain of applicability declared before the validation set was scored, this places
**86.8 % of 38 held-out compounds within a factor of two** of measurement at **33.7 % median
error**, against 50.5 % uncorrected — with each compound's entire chemistry withheld from the
regression model *and* from the calibration.

Two parameters. Applicable by hand to any published transport calculation.

| | |
|---|---|
| **Paper** | `paper/manuscript.tex` (see `paper/README.md` to build it) |
| **The result** | 33.7 % median error, 86.8 % within 2×, Spearman ρ = 0.65, on 38 compounds over 18 chemistry clusters |
| **The corpus** | 52 measured half Heuslers from 164 publications, alongside 289 with transport calculations |
| **Predictions** | five issued, eleven declined with the reason each fails, ten conditional |
| **Licence** | code MIT, data CC BY 4.0 — see `LICENSE` and `LICENSE-DATA` |

---

## Reproduce the headline in three commands

```bash
pip install -e .                      # or: pip install -r requirements.txt
python analysis/run_target_blind_test.py --seed 0
python analysis/compute_seed_averaged.py
```

The last prints the per-seed table and the seed-averaged headline. Run everything **from the
repository root** — data paths are relative to it.

Full instructions, including how to rebuild the corpus from scratch, are in
**[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

## What is here

```
paper/        the manuscript, the supplementary, all 8 figures and the scripts that draw them
              paper/evidence/   per-compound scores and the five model seeds behind the headline
analysis/     the calibration, validation and prediction code -- everything the paper quotes
corpus/       rebuilds the corpus from the public APIs (Starrydata, AFLOW, MP, JARVIS, OQMD, COD)
checks/       the pre-submission gates: citations, LaTeX structure, completeness
pipeline/     the dataset builder and model wrapper the analysis imports
data/         the curated corpus, the predictions, and the results the paper quotes
docs/         how to reproduce, and the submission checklist
tests/        unit tests for the unit-canonicalisation and formula-matching logic
```

## What is deliberately not here

This repository is the κ-calibration work only. Three things are excluded on purpose:

- **Bulk copies of the upstream databases.** Starrydata2, AFLOW, Materials Project, JARVIS, OQMD
  and COD are other people's resources under their own terms. `corpus/` refetches them from the
  public APIs instead. See `LICENSE-DATA`.
- **Publisher PDFs and full-text XML.** Downloaded under institutional subscription and not ours
  to redistribute.
- **The broader Heusler mining pipeline.** An earlier effort extracted magnetic, mechanical and
  structural properties too. It contributed 31 of 4,094 experimental rows here — 0.76 % — and none
  of its code is on this paper's path, so it is not part of this release.

## How the two scales are kept apart

The single property that makes the corpus work: **measurements and calculations are never pooled.**
Every value carries a method tier — 0 experimental, 1 full Boltzmann transport, 2 machine-learned
potential, 3 semi-empirical — assigned from the *method string the source reports*, not from the
repository it came from. The two populations meet at exactly one place, the transfer function.

That distinction is load-bearing rather than pedantic. Of 505 half Heuslers carrying a nominally
first-principles label, only 289 carry a method string that positively identifies a transport
calculation; 214 are semi-empirical estimates, and semi-empirical and transport values for the same
compound differ by a median factor of 1.38 with a tail to 7.2.

## Citing

See `CITATION.cff`. Please cite both the paper and the archived release. If you use the data you
are also using the upstream sources above; the manuscript's reference list names every publication
behind every measurement so that they can be cited too.

## Limitations worth knowing before you use it

The method predicts κ_L **at a stated temperature**, not the shape of κ_L(T) — the correction
carries a single global exponent and the measured temperature slope is not reproduced. Its
parameters are fitted, not constants of nature. And it applies to semiconducting (VEC = 18), cubic,
non-polymorphic half Heuslers; outside that domain the median error is 105 %, which we report
rather than conceal. Section 3.7 of the paper states each limitation with the number that
quantifies it.
