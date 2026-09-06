# Reproducing this work

Two routes. Almost everyone wants the first.

| route | needs | takes |
|---|---|---|
| **A — reproduce the paper** from the curated data | Python 3.10+, the shipped CSVs | minutes |
| **B — rebuild the corpus** from the public APIs | network, and API keys for some sources | hours to days |

Run everything **from the repository root**. Data paths inside the scripts are relative to it.

```bash
pip install -e .          # installs pipeline/ and the dependencies
```

---

## Route A — reproduce the paper

### The headline in two commands

```bash
python analysis/run_target_blind_test.py --seed 0
python analysis/compute_seed_averaged.py
```

The second prints a per-seed table and the seed-averaged headline: **33.7 % median error, 86.8 %
within a factor of two**, over 38 in-domain compounds.

### Why the headline is a median over five seeds

The regression model has random components, and the median error moves from 31.0 % to 40.0 %
depending on the seed. A single-seed number reports the seed as much as the method, so every
model-dependent figure in the paper is the median over seeds 0–4, quoted with its range. To
regenerate all five:

```bash
for s in 0 1 2 3 4; do
  python analysis/run_target_blind_test.py --seed $s --out paper/evidence/blind_d2_s$s.csv
done
python analysis/compute_seed_averaged.py
```

Each seed refits a model per chemistry cluster, so this takes a while. `compute_seed_averaged.py`
reads seed 0 from `data/exports/kappa_v2/target_blind_test.csv` and seeds 1–4 from the
`paper/evidence/blind_d2_s*.csv` files.

Expected per-seed values, for checking your run:

| seed | median error | within 2× | within 30 % | bias |
|---|---|---|---|---|
| 0 | 31.0 % | 86.8 % | 47.4 % | 1.00 |
| 3 | 33.7 % | 81.6 % | 42.1 % | 0.91 |

### The rest of the numbers

```bash
python analysis/run_loco_chemistry.py            # leave-one-chemistry-out validation
python analysis/extend_blind_test.py             # fits c and p; the calibrated blind test
python analysis/compute_conformal_indomain.py    # the 1.31 / 1.92 / 2.24 interval bands
python analysis/compute_null_baselines.py        # the two compound-blind nulls
python analysis/compute_family_null.py           # the family power-law baseline
python analysis/compute_baselines_ablation.py    # Slack/Debye-Callaway, and the exponent ablation
python analysis/compute_results_indomain.py      # the per-family calibration test
python analysis/compute_interlab_ceiling.py      # inter-laboratory reproducibility of the data
python analysis/make_paper_predictions.py        # Table 2: the five issued predictions
python analysis/compute_conditional.py           # Table S1: the ten conditional estimates
python analysis/make_paper_numbers.py            # the registry the manuscript quotes from
```

`analysis/make_paper_numbers.py` writes `data/exports/kappa_v2/paper_numbers.json`. Every number in the
manuscript should be traceable to it or to `paper/evidence/evidence_chain.csv`.

### The figures

```bash
python paper/fig01_corpus.py        # ... through fig08_method.py
```

They write into `paper/figures/`. Run them from the repository root, not from `paper/`.

### The gates

```bash
python checks/check_tex_structure.py
python checks/verify_citations.py paper/manuscript.tex
python checks/check_supplementary.py
python checks/audit_completion.py
```

`check_tex_structure.py` fails while the submission placeholders are unfilled. That is deliberate.

---

## Route B — rebuild the corpus

Only if you want to regenerate `data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv` from source
rather than use the shipped copy.

```bash
pip install -e ".[corpus]"
```

Run in this order — the chain has no single driver, and each step consumes the previous one's
output:

```bash
python corpus/harvest_starrydata.py            # experimental kappa(T) curves  (the tier-0 half)
python corpus/harvest_starrydata_doped.py
python corpus/harvest_aflow.py                 # AFLOW AGL, tier-3 coverage
python corpus/harvest_mp.py                    # Materials Project elastic/structure
python corpus/harvest_phonix.py                # published tier-1 BTE rows
python corpus/expand_theory_pool.py
python corpus/mine_gap_chemistry.py

python corpus/build_kappa_pool.py              # merge + assign method_tier
python corpus/resolve_structures.py            # attach a sourced structure, or exclude
python corpus/fetch_structures_optimade.py     # recover structures for the excluded
python corpus/build_training_set.py            # -> HEUSLER_KAPPA_TRAINING_SET.csv
python corpus/add_recovered_compounds.py       # 31 rows; uses the frozen CSV, no database needed
python corpus/apply_audit_corrections.py       # the resistivity gate; MUST run after the above
python corpus/run_perchem_tier_experiment.py   # -> HEUSLER_KAPPA_TIER3_GAP_ROWS.csv
```

Two things about that list are worth stating plainly rather than leaving you to discover them:

- **`apply_audit_corrections.py` rewrites the training set in place.** It applies the
  Wiedemann-Franz resistivity gate that removes 136 rows, including all 47 rows of `TiCoSn`. It
  must run after `build_training_set.py` and `add_recovered_compounds.py`, and before anything
  else. Nothing enforces that ordering.
- **`run_perchem_tier_experiment.py` is named as an experiment but is a required build step.** It
  is the only writer of `HEUSLER_KAPPA_TIER3_GAP_ROWS.csv`, which thirteen scripts read — including
  the blind test. Skip it and the blind test fails on a missing file.

### What Route B does not include

The **LLM extraction** described in Section 2.1 of the paper needs a Gemini API key and access to
the source publications, which we cannot redistribute:

```bash
pip install -e ".[extract]"     # then see pipeline/s04_extract.py and prompts/extraction_system.md
```

It contributed 31 of 4,094 experimental rows — 0.76 %. `pipeline/s04_extract.py` carries the
3000-byte source-length gate that stops the model being handed a title-only stub, which it will
otherwise answer with fluent and entirely fabricated measurements. `corpus/quarantine_stubs.py`
flags the affected rows. Both are shipped because the paper describes the method, and a method
whose code is withheld is not a method a referee can check.

The **upstream bulk databases** — Starrydata2, AFLOW, Materials Project, JARVIS, OQMD, COD — are
fetched by the harvest scripts rather than mirrored here. See `LICENSE-DATA`.

---

## If something fails

| symptom | cause |
|---|---|
| `ModuleNotFoundError: pipeline` | not run from the repository root, or `pip install -e .` not done |
| `FileNotFoundError: ...TIER3_GAP_ROWS.csv` | run `corpus/run_perchem_tier_experiment.py` |
| `ModuleNotFoundError: catboost` / `sklearn` | install the base dependencies, not just the corpus extras |
| figure script writes nothing | run it from the repository root, not from inside `paper/` |
| `check_tex_structure.py` says PROBLEMS FOUND | expected until the submission placeholders are filled |
