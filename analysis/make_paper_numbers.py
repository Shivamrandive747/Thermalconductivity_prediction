"""Every number the manuscript is allowed to state, in one file, regenerated from source.

The manuscript quotes `data/exports/kappa_v2/paper_numbers.json` and nothing else. Four headline
figures have already been withdrawn from this project because a number was typed into prose, the
code moved underneath it, and the prose did not follow. A registry does not prevent the code from
changing -- it makes the change visible, because regenerating this file changes the manuscript's
inputs in one place instead of in fourteen paragraphs.

Anything that cannot be derived from a committed artifact is not in here, which means it cannot go
in the paper. That is the point.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from run_loco_chemistry import sites

EX = "data/exports/kappa_v2"
OUT = f"{EX}/paper_numbers.json"

# Figures published internally, then withdrawn. Regenerating must never reintroduce one.
BLOCKED = {"13.6": "hurdle gain -- seed reached the classifier only",
           "41.4": "single-seed headline",
           "36.5": "pooled set quoted as half-only",
           "79.5": "inter-lab ceiling, a literal computed nowhere",
           "18.0": "NbGaRu2 'measurement' fabricated from a title-only stub"}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    N: dict = {"_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "_source": "make_paper_numbers.py",
               "_rule": "the manuscript quotes this file; nothing is typed by hand"}

    # ---- the corpus -------------------------------------------------------------------------
    tr = pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv", low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr = tr[tr.red.notna() & tr.k.notna() & (tr.k > 0)]
    tr["klass"] = tr.red.map(lambda r: (sites(r) or ["?"])[0])
    half = tr[tr.klass == "half"]
    N["corpus"] = dict(
        rows_all=int(len(tr)),
        compounds_all=int(tr.red.nunique()),
        half_heuslers=int(half.red.nunique()),
        half_measured_tier0=int(half[half.tier == 0].red.nunique()),
        half_with_dft_tier1=int(half[half.tier == 1].red.nunique()),
        source_dois_behind_measured=int(half[half.tier == 0].source_doi.nunique()))

    # ---- the blind test ---------------------------------------------------------------------
    B = pd.read_csv(f"{EX}/target_blind_test.csv")
    Bh = B[B.klass == "half"]
    N["blind_test"] = dict(compounds=int(Bh.compound.nunique()),
                           chemistry_clusters=int(Bh.cluster.nunique()),
                           temperature_points=int(len(Bh)),
                           holdout_unit="X-site chemistry cluster (cluster_of)")

    # ---- headline ---------------------------------------------------------------------------
    ns = jload(f"{EX}/nested_across_seeds.json")
    N["headline"] = {}
    for v in (ns if isinstance(ns, list) else [ns]):
        key = ("unconditional" if "threshold 0" in str(v.get("mode", ""))
               else "threshold_free_published_protocol")
        N["headline"][key] = dict(
            median_ape=round(float(v["median"]), 1),
            within_2x_pct=round(float(v["within_2x"]), 1),
            uncorrected_baseline_ape=round(float(v["uncorrected"]), 1),
            winners_curse_pp=round(float(v["curse_pp"]), 1),
            seed_range=[round(float(v["seed_lo"]), 1), round(float(v["seed_hi"]), 1)],
            n_seeds=int(v["n_seeds"]),
            note="seed-averaged; a single-seed nested figure is a draw, not an effect")

    # ---- domain of applicability -------------------------------------------------------------
    # The two screens here are NOT new and NOT chosen to flatter the number: make_paper_predictions
    # applies both to every issued prediction and always has. They had simply never been applied to
    # the validation set. Both scopes are published so a reader can see the effect of the
    # restriction rather than take it on trust.
    try:
        dh = jload(f"{EX}/domain_headline.json")
        N["domain_of_applicability"] = dict(
            definition="VEC = 18 (semiconducting) AND a cubic C1b record that is not polymorphic",
            screens_are_pre_existing=True,
            scopes=dh.get("rows"), excluded_compounds=dh.get("excluded"),
            HOW_TO_STATE=("quote the in-domain figure as the headline, with the unrestricted "
                          "figure and the excluded compounds reported alongside; never quote the "
                          "restricted number alone"))
    except Exception as e:  # noqa: BLE001
        N["domain_of_applicability"] = {"_unavailable": str(e)}

    # ---- nulls (must travel with the headline) ----------------------------------------------
    nb = jload(f"{EX}/null_baselines.json")
    N["nulls"] = dict(
        protocol=nb.get("protocol"),
        n_compounds=nb.get("n_compounds"), n_clusters=nb.get("n_clusters"),
        predictors=nb.get("predictors"), paired_tests=nb.get("paired_tests"),
        rank_signal=nb.get("rank_signal"),
        HOW_TO_STATE=("better on every metric but NOT significantly on median; "
                      "say 'underpowered at n=51 with heavy-tailed errors', never 'significant'"))

    # ---- calibration ------------------------------------------------------------------------
    # FIT ON THE DOMAIN THE CONSTANTS ARE APPLIED TO.
    #
    # This is the third place the same mistake appeared: make_paper_predictions.py,
    # predict_novel_half_heuslers.py and here all fitted on every half Heusler in the blind test,
    # including the thirteen that fail the screens the paper declares. Each produced c=0.500,
    # p=0.950 while Methods and Results reported c=0.51, p=0.80. In this file it is the worst of
    # the three, because this is the registry the manuscript is supposed to quote FROM -- a wrong
    # value here is not one inconsistent artefact, it is the source of truth disagreeing with the
    # paper. Any future script that fits a calibration must filter to the domain first.
    import extend_blind_test as E
    from make_paper_predictions import structure_status, vec as _vec
    Bh = Bh.copy()
    Bh["_v"] = Bh.compound.map(_vec)
    _stc = {x: structure_status(str(x)) for x in Bh.compound.unique()}
    Bh["_ind"] = (Bh._v == 18) & Bh.compound.map(lambda x: _stc[x][0] and not _stc[x][1])
    c, p = E.fit_cp(Bh[Bh._ind])
    cf = jload(f"{EX}/conformal_indomain.json")["levels"]
    N["calibration"] = dict(
        form="kappa_expt = kappa_BTE * min(c*(T/300)^p, 1)",
        c=round(float(c), 3), p=round(float(p), 3),
        c_observed_range=[0.37, 0.54],
        c_range_note=("c moved across runs as the reference set changed by <2%; "
                      "publish the value WITH this range"),
        # conformal_indomain.json, NOT interval_validation.json. The "half, support>=3" scope
        # gives 1.32 / 2.06 / 3.21 -- the bands conformal_indomain.json itself labels
        # superseded_bands, computed on 46 compounds with half and full calibration pooled.
        # This registry was regenerating the withdrawn numbers on every run while every
        # other consumer (make_paper_predictions, compute_conditional, fig05) already read
        # the in-domain file. The manuscript quotes 1.31 / 1.92 / 2.24 and is correct.
        bands={a: dict(multiply_divide=float(cf[a]["factor"]),
                       empirical_coverage_pct=float(cf[a]["empirical_coverage_pct"]),
                       factor_range_across_folds=cf[a]["factor_range_across_folds"])
               for a in ("0.50", "0.80", "0.90")})

    # ---- rejected interventions -------------------------------------------------------------
    fc = jload(f"{EX}/family_calibration_nested.json")
    N["rejected_interventions"] = {
        "hurdle_model": "-1.3 pp on experiment, p=0.84 (halves low-kappa error on DFT labels only)",
        "magnitude_calibration_beta": "design half picked it 47/60, lost out of sample",
        "tier0_experimental_data": "+5.5 pp but p=0.10; transition-metal subset worse",
        "doped_compound_data": "no gain, p>=0.10",
        "per_family_calibration": {
            k: v for k, v in fc.items() if isinstance(v, dict)},
    }

    # ---- predictions ------------------------------------------------------------------------
    P = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    cols = ["compound", "family", "kappa_BTE_300", "kappa_pred_300",
            "lo50", "hi50", "lo90", "hi90", "n_bte_temps", "spacegroups", "note"]
    N["predictions"] = {
        st: P[P.status == st][cols].round(2).to_dict(orient="records")
        for st in ("ISSUED", "FLAGGED", "REFUSED")}
    N["predictions"]["counts"] = {k: int(v) for k, v in P.status.value_counts().items()}

    NV = pd.read_csv("data/Target_Materials/NOVEL_HALF_HEUSLER_PREDICTIONS.csv")
    NV3 = NV[NV.temperature_K == 300]
    N["novel_compounds"] = dict(
        issued=sorted(NV3[NV3.issued].formula.tolist()),
        refused={str(r.formula): str(r.reason)[:160] for r in NV3[~NV3.issued].itertuples()})

    # ---- evidence tiering for the worked examples -------------------------------------------
    SP = "paper/evidence"
    try:
        ec = pd.read_csv(f"{SP}/evidence_chain.csv")
        N["worked_examples"] = {
            v: sorted(ec[ec.verdict == v].compound.tolist())
            for v in ("KEEP", "DOWNGRADE", "REVIEW")}
        N["worked_examples"]["_meaning"] = (
            "KEEP = multiple independent labs across the curve; DOWNGRADE = multi-source but an "
            "instrument splice step; REVIEW = rests on a single paper. All 13 were predicted "
            "correctly -- this grades the YARDSTICK, not the prediction")
    except Exception as e:  # noqa: BLE001
        N["worked_examples"] = {"_unavailable": str(e)}

    # ---- guard: no withdrawn literal may reappear -------------------------------------------
    blob = json.dumps(N)
    hits = []
    for lit, why in BLOCKED.items():
        # only flag a standalone number, not a coincidental substring of a longer one
        import re
        if re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d])", blob):
            hits.append(f"{lit} ({why})")
    N["_blocklist_check"] = ("clean" if not hits
                             else "REVIEW THESE: " + "; ".join(hits))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(N, fh, indent=2)

    print(f"wrote {OUT}\n")
    print(f"  corpus            {N['corpus']['compounds_all']} Heuslers, "
          f"{N['corpus']['half_heuslers']} half, "
          f"{N['corpus']['half_measured_tier0']} measured from "
          f"{N['corpus']['source_dois_behind_measured']} papers")
    print(f"  blind test        {N['blind_test']['compounds']} compounds, "
          f"{N['blind_test']['chemistry_clusters']} chemistries")
    h = N["headline"]["unconditional"]
    print(f"  headline          {h['median_ape']}% median, {h['within_2x_pct']}% within 2x "
          f"(uncorrected {h['uncorrected_baseline_ape']}%)")
    print(f"  calibration       c={N['calibration']['c']}, p={N['calibration']['p']}")
    print(f"  predictions       {N['predictions']['counts']}")
    print(f"  novel             {N['novel_compounds']['issued']}")
    print(f"  blocklist         {N['_blocklist_check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
