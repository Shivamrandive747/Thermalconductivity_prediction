"""Blind test on the TARGET LIST: predict the targets we already have data for.

THE TEST THAT WAS MISSING. Every accuracy number in this project so far came from a 24-compound
validation set or from cross-validation over the training pool. But 154 of the 258 stoichiometric
Heusler targets already carry a kappa value -- 64 experimental, 128 full-BTE -- so they can be held
out, predicted, and scored against a real number. That is a direct measurement of how well we
predict the actual deliverable, and it had never been run.

HOW THE HOLDOUT WORKS. Removing just the compound is not enough: `ZrCoSb` sits among 40 other
Co-site half Heuslers, so leaving those in makes the prediction interpolation. Each fold removes
THE WHOLE CHEMISTRY CLUSTER (X2 element for full Heuslers, X-site element for half), reusing
`cluster_of()` from run_loco_chemistry.py, and asserts the cluster is really gone.

THREE RULES THAT KEEP THE NUMBER HONEST:

  1. SCORE PER COMPOUND, NEVER PER ROW. The previously reported 36.5% was a per-row median; on the
     same data the per-compound median was 84.3%, because compounds with many temperature points
     dominated the row pool.
  2. EXPERIMENT AND DFT ARE NEVER POOLED. Agreement with a first-principles calculation and
     agreement with a measured sample are different claims and get separate tables.
  3. A LOW REFERENCE IS FLAGGED AND STILL COUNTED. Compounds whose MEDIAN reference sits below
     0.5 W/m/K are listed for the reader's judgement but remain in every statistic. Two earlier
     versions of this rule were wrong in opposite directions: one dropped them from the printed
     figures despite this docstring promising otherwise (which flattered the result, and would have
     discarded `MgAgSb` at 0.70 W/m/K -- a real ultralow-kappa half Heusler, not an artefact), and
     the flag itself fired on `.any()` over temperature rows, so a single low-temperature point
     condemned `VCoSn` even though its median reference is 6.55 W/m/K.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import make_model, metrics
from run_loco_chemistry import cluster_of, sites

GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT_CSV = "data/exports/kappa_v2/target_blind_test.csv"
OUT_JSON = "data/exports/kappa_v2/target_blind_test.json"
SANITY_FLOOR = 0.5          # W/m/K -- below this, a bulk Heusler reference is suspect


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def target_set() -> dict:
    """Reduced formula -> class, for stoichiometric Heusler targets only."""
    tg = set()
    for f in glob.glob("data/Target_Materials/*.xlsx") + glob.glob("data/Target_Materials/*.csv"):
        if any(k in f for k in ("DOPED", "TIER3", "TRAINING_SET", "Validation", "HUNTING",
                                "NOVEL", "RESULTS", "Target_Scope", "DOMAIN", "REVERIFIED",
                                "blind")):
            continue
        try:
            sheets = ([pd.read_csv(f)] if f.endswith(".csv")
                      else [pd.ExcelFile(f).parse(n) for n in pd.ExcelFile(f).sheet_names])
        except Exception:  # noqa: BLE001
            continue
        for d in sheets:
            for c in ("compound", "formula", "reduced_formula"):
                if c in d.columns:
                    tg |= {str(x).strip() for x in d[c].dropna()}
    out = {}
    for x in tg:
        s = sites(x)
        if not s:
            continue
        try:
            if not all(abs(v - round(v)) < 0.02
                       for v in Composition(x).get_el_amt_dict().values()):
                continue
        except Exception:  # noqa: BLE001
            continue
        r = red(x)
        if r:
            out[r] = s[0]
    return out


def y_family(f) -> str | None:
    """The middle-site element for a half Heusler -- the family that predicts accuracy."""
    s = sites(f)
    if not s or s[0] != "half":
        return None
    try:
        a = sorted(Composition(str(f)).get_el_amt_dict().items(), key=lambda kv: -kv[1])
        els = sorted(((e, Composition(e).elements[0].X or 99) for e, _ in a), key=lambda z: z[1])
        return els[1][0]
    except Exception:  # noqa: BLE001
        return None


def main(model: str = "catboost", tier: int = 0, seed: int = 0,
         out: str | None = None) -> int:
    # `seed` and `out` exist so the five model seeds behind the paper's seed-averaged headline can
    # be REGENERATED, not merely verified. Before this, the four non-zero seeds in paper/evidence/
    # had no producer in the repository: compute_seed_averaged.py globs blind_d2_s*.csv, but
    # nothing wrote s1..s4. Defaults reproduce the previous behaviour exactly (seed 0, fixed path).
    out_csv = out or OUT_CSV
    out_json = (out_csv[:-4] if out_csv.endswith(".csv") else out_csv) + ".json"

    targets = target_set()
    print(f"stoichiometric Heusler targets: {len(targets)}")
    print(f"model seed: {seed}")

    # ---- the reference values we will be scored against -------------------------
    tr = pd.read_csv(TRAIN)
    tr["red"] = tr.formula.map(red)
    t = pd.to_numeric(tr.method_tier, errors="coerce")
    ref = tr[(t == tier) & tr.red.isin(targets)].copy()
    ref["kappa_L"] = pd.to_numeric(ref.kappa_L, errors="coerce")
    ref["temperature_K"] = pd.to_numeric(ref.temperature_K, errors="coerce")
    ref = ref[ref.kappa_L.notna() & (ref.kappa_L > 0) & ref.temperature_K.notna()]
    label = "EXPERIMENT (tier 0)" if tier == 0 else f"full-BTE (tier {tier})"
    print(f"targets with {label} reference data: {ref.red.nunique()}")

    # ---- the model: production settings, tier-1 labels --------------------------
    b = ds.build(tier_max=1, tier_min=1, group_by="element_system",
                 extra_csv=GAP, verbose=False)
    forms = b["meta"].formula.values
    clus = np.array([cluster_of(f) for f in forms])
    X, y, w = b["X"], b["y"], b["weights"]
    reds = np.array([red(f) for f in forms])
    print(f"training pool: {len(y)} rows, {len(set(forms))} compounds, "
          f"{len(set(clus))} chemistry clusters")

    # group the references to one value per compound and temperature bin
    ref["Tbin"] = (ref.temperature_K / 100).round() * 100
    per_cond = (ref.groupby(["red", "Tbin"])
                .agg(k_ref=("kappa_L", "median"), n_rows=("kappa_L", "size"),
                     formula=("formula", "first")).reset_index())
    print(f"reference conditions (compound x temperature bin): {len(per_cond)}")

    # ---- one model per chemistry cluster, holding the whole cluster out ---------
    need = {}
    for r in per_cond.red.unique():
        f0 = per_cond[per_cond.red == r].formula.iloc[0]
        need.setdefault(cluster_of(f0), []).append(r)
    print(f"clusters to fit: {len(need)}\n")

    rows = []
    fallback_used: list = []   # targets whose descriptors had to be rebuilt outside build()
    for i, (cl, members) in enumerate(sorted(need.items()), 1):
        te = clus == cl
        trm = ~te
        if trm.sum() < 50:
            print(f"  [{i}/{len(need)}] {cl:<22} SKIP -- only {trm.sum()} training rows left")
            continue
        assert not (clus[trm] == cl).any(), f"cluster {cl} leaked into training"
        # CatBoost names its seed random_seed, LightGBM random_state; make_model merges **p last,
        # so this overrides the built-in default of 0 rather than fighting it.
        seed_key = "random_seed" if model == "catboost" else "random_state"
        mdl = make_model(model, {seed_key: seed})
        mdl.fit(X[trm], y[trm], sample_weight=w[trm])
        for r in members:
            sub = per_cond[per_cond.red == r]
            # build a feature row per temperature bin from this compound's own descriptors
            src = None
            idx = np.where(reds == r)[0]
            if len(idx):
                src = X.iloc[idx[0]].copy()
            else:
                # FALLBACK. 21 of the 64 targets are experimental-only, so they never appear in the
                # tier-1 training matrix and their descriptors must be rebuilt here. An earlier
                # version took `tr[tr.red == r].iloc[0]` -- an arbitrary row at ANY method tier,
                # unlike build(), which picks its structure from tier-filtered rows. Worse, if that
                # row happened to lack struct_species/struct_frac_coords, the four bond-geometry
                # features silently came out NaN and the compound was predicted on 41 of 45
                # features. So: prefer a row that HAS coordinates, then the most authoritative tier.
                cand = tr[tr.red == r].copy()
                cand["_has_xyz"] = (cand.struct_species.notna()
                                    & cand.struct_frac_coords.notna())
                cand["_t"] = pd.to_numeric(cand.method_tier, errors="coerce").fillna(9)
                cand = cand.sort_values(["_has_xyz", "_t"], ascending=[False, True],
                                        kind="mergesort")
                if not len(cand):
                    continue
                s = cand.iloc[0]
                # SUPPLY THE C1b COORDINATES WHEN THE SOURCE ROW LACKS THEM.
                #
                # Without `struct_species` AND `struct_frac_coords`, `_bond_stats` returns four
                # NaNs, so six of the 49 scored half Heuslers -- BaAgSb, LaSbPd, NbCoSb, VCoSb,
                # YSbPd and TiGeRu2 -- were predicted on 41 of 45 features. The block above
                # already PREFERS a row with coordinates and the print below already REPORTS the
                # shortfall, but nothing repaired it, so the compounds were scored anyway.
                #
                # There is nothing to infer. For an ideal C1b the coordinate set is fixed by the
                # prototype -- X at 4a (0,0,0), Y at 4c (1/4,1/4,1/4), Z at 4b (1/2,1/2,1/2) --
                # and the resulting features are a deterministic function of the lattice constant:
                # bond_min = bond_mean = a*sqrt(3)/4, bond_std = 0, bond_ratio = 1. 100% of the
                # half-Heusler rows that DO carry coordinates agree with this.
                #
                # Leaving them NaN is not neutral: the model's `nan_mode="Min"` treats a missing
                # value as the smallest, i.e. as a claim that these are the shortest bonds in the
                # dataset. Measured cost on the seed-0 blind test: per-compound median 50.30% ->
                # 38.45%, within-2x 73.5% -> 77.6%, within-30% 24.5% -> 36.7%. LaSbPd alone goes
                # 50.3% -> 14.8%. The affected compounds were also dragging the shared
                # leave-one-cluster-out calibration, which is why 59% of ALL 49 improve, not just
                # the six. The paired Wilcoxon is p=0.30 -- but this is a defect being removed,
                # not an intervention being adopted, so it does not need to clear significance.
                _sp, _co = s.struct_species, s.struct_frac_coords
                if pd.isna(_sp) or pd.isna(_co):
                    _els = sorted(Composition(str(s.formula)).elements,
                                  key=lambda e: (e.X if e.X else 99.0))
                    if len(_els) == 3 and targets.get(r) == "half":
                        _sp = str([str(e) for e in _els])
                        _co = "[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]"
                        s = s.copy()
                        s["struct_species"], s["struct_frac_coords"] = _sp, _co
                f = ds._per_formula_features(s.formula, s.struct_a_A, s.struct_prototype,
                                             s.struct_species, s.struct_frac_coords)
                if f is None:
                    continue
                src = pd.Series({c: f.get(c, np.nan) for c in X.columns})
                missing = [c for c in X.columns if c not in ("temperature_K", "inv_T", "log_T")
                           and pd.isna(src.get(c))]
                fallback_used.append((r, int(s._t), len(missing)))
            for _, cond in sub.iterrows():
                q = src.copy()
                q["temperature_K"] = float(cond.Tbin)
                q["inv_T"] = 1000.0 / float(cond.Tbin)
                q["log_T"] = np.log10(float(cond.Tbin))
                pred = float(10 ** mdl.predict(pd.DataFrame([q])[X.columns])[0])
                rows.append(dict(compound=r, cluster=cl, T=float(cond.Tbin),
                                 k_ref=float(cond.k_ref), k_pred=round(pred, 4),
                                 n_ref_rows=int(cond.n_rows),
                                 klass=targets.get(r), Yfam=y_family(cond.formula)))
        if i % 10 == 0 or i == len(need):
            print(f"  [{i}/{len(need)}] {cl:<22} cumulative predictions: {len(rows)}", flush=True)

    if fallback_used:
        reduced = [(f, t, n) for f, t, n in fallback_used if n]
        print(f"\n  descriptors rebuilt outside the training matrix for "
              f"{len(fallback_used)} experimental-only targets")
        if reduced:
            print(f"  of those, {len(reduced)} still lack some geometry features "
                  f"(predicted on fewer than {len(X.columns)} inputs):")
            for f, t, n in sorted(reduced, key=lambda z: -z[2]):
                print(f"    {f:<11} tier {t}   {n} feature(s) missing")
        else:
            print("  all of them got a complete feature set")

    D = pd.DataFrame(rows)
    if not len(D):
        print("no predictions produced")
        return 1
    D["ratio"] = D.k_pred / D.k_ref
    D["ape"] = 100 * (D.k_pred - D.k_ref).abs() / D.k_ref
    D["suspect_ref"] = D.k_ref < SANITY_FLOOR
    D.to_csv(out_csv, index=False)

    # ---- PER COMPOUND, never per row -------------------------------------------
    g = (D.groupby(["compound", "klass", "Yfam"], dropna=False)
         .agg(n_cond=("ape", "size"), k_ref=("k_ref", "median"), k_pred=("k_pred", "median"),
              ape=("ape", "median"), ratio=("ratio", "median"),
              n_low_rows=("suspect_ref", "sum")).reset_index())
    # FLAG ON THE VALUE ACTUALLY BEING SCORED, which is the compound's MEDIAN reference.
    # The previous version used `suspect=("suspect_ref", "any")`, so ONE low-temperature row
    # condemned an entire compound: VCoSn was reported as "reference below 0.5 W/m/K" while its
    # median reference is 6.55 W/m/K, because a single point dipped low. The printed reason then
    # contradicted the printed number.
    g["suspect"] = g.k_ref < SANITY_FLOOR

    print("\n" + "=" * 92)
    print(f"BLIND TEST vs {label} -- scored PER COMPOUND, whole chemistry withheld")
    print("=" * 92)
    flagged = g[g.suspect]
    # KEPT IN THE STATISTICS. Rule 3 of this script's own docstring says a low reference is flagged
    # and not used silently -- but the earlier code built `clean = g[~g.suspect]` and quietly
    # dropped them from every printed figure, which is the opposite. `MgAgSb` at 0.70 W/m/K is a
    # real ultralow-kappa half Heusler, not an artefact, and excluding it flatters the result.
    clean = g
    if len(flagged):
        print(f"\n  FLAGGED -- median reference below {SANITY_FLOOR} W/m/K. KEPT in all statistics"
              f" below; listed so a reader can judge each one:")
        for _, r in flagged.iterrows():
            print(f"    {r.compound:<11}ref {r.k_ref:.3f}  pred {r.k_pred:.2f}  "
                  f"({r.ape:.0f}% error)")
    low_rows = g[(~g.suspect) & (g.n_low_rows > 0)]
    if len(low_rows):
        print(f"\n  compounds whose MEDIAN reference is fine but which have some low-temperature "
              f"rows below {SANITY_FLOOR} W/m/K (not flagged, reported for transparency):")
        for _, r in low_rows.iterrows():
            print(f"    {r.compound:<11}median ref {r.k_ref:.2f}   "
                  f"{int(r.n_low_rows)} of {int(r.n_cond)} conditions below the floor")

    for kl in ("half", "full"):
        s = clean[clean.klass == kl]
        if not len(s):
            continue
        w2 = s.ratio.between(0.5, 2.0)
        print(f"\n  --- {kl.upper()} HEUSLERS: {len(s)} compounds ---")
        print(f"    median error   {s.ape.median():>7.1f}%")
        n30 = int((s.ape < 30).sum())
        print(f"    within 30%     {n30:>3} / {len(s)}"
              f"   ({100 * (s.ape < 30).mean():.0f}%)")
        print(f"    within 2x      {int(w2.sum()):>3} / {len(s)}   ({100 * w2.mean():.0f}%)")
        print(f"    median ratio   {s.ratio.median():>7.2f}x")
        if kl == "half":
            print("\n    by Y-site family:")
            for fam, gg in s.groupby("Yfam"):
                if len(gg) >= 2:
                    print(f"      {str(fam):<4} n={len(gg):<3} median {gg.ape.median():>6.1f}%"
                          f"   within2x {100 * gg.ratio.between(0.5, 2.0).mean():>3.0f}%")

    print(f"\n  --- best predicted (all classes) ---")
    for _, r in clean.nsmallest(15, "ape").iterrows():
        print(f"    {r.compound:<11}{str(r.klass):<6}ref {r.k_ref:>7.2f}  pred {r.k_pred:>7.2f}"
              f"  {r.ape:>6.1f}%  {r.ratio:>5.2f}x")

    summary = {}
    for kl in ("half", "full"):
        s = clean[clean.klass == kl]
        if len(s):
            summary[kl] = dict(n=int(len(s)), median_ape=float(s.ape.median()),
                               within_30=float((s.ape < 30).mean()),
                               within_2x=float(s.ratio.between(0.5, 2.0).mean()),
                               median_ratio=float(s.ratio.median()))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(dict(
        reference=label, tier=tier, n_compounds=int(len(clean)),
        n_flagged=int(len(flagged)), by_class=summary), indent=2))
    print(f"\nwrote {out_csv}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost")
    ap.add_argument("--tier", type=int, default=0,
                    help="0 = score against EXPERIMENT, 1 = score against full-BTE")
    ap.add_argument("--seed", type=int, default=0,
                    help="model seed. The paper averages over 0-4; see compute_seed_averaged.py")
    ap.add_argument("--out", default=None,
                    help="output CSV path (the .json sibling is derived). "
                         "Default: data/exports/kappa_v2/target_blind_test.csv")
    raise SystemExit(main(**vars(ap.parse_args())))
