"""The honest headline: accuracy measured on compounds that took no part in ANY design choice.

WHY THIS EXISTS. Every number reported so far was measured on the same 64 compounds that were used
to choose the correction's functional form (8+ candidates were compared by their accuracy on these
compounds), its fitting objective (rows vs per-compound), and the chemistry-support threshold
(MIN_SUPPORT=3 was picked after tabulating which compounds it removed and what that did to the
median). Those are three model-selection decisions made on the evaluation set. The resulting figure
is therefore optimistic by an unknown amount, and "unknown" is not publishable.

This script measures the amount. Each repeat:

  1. splits the compounds 50/50 into a DESIGN half and a TEST half,
  2. on the DESIGN half only: sweeps functional form, objective and support threshold, and picks
     the combination that scores best there,
  3. freezes every choice and applies it, untouched, to the TEST half,
  4. records the TEST-half accuracy.

The distribution of step 4 over many repeats is the honest estimate. The gap between it and the
design-half score is the winner's curse, measured rather than assumed.

THE SPLIT IS BY CHEMISTRY CLUSTER, NOT BY COMPOUND. This is the subtle part. The base predictions
come from one model per chemistry cluster (`run_target_blind_test.py` fits a model per cluster and
applies it to every member), so two compounds in the same cluster share that model's idiosyncratic
bias and are NOT independent. Splitting by compound would put `ScSbRu2` in DESIGN and `NbInRu2` in
TEST while both were predicted by the identical model -- leakage through the shared fold. The 64
compounds contain only ~27 clusters, so the effective sample is much smaller than it looks, and the
honest split respects that.

WHAT IS SWEPT (the same space that was explored informally):
  forms      capped power law  min(c*(T/300)^p, 1)   -- the published choice
             uncapped power    c*(T/300)^p
             flat              c                      -- pure bias removal, the null model
  objective  median over rows  |  median of per-compound medians
  threshold  minimum chemistry support in {0, 2, 3, 4, 6}
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from run_loco_chemistry import cluster_of

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
SUPPORT = "data/exports/kappa_v2/cluster_support_real.csv"
OUT = "data/exports/kappa_v2/nested_validation.json"
CS = np.linspace(0.10, 1.20, 56)
PS = np.linspace(-1.5, 1.5, 31)
FORMS = ("capped_power", "uncapped_power", "flat")
OBJECTIVES = ("rows", "per_compound")
THRESHOLDS = (0, 2, 3, 4, 6)


def factor(form: str, c: float, p: float, T: np.ndarray) -> np.ndarray:
    if form == "flat":
        return np.full(T.shape, c)
    f = c * (T / 300.0) ** p
    return np.minimum(f, 1.0) if form == "capped_power" else f


def _grid(form: str, T: np.ndarray) -> np.ndarray:
    """(nC, nP, nRows) correction factors. `flat` ignores p, so only one p row is meaningful."""
    if form == "flat":
        return np.repeat((CS[:, None, None] * np.ones_like(T)[None, None, :]), 1, axis=1)
    f = CS[:, None, None] * (T[None, None, :] / 300.0) ** PS[None, :, None]
    return np.minimum(f, 1.0) if form == "capped_power" else f


def _pad_index(codes: np.ndarray) -> np.ndarray:
    """(n_compounds, max_rows) row indices, -1 where a compound has fewer rows.

    Needed to take a per-compound median without a Python loop over the (c, p) grid. The naive
    version cost ~62 million np.median calls per run, which made this script unusable.
    """
    n = codes.max() + 1
    counts = np.bincount(codes, minlength=n)
    idx = np.full((n, counts.max()), -1, dtype=int)
    for k in range(n):
        pos = np.flatnonzero(codes == k)
        idx[k, :len(pos)] = pos
    return idx


def fit(form: str, objective: str, d: pd.DataFrame) -> tuple[float, float]:
    """Best (c, p) for this form under this objective, on the rows given."""
    kb, kr, T = d.k_pred.values, d.k_ref.values, d["T"].values.astype(float)
    G = _grid(form, T)
    err = np.abs(np.log10(kb[None, None, :] * G) - np.log10(kr[None, None, :]))
    if objective == "rows":
        score = np.median(err, axis=2)
    else:
        idx = _pad_index(pd.Categorical(d.compound).codes)
        gathered = err[:, :, idx.clip(min=0)]                 # (nC, nP, ncpd, maxrows)
        gathered = np.where(idx[None, None, :, :] >= 0, gathered, np.nan)
        score = np.nanmedian(np.nanmedian(gathered, axis=3), axis=2)
    i, j = np.unravel_index(np.argmin(score), score.shape)
    return float(CS[i]), (0.0 if form == "flat" else float(PS[j]))


def per_compound(d: pd.DataFrame, form: str, c: float, p: float) -> pd.DataFrame:
    k = d.k_pred.values * factor(form, c, p, d["T"].values.astype(float))
    t = d.assign(_k=k, _ape=100 * np.abs(k - d.k_ref.values) / d.k_ref.values,
                 _ratio=k / d.k_ref.values)
    return (t.groupby(["compound", "klass", "sib", "cluster"], dropna=False)
            .agg(ape=("_ape", "median"), ratio=("_ratio", "median")).reset_index())


def score_of(g: pd.DataFrame) -> dict:
    return dict(n=int(len(g)), median_ape=float(g.ape.median()),
                within_30=float((g.ape < 30).mean()),
                within_2x=float(g.ratio.between(0.5, 2.0).mean()),
                bias=float(g.ratio.median()))


def main(repeats: int = 200, seed: int = 0, klass: str = "all",
         blind: str | None = None, tag: str = "") -> int:
    d = pd.read_csv(blind or BLIND)
    d["cluster"] = d.compound.map(cluster_of)
    # SCOPE TO ONE STRUCTURE CLASS. Without this the design half selects the form, objective and
    # threshold on a POOLED half+full population, and the reported test median is a blend of two
    # very different problems -- one split measured pooled 48.4% against half-only 33.7% and
    # full-only 66.6%. Worse, full Heuslers are usually the minority (one split had 2 of 26 design
    # compounds), so the frozen choice was effectively made on half-Heusler performance and then
    # applied to and reported for both. The project's own rule is that the two classes are never
    # pooled; this makes that rule enforceable here.
    if klass != "all":
        n0 = d.compound.nunique()
        d = d[d.klass == klass].copy()
        print(f"scope: {klass} Heuslers only -- {d.compound.nunique()} of {n0} compounds")
    sup = pd.read_csv(SUPPORT, index_col=0).n_compounds
    d["sib"] = d.cluster.map(sup).fillna(0).astype(int)

    clusters = sorted(d.cluster.unique())
    print(f"blind conditions {len(d)}   compounds {d.compound.nunique()}   "
          f"chemistry clusters {len(clusters)}")
    print(f"  the effective independent sample is {len(clusters)} clusters, not "
          f"{d.compound.nunique()} compounds -- splits are by cluster\n")

    rng = np.random.default_rng(seed)
    rows = []
    picks: list = []
    for rep in range(repeats):
        rng.shuffle(clusters)
        half = len(clusters) // 2
        des_c, tst_c = set(clusters[:half]), set(clusters[half:])
        des, tst = d[d.cluster.isin(des_c)], d[d.cluster.isin(tst_c)]
        if des.compound.nunique() < 8 or tst.compound.nunique() < 8:
            continue

        # ---- explore on the DESIGN half only -----------------------------------
        best = None
        for form in FORMS:
            for obj in OBJECTIVES:
                c, p = fit(form, obj, des)
                gd = per_compound(des, form, c, p)
                for th in THRESHOLDS:
                    s = gd[gd.sib >= th]
                    if len(s) < 6:
                        continue
                    m = float(s.ape.median())
                    if best is None or m < best[0]:
                        best = (m, form, obj, th, c, p)
        if best is None:
            continue
        dm, form, obj, th, c, p = best

        # ---- freeze, then apply once to the untouched TEST half -----------------
        gt = per_compound(tst, form, c, p)
        st = gt[gt.sib >= th]
        if len(st) < 6:
            continue
        rec = score_of(st)
        rec.update(design_median=dm, form=form, objective=obj, threshold=th, c=c, p=p,
                   n_design=int(des.compound.nunique()))
        # the same test half with NO correction, as the null baseline
        gn = per_compound(tst, "flat", 1.0, 0.0)
        gn = gn[gn.sib >= th]
        rec["null_median"] = float(gn.ape.median())
        rec["null_within_2x"] = float(gn.ratio.between(0.5, 2.0).mean())
        # per-class breakdown, so a pooled run cannot hide which class drove the number
        for kl in ("half", "full"):
            sk = st[st.klass == kl]
            rec[f"median_{kl}"] = float(sk.ape.median()) if len(sk) else float("nan")
            rec[f"n_{kl}"] = int(len(sk))
        # per-kappa-band TEST medians. A tail improvement must be shown on HELD-OUT data: the
        # magnitude-calibration experiment produced a low-kappa gain that existed only in-sample,
        # and without recording bands here that mistake is unavoidable rather than merely likely.
        kref = tst.groupby("compound").k_ref.median()
        st2 = st.assign(_k=st.compound.map(kref))
        for lo, hi, nm in ((0, 2, "lo"), (2, 5, "mid"), (5, 1e9, "hi")):
            sb = st2[(st2._k >= lo) & (st2._k < hi)]
            rec[f"band_{nm}"] = float(sb.ape.median()) if len(sb) else float("nan")
            rec[f"nband_{nm}"] = int(len(sb))
        rows.append(rec)
        picks.append((form, obj, th))
        if (rep + 1) % 50 == 0:
            print(f"  {rep + 1}/{repeats} repeats", flush=True)

    R = pd.DataFrame(rows)
    if not len(R):
        print("no usable splits")
        return 1

    print("\n" + "=" * 92)
    print(f"NESTED VALIDATION -- {len(R)} cluster-level 50/50 splits")
    print("=" * 92)
    print(f"  {'quantity':<34}{'median':>10}{'5th pct':>10}{'95th pct':>10}")
    for key, lab in (("design_median", "DESIGN half median err (in-sample)"),
                     ("median_ape", "TEST half median err (HONEST)"),
                     ("null_median", "TEST half, NO correction"),
                     ("within_2x", "TEST half within 2x"),
                     ("null_within_2x", "TEST half within 2x, no correction"),
                     ("within_30", "TEST half within 30%"),
                     ("bias", "TEST half bias")):
        v = R[key]
        scale = 100 if key in ("within_2x", "within_30", "null_within_2x") else 1
        u = "%" if scale == 100 or "err" in lab or "median" in key else ""
        print(f"  {lab:<34}{scale * v.median():>9.1f}{u}"
              f"{scale * v.quantile(.05):>9.1f}{u}{scale * v.quantile(.95):>9.1f}{u}")

    opt = float((R.median_ape - R.design_median).median())
    print(f"\n  WINNER'S CURSE: the design half scores {R.design_median.median():.1f}% and the "
          f"untouched test half {R.median_ape.median():.1f}%")
    print(f"  -> selecting form, objective and threshold on the evaluation set flatters the median "
          f"by {opt:+.1f} pp")
    gain = float((R.null_median - R.median_ape).median())
    print(f"  The correction still earns its place: {R.null_median.median():.1f}% uncorrected -> "
          f"{R.median_ape.median():.1f}% corrected on data it never saw ({gain:+.1f} pp).")

    print("\n  WHICH CHOICES THE DESIGN HALF ACTUALLY PICKED (stability of the selection):")
    print("")
    print("  TEST-half median error by measured-kappa band (HELD OUT, not in-sample):")
    print(f"    {'band':<16}{'median':>9}{'5th':>8}{'95th':>8}{'typical n':>11}")
    for _nm, _lab in (("lo", "< 2 W/m/K"), ("mid", "2-5"), ("hi", "> 5")):
        _c = R[f"band_{_nm}"].dropna()
        if len(_c):
            print(f"    {_lab:<16}{_c.median():>8.1f}%{_c.quantile(.05):>7.1f}%"
                  f"{_c.quantile(.95):>7.1f}%{R[f'nband_{_nm}'].median():>11.0f}")
    print("")
    P = pd.DataFrame(picks, columns=["form", "objective", "threshold"])
    for col in P.columns:
        vc = P[col].value_counts(normalize=True)
        print(f"    {col:<11}" + "   ".join(f"{k}: {100 * v:.0f}%" for k, v in vc.items()))
    print(f"    fitted c: median {R.c.median():.3f} [{R.c.quantile(.05):.2f}"
          f"-{R.c.quantile(.95):.2f}]   p: median {R.p.median():+.3f} "
          f"[{R.p.quantile(.05):+.2f}-{R.p.quantile(.95):+.2f}]")

    out_json = OUT.replace(".json", f"{tag}.json") if tag else OUT
    Path(out_json).write_text(json.dumps(dict(
        n_splits=int(len(R)), split_unit="chemistry cluster",
        n_clusters=len(set(d.cluster)), n_compounds=int(d.compound.nunique()),
        honest=dict(median_ape=float(R.median_ape.median()),
                    p05=float(R.median_ape.quantile(.05)),
                    p95=float(R.median_ape.quantile(.95)),
                    within_2x=float(R.within_2x.median()),
                    within_30=float(R.within_30.median()),
                    bias=float(R.bias.median())),
        in_sample_design=float(R.design_median.median()),
        winners_curse_pp=opt,
        uncorrected_null=dict(median_ape=float(R.null_median.median()),
                              within_2x=float(R.null_within_2x.median())),
        selection_stability={c: P[c].value_counts(normalize=True).to_dict() for c in P.columns},
    ), indent=2, default=str))
    R.to_csv(f"data/exports/kappa_v2/nested_validation_splits{tag}.csv", index=False)
    print(f"\nwrote {OUT}")
    print("wrote data/exports/kappa_v2/nested_validation_splits.csv")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", default=None, help="alternative blind-test CSV to score")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--klass", default="all", choices=["all", "half", "full"],
                    help="restrict to one structure class; 'all' pools them and is NOT a "
                         "publishable headline")
    raise SystemExit(main(**vars(ap.parse_args())))
