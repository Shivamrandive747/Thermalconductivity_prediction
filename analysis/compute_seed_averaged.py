"""The in-domain result across all five seeds, because Methods claims seed averaging.

Section 3.4 states that results are averaged over five random seeds. They were not: every headline
figure comes from `target_blind_test.csv`, which is seed 0. A review established that seed 0 is the
best of the five available runs, and this project has been burned once already by a single-seed
figure -- a seed reaching the classifier but not the regressors manufactured a 13.6 pp "improvement"
that did not exist.

So either the sentence goes or the averaging happens. This does the averaging: the paper's exact
in-domain protocol -- domain screen, nested calibration, both nulls leave-one-chemistry-out --
applied to each seed's predictions in turn.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import extend_blind_test as E
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

SP = "paper/evidence"
OUT = "data/exports/kappa_v2/seed_averaged_indomain.json"


def in_domain(d):
    d = d.copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    return d


def nested(d):
    out = []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        c, p = E.fit_cp(tr)
        if np.isfinite(c):
            out.append(te.assign(k=te.k_pred.values * E.apply_cp(te["T"].values, c, p)))
    return pd.concat(out) if out else None


def null(d, kind):
    out = []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        if kind == "constant":
            k = np.full(len(te), float(np.median(tr.k_ref)))
        else:
            q, la = np.polyfit(np.log(tr["T"].values / 300.0), np.log(tr.k_ref.values), 1)
            k = np.exp(la) * (te["T"].values / 300.0) ** q
        out.append(te.assign(k=k))
    return pd.concat(out)


def fit_c_only(d):
    """The one-parameter ablation: fit c with p held at zero.

    Mirrors E.fit_cp exactly -- same grid over c, same per-compound median of absolute log
    errors -- but with no temperature exponent, so the correction is a single constant factor.
    Written here rather than reused because extend_blind_test.fit_cp always fits both.
    """
    kb, kr, T = d.k_pred.values, d.k_ref.values, d["T"].values.astype(float)
    if len(d) < 4:
        return float("nan")
    codes = pd.Categorical(d.compound).codes
    masks = [codes == k for k in range(codes.max() + 1)]
    best, bc = np.inf, float("nan")
    for c in E.CS:
        e = np.abs(np.log10(kb * min(c, 1.0)) - np.log10(kr))
        s = float(np.median([np.median(e[m]) for m in masks]))
        if s < best:
            best, bc = s, float(c)
    return bc


def nested_c_only(d):
    out = []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        c = fit_c_only(tr)
        if np.isfinite(c):
            out.append(te.assign(k=te.k_pred.values * min(c, 1.0)))
    return pd.concat(out) if out else None


def metrics(P):
    """The four numbers Table 1 reports, from a per-compound frame."""
    return dict(median_ape=round(float(P.a.median()), 1),
                within_2x=round(float(P.r.between(.5, 2).mean() * 100), 1),
                within_30=round(float((P.a <= 30).mean() * 100), 1),
                bias=round(float(P.r.median()), 2))


def per_cmp(D):
    return (D.assign(a=(D.k - D.k_ref).abs() / D.k_ref * 100, r=D.k / D.k_ref)
             .groupby("compound").agg(a=("a", "median"), r=("r", "median"),
                                      ref=("k_ref", "median"), k=("k", "median")))


def main() -> int:
    files = {0: "data/exports/kappa_v2/target_blind_test.csv"}
    for f in sorted(glob.glob(f"{SP}/blind_d2_s*.csv")):
        s = int(f.rstrip(".csv").split("_s")[-1])
        files.setdefault(s, f)
    print(f"seed prediction files found: {sorted(files)}")

    rows = []
    for s in sorted(files):
        raw = pd.read_csv(files[s])
        raw = raw[raw.klass == "half"] if "klass" in raw.columns else raw
        d = in_domain(raw)
        M = per_cmp(nested(d))
        NC, NP = per_cmp(null(d, "constant")), per_cmp(null(d, "power"))
        ic = M.index.intersection(NC.index)
        _, pc = wilcoxon(M.a[ic], NC.a[ic])
        _, pp = wilcoxon(M.a[ic], NP.a[ic])
        rho, prho = spearmanr(M.ref, M.k)
        # the same four metrics for the rows Table 1 puts beside the headline, so the caption's
        # "median over five model seeds" is true of every model-dependent number, not just two.
        U = per_cmp(d.assign(k=d.k_pred))            # uncorrected surrogate
        A = per_cmp(nested_c_only(d))                # one-parameter ablation, c only
        cal_m, unc_m, abl_m = metrics(M), metrics(U), metrics(A)
        rows.append(dict(seed=s, n=int(len(M)), **cal_m,
                         unc_median_ape=unc_m["median_ape"], unc_within_2x=unc_m["within_2x"],
                         unc_within_30=unc_m["within_30"], unc_bias=unc_m["bias"],
                         abl_median_ape=abl_m["median_ape"], abl_within_2x=abl_m["within_2x"],
                         abl_within_30=abl_m["within_30"], abl_bias=abl_m["bias"],
                         p_constant=round(float(pc), 4), p_power=round(float(pp), 4),
                         spearman=round(float(rho), 3),
                         better_pct=round(float((M.a[ic] < NC.a[ic]).mean() * 100), 1)))
        print(f"  seed {s}: n={rows[-1]['n']}  median {rows[-1]['median_ape']:.1f}%  "
              f"within2x {rows[-1]['within_2x']:.1f}%  p={pc:.3f}/{pp:.3f}  rho={rho:+.3f}")

    R = pd.DataFrame(rows)
    summ = dict(n_seeds=int(len(R)),
                median_ape_median=round(float(R.median_ape.median()), 1),
                median_ape_range=[float(R.median_ape.min()), float(R.median_ape.max())],
                within_2x_median=round(float(R.within_2x.median()), 1),
                within_2x_range=[float(R.within_2x.min()), float(R.within_2x.max())],
                p_constant_median=round(float(R.p_constant.median()), 3),
                p_power_median=round(float(R.p_power.median()), 3),
                n_seeds_clearing_both=int(((R.p_constant < .05) & (R.p_power < .05)).sum()),
                spearman_median=round(float(R.spearman.median()), 3),
                spearman_range=[float(R.spearman.min()), float(R.spearman.max())],
                seed0_rank_on_error=int((R.median_ape <= R.loc[R.seed == 0, "median_ape"].iloc[0]).sum()),
                per_seed=rows)

    # Table 1 in full, seed-averaged. Every model-dependent cell, not only the two the caption
    # used to be able to support.
    def band(col):
        return dict(median=round(float(R[col].median()), 1),
                    range=[float(R[col].min()), float(R[col].max())])
    summ["table1_seed_medians"] = {
        "calibrated": {k: band(k) for k in ("median_ape", "within_2x", "within_30", "bias")},
        "uncorrected_surrogate": {k.replace("unc_", ""): band(k) for k in
                                  ("unc_median_ape", "unc_within_2x", "unc_within_30", "unc_bias")},
        "one_parameter_c_only": {k.replace("abl_", ""): band(k) for k in
                                 ("abl_median_ape", "abl_within_2x", "abl_within_30", "abl_bias")},
        "_note": ("every cell is the median across the same 5 model seeds; the ranges show how "
                  "far a single-seed figure can sit from it")}
    json.dump(summ, open(OUT, "w"), indent=2)

    print(f"\n  SEED-AVERAGED (median over {summ['n_seeds']} seeds)")
    print(f"    median error   {summ['median_ape_median']}%   "
          f"(range {summ['median_ape_range'][0]}-{summ['median_ape_range'][1]})")
    print(f"    within 2x      {summ['within_2x_median']}%   "
          f"(range {summ['within_2x_range'][0]}-{summ['within_2x_range'][1]})")
    print(f"    p vs nulls     {summ['p_constant_median']} / {summ['p_power_median']}")
    print(f"    seeds clearing both nulls at 0.05: {summ['n_seeds_clearing_both']} of {summ['n_seeds']}")
    print(f"    Spearman       {summ['spearman_median']}   "
          f"(range {summ['spearman_range'][0]}-{summ['spearman_range'][1]})")
    print(f"    seed 0 ranks {summ['seed0_rank_on_error']} of {summ['n_seeds']} on error "
          f"(1 = best)")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
