"""Compute the null baselines and the temperature-slope statistics that the deliverable quotes.

WHY THIS FILE EXISTS. An audit found that the null-baseline figures (48.4%, 44.4%, the Wilcoxon
p-values, the Spearman rank correlation) and the temperature-slope figures were HARDCODED LITERALS
in three publisher scripts and computed nowhere. That is structurally identical to the "79.5%
inter-laboratory ceiling" this project withdrew, whose sin -- in `make_final_results.py`'s own
words -- was that it "is computed nowhere in this repository". These numbers carry the single most
referee-exposed claim in the paper (that the model's median-error advantage over a compound-blind
baseline is NOT significant), so they must be regenerable. Publishers read this file's JSON output
instead of restating the numbers.

WHAT THE NULLS ARE, AND WHY THEY ARE THE RIGHT ONES. A referee's first move is to ask whether the
machine learning earns its place. So we build predictors containing NO information about which
compound is being predicted, and fit them exactly as the real calibration is fitted -- on the other
chemistry clusters, never on the compound being scored:

    NULL 1  a single constant kappa: the geometric mean of every training-side measurement.
    NULL 2  A*(T/300)^q, two free parameters, so it can follow the average temperature trend but
            still knows nothing about the compound.

The model must beat these to justify itself. Reporting them is not self-sabotage; a referee builds
them in ten minutes, and it is better that the paper does it first.

THE TEMPERATURE-SLOPE TEST. For each compound with >= 3 temperature bins spanning >= 300 K, fit
log kappa against log T for the measurement and for the calibrated prediction. If the model
understood each compound's phonon physics the two slopes would correlate. Compounds whose MEASURED
slope is positive are excluded: lattice conductivity cannot rise with temperature, so those are
broken references, not model failures.
"""
from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

import extend_blind_test as E

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
OUT = "data/exports/kappa_v2/null_baselines.json"
PER_CLASS = {"half": None, "full": None}   # filled by fitting


def main() -> int:
    D = pd.read_csv(BLIND)
    H = D[D.klass == "half"].dropna(subset=["T", "k_ref", "k_pred"]).copy()
    clusters = sorted(H.cluster.unique())
    print(f"half-Heusler blind test: {len(H)} conditions, "
          f"{H.compound.nunique()} compounds, {len(clusters)} chemistry clusters")

    rows = []
    for cl in clusters:
        tr, te = H[H.cluster != cl], H[H.cluster == cl]
        assert cl not in set(tr.cluster), "cluster leaked into its own fit"
        c, p = E.fit_cp(tr)                                    # the model's own calibration
        k_const = float(np.exp(np.log(tr.k_ref).mean()))       # NULL 1
        q, lA = np.polyfit(np.log(tr["T"].values / 300.0), np.log(tr.k_ref.values), 1)
        A = float(np.exp(lA))                                  # NULL 2
        for cp_, g in te.groupby("compound"):
            kk = g.k_pred.values * E.apply_cp(g["T"].values, c, p)
            n2 = A * (g["T"].values / 300.0) ** q
            ape = lambda pr: float(np.median(np.abs(pr - g.k_ref.values)
                                             / g.k_ref.values * 100.0))
            rat = lambda pr: float(np.median(pr / g.k_ref.values))
            rows.append(dict(compound=cp_, cluster=cl, k_ref=float(g.k_ref.median()),
                             model=ape(kk), const=ape(np.full(len(g), k_const)),
                             power=ape(n2), model_ratio=rat(kk),
                             const_ratio=rat(np.full(len(g), k_const)), power_ratio=rat(n2)))
    R = pd.DataFrame(rows)

    res = {"n_compounds": int(len(R)), "n_clusters": len(clusters),
           "protocol": ("leave-one-chemistry-cluster-out; the null's parameters and the model's "
                        "calibration are both fitted on the training clusters only"),
           "predictors": {}}
    print(f"\n  {'predictor':<28}{'median APE':>12}{'within 2x':>11}{'within 30%':>12}")
    for nm, col, rc in (("model + calibration", "model", "model_ratio"),
                        ("null: constant kappa", "const", "const_ratio"),
                        ("null: A*(T/300)^q", "power", "power_ratio")):
        w2 = float(R[rc].between(0.5, 2.0).mean() * 100)
        w30 = float((R[col] < 30).mean() * 100)
        res["predictors"][nm] = dict(median_ape=round(float(R[col].median()), 2),
                                     within_2x_pct=round(w2, 1), within_30_pct=round(w30, 1))
        print(f"  {nm:<28}{R[col].median():>11.1f}%{w2:>10.1f}%{w30:>11.1f}%")

    res["paired_tests"] = {}
    for nm, col in (("vs_power_law_null", "power"), ("vs_constant_null", "const")):
        _, pv = wilcoxon(np.log10(R.model + 1e-9), np.log10(R[col] + 1e-9))
        res["paired_tests"][nm] = dict(
            wilcoxon_p=round(float(pv), 4),
            model_better_pct=round(float((R.model < R[col]).mean() * 100), 1))
        print(f"  paired Wilcoxon {nm:<20} p = {pv:.4f}   "
              f"model better on {(R.model < R[col]).mean()*100:.0f}%")

    mp = H.groupby("compound").k_pred.median()
    mr = H.groupby("compound").k_ref.median()
    rs, ps = spearmanr(mp, mr)
    res["rank_signal"] = dict(spearman=round(float(rs), 3), p=float(f"{ps:.3g}"),
                              n=int(len(mp)),
                              note="does the model rank compounds better than chance?")
    print(f"\n  Spearman(kappa_pred, kappa_ref) = {rs:+.3f}   p = {ps:.2e}   n = {len(mp)}")

    # ---- temperature slopes -----------------------------------------------------
    cph = E.fit_cp(H)
    sl = []
    for cp_, g in H.groupby("compound"):
        Tv = g["T"].values
        if len(set(Tv)) < 3 or (Tv.max() - Tv.min()) < 300:
            continue
        kk = g.k_pred.values * E.apply_cp(Tv, *cph)
        lt = np.log(Tv)
        sl.append((cp_, float(np.polyfit(lt, np.log(g.k_ref.values), 1)[0]),
                   float(np.polyfit(lt, np.log(kk), 1)[0])))
    S = pd.DataFrame(sl, columns=["compound", "measured_slope", "predicted_slope"])
    Sp = S[S.measured_slope < 0]        # a positive measured slope is a broken reference
    rs2, ps2 = spearmanr(Sp.measured_slope, Sp.predicted_slope)
    rp2, pp2 = pearsonr(Sp.measured_slope, Sp.predicted_slope)
    res["temperature_slope"] = dict(
        n_assessable=int(len(S)), n_physical=int(len(Sp)),
        measured_slope_sd=round(float(Sp.measured_slope.std()), 3),
        predicted_slope_sd=round(float(Sp.predicted_slope.std()), 3),
        measured_range=[round(float(Sp.measured_slope.min()), 2),
                        round(float(Sp.measured_slope.max()), 2)],
        predicted_range=[round(float(Sp.predicted_slope.min()), 2),
                         round(float(Sp.predicted_slope.max()), 2)],
        spearman=round(float(rs2), 3), spearman_p=round(float(ps2), 4),
        pearson=round(float(rp2), 3), pearson_p=round(float(pp2), 4),
        conclusion=("the model does NOT predict temperature dependence: all variation comes from "
                    "the single fitted global exponent, not from the compound"))
    print(f"\n  temperature slopes, {len(Sp)} compounds with a physical (negative) reference:")
    print(f"    measured  sd {Sp.measured_slope.std():.3f}  range "
          f"[{Sp.measured_slope.min():+.2f}, {Sp.measured_slope.max():+.2f}]")
    print(f"    predicted sd {Sp.predicted_slope.std():.3f}  range "
          f"[{Sp.predicted_slope.min():+.2f}, {Sp.predicted_slope.max():+.2f}]")
    print(f"    Spearman {rs2:+.3f} (p={ps2:.3f})   Pearson {rp2:+.3f} (p={pp2:.3f})")

    S.to_csv("data/exports/kappa_v2/temperature_slopes.csv", index=False)
    R.to_csv("data/exports/kappa_v2/null_baselines_per_compound.csv", index=False)
    open(OUT, "w").write(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")
    print("wrote data/exports/kappa_v2/null_baselines_per_compound.csv")
    print("wrote data/exports/kappa_v2/temperature_slopes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
