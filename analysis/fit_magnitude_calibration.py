"""Fix the magnitude bias: does a kappa-dependent term beat the temperature-only calibration?

THE DIAGNOSED PROBLEM. On the 46 measured half Heuslers the error is not random and is not caused by
missing data. It tracks the size of kappa itself -- Spearman(measured kappa, predicted/measured) =
-0.661, p = 6e-7:

    measured < 2 W/m/K   (n=6)  : 83.8% error, we OVER-predict  by 1.84x
    measured 2-5         (n=22) : 32.6% error, bias 0.97x -- well calibrated
    measured 5-10        (n=16) : 39.8% error, we UNDER-predict (0.63x)
    measured > 10        (n=2)  : 76.0% error, 0.24x

That is range compression -- the classic consequence of fitting a regularised model to a noisy
target. The current correction is `min(c*(T/300)^p, 1)`, a function of temperature ALONE, so it
cannot express a magnitude dependence and structurally cannot fix this.

WHY IT MATTERS BEYOND THE ERROR BAR. Thermoelectric candidates are the LOW-kappa ones, and low
kappa is exactly where we over-predict by 1.84x. Our failure mode is therefore ranking a genuinely
excellent ultralow-kappa material as merely average -- false negatives, the expensive kind for a
screening tool.

WHAT IS TESTED. Three nested forms, written so each contains the previous one:

    A  factor = min(c * (T/300)^p, 1)                      the current form
    B  factor = min(c * kappa^(b-1) * (T/300)^p, 1)        adds one magnitude exponent
    C  factor = min(c * kappa^(b-1), 1)                    magnitude only, no temperature

kappa_expt = kappa_bulk * factor, so B with b = 1 reduces EXACTLY to A. The cap stays because grain
boundaries and defects can only lower kappa, never raise it.

HOW IT IS JUDGED, and why that matters more than the fit itself. A third free parameter always
improves an in-sample fit, and this project has already measured its own winner's curse at +10.7
percentage points. So the decision is made on a NESTED split -- clusters divided in half, the form
chosen on one half, applied untouched to the other -- and on whether the magnitude bias actually
disappears, not merely whether the median moves.

KILL CRITERION, fixed in advance: form B is adopted only if it beats A on the nested test half AND
the residual magnitude correlation moves substantially toward zero. A lower in-sample median is not
sufficient evidence and will not be treated as such.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from run_loco_chemistry import cluster_of

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
OUT = "data/exports/kappa_v2/magnitude_calibration.json"
CS = np.linspace(0.10, 1.60, 76)      # prefactor
BS = np.linspace(0.40, 1.80, 71)      # magnitude exponent; 1.0 == current form
PS = np.linspace(-1.0, 1.5, 51)       # temperature exponent


def factor(form, c, b, p, k, T):
    """Suppression factor, capped at 1. `k` is the model's bulk prediction, `T` in kelvin."""
    f = np.full(np.shape(k), float(c), dtype=float)
    if form in ("B", "C"):
        f = f * np.power(k, b - 1.0)
    if form in ("A", "B"):
        f = f * np.power(np.asarray(T, dtype=float) / 300.0, p)
    return np.minimum(f, 1.0)


def grid_for(form):
    return (CS,
            BS if form in ("B", "C") else np.array([1.0]),
            PS if form in ("A", "B") else np.array([0.0]))


def _pad_index(codes):
    """(n_compounds, max_rows) row indices, -1 where a compound has fewer rows.

    Lets the per-compound median be taken without a Python loop over the parameter grid. The naive
    version needed ~11 million np.median calls per fit and never finished.
    """
    n = codes.max() + 1
    counts = np.bincount(codes, minlength=n)
    idx = np.full((n, counts.max()), -1, dtype=int)
    for i in range(n):
        pos = np.flatnonzero(codes == i)
        idx[i, :len(pos)] = pos
    return idx


def fit(form, d):
    """Best (c, b, p) under the per-compound objective: median over compounds of each
    compound's median absolute log error. Matches how every result is reported.

    Loops over c only and vectorises b, p and the per-compound median together, which keeps the
    working array near a megabyte instead of the half-gigabyte a fully vectorised version needs.
    """
    k, kr, T = d.k_pred.values, d.k_ref.values, d["T"].values.astype(float)
    idx = _pad_index(pd.Categorical(d.compound).codes)
    valid = idx >= 0
    cs, bs, ps = grid_for(form)
    kb = np.power(k[None, :], bs[:, None] - 1.0)          # (nB, rows)
    tp = np.power(T[None, :] / 300.0, ps[:, None])        # (nP, rows)
    logkr = np.log10(kr)
    best, arg = None, (np.nan, np.nan, np.nan)
    for c in cs:
        f = np.full((len(bs), len(ps), len(k)), float(c))
        if form in ("B", "C"):
            f = f * kb[:, None, :]
        if form in ("A", "B"):
            f = f * tp[None, :, :]
        np.minimum(f, 1.0, out=f)
        err = np.abs(np.log10(k[None, None, :] * f) - logkr[None, None, :])
        g = np.where(valid[None, None, :, :], err[:, :, idx.clip(min=0)], np.nan)
        score = np.nanmedian(np.nanmedian(g, axis=3), axis=2)      # (nB, nP)
        i, j = np.unravel_index(np.nanargmin(score), score.shape)
        if best is None or score[i, j] < best:
            best, arg = float(score[i, j]), (float(c), float(bs[i]), float(ps[j]))
    return arg


def per_compound(d, form, c, b, p):
    kk = d.k_pred.values * factor(form, c, b, p, d.k_pred.values, d["T"].values)
    t = d.assign(_k=kk, _a=100 * np.abs(kk - d.k_ref.values) / d.k_ref.values,
                 _r=kk / d.k_ref.values)
    return (t.groupby(["compound", "cluster"], dropna=False)
            .agg(k_ref=("k_ref", "median"), kk=("_k", "median"),
                 ape=("_a", "median"), ratio=("_r", "median")).reset_index())


def score(g):
    return dict(n=int(len(g)), median_ape=float(g.ape.median()),
                within_30=float((g.ape < 30).mean()),
                within_2x=float(g.ratio.between(0.5, 2.0).mean()),
                bias=float(g.ratio.median()),
                mag_rho=float(spearmanr(g.k_ref, g.ratio).statistic))


def band_table(g, label):
    print(f"    {label}")
    print(f"      {'measured kappa':<16}{'n':>4}{'median err':>12}{'bias':>8}")
    for lo, hi, lab in ((0, 2, "< 2 W/m/K"), (2, 5, "2-5"), (5, 10, "5-10"), (10, 1e9, "> 10")):
        s = g[(g.k_ref >= lo) & (g.k_ref < hi)]
        if len(s):
            print(f"      {lab:<16}{len(s):>4}{s.ape.median():>11.1f}%{s.ratio.median():>7.2f}x")


def main() -> int:
    d = pd.read_csv(BLIND)
    d["cluster"] = d.compound.map(cluster_of)
    d = d[d.klass == "half"].copy()          # HALF HEUSLERS ONLY -- the agreed scope
    print(f"half-Heusler blind conditions {len(d)}   compounds {d.compound.nunique()}"
          f"   chemistry clusters {d.cluster.nunique()}")

    # ---- 1. in-sample fits, for the record only -------------------------------
    print("\n" + "=" * 88)
    print("STEP 1 -- fitted forms (leave-one-CLUSTER-out, in-sample form choice)")
    print("=" * 88)
    res = {}
    for form in ("A", "B", "C"):
        fac = np.full(len(d), np.nan)
        pars = {}
        for cl in sorted(d.cluster.unique()):
            te = (d.cluster == cl).values
            tr = ~te
            assert not (d.cluster.values[tr] == cl).any(), f"{cl} leaked into its own calibration"
            c, b, p = fit(form, d[tr])
            if c != c:
                continue
            fac[te] = factor(form, c, b, p, d.k_pred.values[te], d["T"].values[te])
            pars[cl] = (c, b, p)
        dd = d.assign(_f=fac).dropna(subset=["_f"])
        g = per_compound(dd.assign(k_pred=dd.k_pred), form,
                         *np.median([pars[c] for c in pars], axis=0))
        # score with the per-fold factors, which is the honest in-sample number
        kk = dd.k_pred.values * dd._f.values
        t = dd.assign(_k=kk, _a=100 * np.abs(kk - dd.k_ref.values) / dd.k_ref.values,
                      _r=kk / dd.k_ref.values)
        gp = (t.groupby(["compound", "cluster"], dropna=False)
              .agg(k_ref=("k_ref", "median"), kk=("_k", "median"),
                   ape=("_a", "median"), ratio=("_r", "median")).reset_index())
        s = score(gp)
        med = np.median([pars[c] for c in pars], axis=0)
        res[form] = dict(score=s, c=float(med[0]), b=float(med[1]), p=float(med[2]),
                         per_compound=gp)
        nm = {"A": "A temperature only (current)", "B": "B + magnitude term",
              "C": "C magnitude only"}[form]
        print(f"\n  {nm}")
        print(f"    c={med[0]:.3f}  b={med[1]:.3f}  p={med[2]:+.3f}")
        print(f"    median err {s['median_ape']:.1f}%   within2x {100*s['within_2x']:.0f}%"
              f"   bias {s['bias']:.2f}x   magnitude rho {s['mag_rho']:+.3f}")
        band_table(gp, "error by kappa band:")

    # ---- 2. does B actually beat A, per compound? ------------------------------
    print("\n" + "=" * 88)
    print("STEP 2 -- paired comparison, B vs A, per compound")
    print("=" * 88)
    m = res["A"]["per_compound"].merge(res["B"]["per_compound"], on="compound",
                                       suffixes=("_A", "_B"))
    diff = m.ape_A - m.ape_B
    try:
        pv = float(wilcoxon(m.ape_A, m.ape_B).pvalue)
    except Exception:
        pv = float("nan")
    print(f"  compounds compared {len(m)}   B better on {int((diff>0).sum())}"
          f"   A better on {int((diff<0).sum())}")
    print(f"  median improvement {diff.median():+.1f} pp   Wilcoxon p = {pv:.4f}")

    # ---- 3. NESTED: choose the form on one half, apply to the other ------------
    print("\n" + "=" * 88)
    print("STEP 3 -- NESTED test: the form is chosen on the design half only")
    print("=" * 88)
    clusters = sorted(d.cluster.unique())
    rng = np.random.default_rng(0)
    rows = []
    for rep in range(60):
        cl = list(clusters)
        rng.shuffle(cl)
        h = len(cl) // 2
        des, tst = d[d.cluster.isin(cl[:h])], d[d.cluster.isin(cl[h:])]
        if des.compound.nunique() < 6 or tst.compound.nunique() < 6:
            continue
        picks = {}
        for form in ("A", "B", "C"):
            c, b, p = fit(form, des)
            gd = per_compound(des, form, c, b, p)
            picks[form] = (float(gd.ape.median()), c, b, p)
        best_form = min(picks, key=lambda f: picks[f][0])
        rec = dict(rep=rep, chosen=best_form)
        for form in ("A", "B", "C"):
            _, c, b, p = picks[form]
            gt = per_compound(tst, form, c, b, p)
            rec[f"test_{form}"] = float(gt.ape.median())
            rec[f"w2x_{form}"] = float(gt.ratio.between(0.5, 2.0).mean())
            rec[f"rho_{form}"] = float(spearmanr(gt.k_ref, gt.ratio).statistic)
        rows.append(rec)
        if (rep + 1) % 20 == 0:
            print(f"  {rep+1} splits done", flush=True)
    R = pd.DataFrame(rows)
    print(f"\n  usable splits: {len(R)}")
    print(f"  {'form':<28}{'test median':>13}{'test within2x':>15}{'magnitude rho':>15}")
    for form, nm in (("A", "A temperature only (current)"), ("B", "B + magnitude term"),
                     ("C", "C magnitude only")):
        print(f"  {nm:<28}{R[f'test_{form}'].median():>12.1f}%"
              f"{100*R[f'w2x_{form}'].median():>14.0f}%{R[f'rho_{form}'].median():>+15.3f}")
    print(f"\n  form the design half picked: {dict(R.chosen.value_counts())}")
    dAB = R.test_A - R.test_B
    try:
        pv2 = float(wilcoxon(R.test_A, R.test_B).pvalue)
    except Exception:
        pv2 = float("nan")
    print(f"  B beats A on the untouched half in {int((dAB>0).sum())} of {len(R)} splits;"
          f" median gain {dAB.median():+.1f} pp, p = {pv2:.4f}")

    # ---- verdict --------------------------------------------------------------
    rho_A, rho_B = R.rho_A.median(), R.rho_B.median()
    better = dAB.median() > 0 and pv2 < 0.05
    fixed = abs(rho_B) < abs(rho_A) * 0.6
    print("\n" + "=" * 88)
    if better and fixed:
        v = "ADOPT form B -- it wins on held-out data AND the magnitude bias collapses"
    elif better:
        v = (f"B wins on error ({dAB.median():+.1f} pp) but the magnitude bias persists "
             f"(rho {rho_A:+.2f} -> {rho_B:+.2f}) -- adopt, but keep reporting the bias")
    elif fixed:
        v = ("B removes the bias but does not improve the median -- KEEP A, report the bias as "
             "a known limitation")
    else:
        v = "KEEP form A -- the magnitude term earns nothing on held-out data"
    print(f"VERDICT: {v}")
    print("=" * 88)

    Path(OUT).write_text(json.dumps(dict(
        scope="half Heuslers only", n_compounds=int(d.compound.nunique()),
        n_clusters=int(d.cluster.nunique()),
        in_sample={f: {k: v for k, v in res[f].items() if k != "per_compound"}
                   for f in res},
        paired_B_vs_A=dict(n=int(len(m)), median_gain_pp=float(diff.median()), p=pv),
        nested={f: dict(test_median=float(R[f"test_{f}"].median()),
                        test_within_2x=float(R[f"w2x_{f}"].median()),
                        magnitude_rho=float(R[f"rho_{f}"].median())) for f in ("A", "B", "C")},
        nested_splits=int(len(R)), chosen_counts=dict(R.chosen.value_counts()),
        nested_B_vs_A_p=pv2, verdict=v), indent=2, default=str))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
