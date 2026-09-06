"""Figure 4 -- the blind test: how well, where it fails, and against what alternative.

Three panels that between them answer every question a referee asks after Figure 3, merging what
were originally three separate figures. Keeping them together matters: the domain restriction, the
family structure and the null comparison are one argument, and split across three figures a reader
has to hold them in mind simultaneously anyway.

  (a) PARITY, chemistry held out. In-domain compounds in the paper's accent, out-of-domain ones as
      open grey markers. The domain restriction is therefore ARGUED here rather than asserted --
      a reader sees the grey points sitting far off the diagonal and understands the screen without
      being told. Nothing is hidden by restricting the headline; the excluded compounds are on the
      same axes as the included ones.
  (b) ERROR BY BONDING FAMILY. Sorted by value, n on every bar. Bi-Pd fails completely and is
      excluded from predictions outright; Sb-Pd and Sb-Pt are where the paper issues. Showing the
      family that fails is what makes the families that work believable.
  (c) AGAINST THE NULLS. Both nulls are fitted with the chemistry held out exactly as the
      calibration is, because a null given less information than the model is not a fair test and
      a referee will say so.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # release layout: analysis/ is a sibling of paper/
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

import figlib as F
from make_paper_predictions import structure_status


def ve(e):
    g = Element(str(e)).group
    return g if g <= 12 else g - 10


def vec(f):
    try:
        return int(sum(ve(e) for e in Composition(f).elements))
    except Exception:  # noqa: BLE001
        return None


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def nulls(d, cluster):
    """Leave-one-chemistry-out constant and power-law nulls, per compound."""
    out = {}
    for name in ("constant", "power"):
        rec = []
        for cl in sorted(set(cluster.values())):
            te = d[d.compound.map(cluster) == cl]
            tr = d[d.compound.map(cluster) != cl]
            if not len(te) or len(tr) < 4:
                continue
            if name == "constant":
                pred = np.full(len(te), float(np.median(tr.k_ref)))
            else:
                lx = np.log(tr["T"].values / 300.0)
                ly = np.log(tr.k_ref.values)
                q, la = np.polyfit(lx, ly, 1)
                pred = np.exp(la) * (te["T"].values / 300.0) ** q
            rec.append(pd.DataFrame(dict(compound=te.compound.values,
                                         ape=np.abs(pred - te.k_ref.values) / te.k_ref.values * 100,
                                         ratio=pred / te.k_ref.values)))
        r = pd.concat(rec).groupby("compound").median()
        out[name] = r
    return out


def main() -> int:
    import extend_blind_test as E
    import matplotlib.pyplot as plt
    from run_loco_chemistry import cluster_of

    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["VEC"] = d.compound.map(vec)
    st = {c_: structure_status(str(c_)) for c_ in d.compound.unique()}
    d["ind"] = (d.VEC == 18) & d.compound.map(lambda c_: st[c_][0] and not st[c_][1])
    d["fam"] = d.compound.map(yzfam)

    dom = d[d.ind]
    c, p = E.fit_cp(dom)

    # IN-DOMAIN COMPOUNDS ARE SCORED WITH THE CALIBRATION HELD OUT, matching the nulls in panel (c)
    # and Table 1. Scoring them with a single fit over the whole domain gave 29.7% where the table
    # says 31.0%, and put an in-sample method against out-of-sample nulls in a panel whose title
    # says the nulls are held out. Out-of-domain compounds keep the whole-domain constants: there
    # is no fold to hold them out of, and they are shown as the constants a user would apply.
    dom_n = F.nested_calibrate(dom, E.fit_cp, E.apply_cp, cluster_of)
    outd = d[~d.ind].copy()
    outd["k_cal"] = outd.k_pred * E.apply_cp(outd["T"].values, c, p)
    d = pd.concat([dom_n, outd], ignore_index=True)
    d["ape2"] = (d.k_cal - d.k_ref).abs() / d.k_ref * 100
    d["ratio2"] = d.k_cal / d.k_ref

    per = d.groupby("compound").agg(ape=("ape2", "median"), ratio=("ratio2", "median"),
                                    ref=("k_ref", "median"), cal=("k_cal", "median"),
                                    ind=("ind", "first"), fam=("fam", "first"))
    ins, out = per[per.ind], per[~per.ind]
    print(f"  in domain {len(ins)}   median {ins.ape.median():.1f}%   "
          f"within2x {ins.ratio.between(.5, 2).mean()*100:.1f}%")
    print(f"  out       {len(out)}   median {out.ape.median():.1f}%   "
          f"within2x {out.ratio.between(.5, 2).mean()*100:.1f}%")

    clus = {cp_: (cluster_of(cp_) or "?") for cp_ in dom.compound.unique()}
    nl = nulls(dom, clus)
    print(f"  null constant  {nl['constant'].ape.median():.1f}%   "
          f"within2x {nl['constant'].ratio.between(.5,2).mean()*100:.1f}%")
    print(f"  null power law {nl['power'].ape.median():.1f}%   "
          f"within2x {nl['power'].ratio.between(.5,2).mean()*100:.1f}%")

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, F.DOUBLE * 0.42))
    # (a) carries the headline result and is square by construction, so it needs the most width
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.05, 0.75], wspace=0.34)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) parity
    lo = float(min(per.ref.min(), per.cal.min())) * 0.6
    hi = float(max(per.ref.max(), per.cal.max())) * 1.6
    F.band_2x(axa, lo, hi, color=F.BAND, label=None)
    axa.scatter(out.ref, out.cal, s=20, facecolors="none", edgecolors=F.REFUSED,
                lw=0.9, marker="s", zorder=3, label=f"outside domain ({len(out)})")
    axa.scatter(ins.ref, ins.cal, s=20, c=F.OURS, marker="o", lw=0.4,
                edgecolors="white", zorder=4, label=f"in domain ({len(ins)})")
    F.parity_axes(axa, lo, hi, "measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)",
                  "predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    # same conductivity ticks as Figures 2, 3 and 5; this panel still carried decade labels,
    # so one quantity was read two ways in one paper.
    F.log_ticks(axa, [v for v in (0.3, 0.5, 1, 2, 3, 5, 10, 20) if lo <= v <= hi])
    axa.legend(loc="upper left", fontsize=6.5, borderaxespad=0.3)
    axa.text(0.97, 0.05, f"{ins.ratio.between(.5,2).mean()*100:.0f}% within 2x",
             transform=axa.transAxes, ha="right", fontsize=7, color=F.OURS, fontweight="bold")
    F.panel_label(axa, "a", dx=-0.24)

    # (b) family
    fam = (ins.groupby("fam").agg(err=("ape", "median"), n=("ape", "size"))
              .query("n >= 2").sort_values("err"))
    ob = out.groupby("fam").agg(err=("ape", "median"), n=("ape", "size")).query("n >= 2")
    # Families the paper actually issues predictions in are drawn in the accent; the rest recede.
    # The panel then makes the argument rather than merely reporting: predictions are issued where
    # the family has been validated, and the two worst families are exactly the two the prediction
    # code refuses. A reader can check that claim against the bar lengths without being told.
    # DERIVED, never hardcoded. The hardcoded set included Sb-Pd, but both Sb-Pd candidates
    # (DySbPd, TmSbPd) are FLAGGED and appear in Figure 5's refusal block -- so the panel starred a
    # family the paper issues nothing in, contradicting Figure 5, Table 2 and Section 5.
    _pred = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    ISSUES_IN = set(_pred[_pred.status == "ISSUED"].family.dropna())
    print(f"  families with issued predictions (derived): {sorted(ISSUES_IN)}")
    y = np.arange(len(fam))
    cols = [F.OURS if f in ISSUES_IN else F.REFUSED for f in fam.index]
    axb.barh(y, fam.err, color=cols, height=0.68, zorder=3)
    axb.set_yticks(y)
    axb.set_yticklabels([f"{f} *" if f in ISSUES_IN else f for f in fam.index], fontsize=6.5)
    for i, (e_, n_) in enumerate(zip(fam.err, fam.n)):
        # "n = 3", not a bare "3": on an axis of median error a lone number beside a bar reads
        # as a second error value.
        axb.text(e_ + max(fam.err) * 0.02, i, f"n = {n_}", va="center", fontsize=6,
                 color="#666666")
    axb.set_xlabel("median error (%)")
    # 1.14 left no room for the count beside the longest bar, so Co-Sn's "n = 3" was clipped
    # at the panel edge.
    axb.set_xlim(0, max(fam.err) * 1.30)
    axb.invert_yaxis()
    axb.set_title("by bonding family  ($*$ = predictions issued)", fontsize=7.5, pad=3)
    F.panel_label(axb, "b", dx=-0.32)

    # (c) nulls
    # short labels -- the long forms collide at this panel width; the full functional form of each
    # null belongs in the caption, not squeezed under a 25 mm axis
    # Panel (c) must agree with Table 1, whose model row is the median over five seeds. Panels (a)
    # and (b) are per-compound and are necessarily one seed; the caption says so. Printing the
    # seed-0 value here while the table printed the seed median put 31% beside 33.7% in the same
    # document -- the class of mismatch this project has now hit three times.
    import json as _json
    try:
        _sa = _json.load(open("data/exports/kappa_v2/seed_averaged_indomain.json"))
        _model_median = float(_sa["median_ape_median"])
        print(f"  panel (c) model bar uses the seed-averaged {_model_median}% "
              f"(this seed alone: {ins.ape.median():.1f}%)")
    except Exception:  # noqa: BLE001
        _model_median = float(ins.ape.median())
    # the title already says these are nulls; repeating the word on the tick made the first
    # lines of adjacent labels touch and print as "constantpower-law".
    labels = ["this\nwork", "constant\n$\\kappa$", "power\nlaw"]
    vals = [_model_median, nl["constant"].ape.median(), nl["power"].ape.median()]
    w2 = [ins.ratio.between(.5, 2).mean() * 100,
          nl["constant"].ratio.between(.5, 2).mean() * 100,
          nl["power"].ratio.between(.5, 2).mean() * 100]
    cols = [F.OURS, F.REFUSED, F.REFUSED]
    x = np.arange(3)
    axc.bar(x, vals, color=cols, width=0.62, zorder=3)
    for i, (v, w) in enumerate(zip(vals, w2)):
        axc.text(i, v + max(vals) * 0.03, f"{v:.0f}%", ha="center", fontsize=7.5,
                 fontweight="bold" if i == 0 else "normal",
                 color=F.OURS if i == 0 else "#777777")
        # inside the bar, on two lines, so it cannot run past the bar edge
        axc.text(i, max(vals) * 0.06, f"{w:.0f}%\nwithin\n2x", ha="center", va="bottom",
                 fontsize=5.8, linespacing=1.15,
                 color="white" if i == 0 else "#4A4A4A")
    axc.set_xticks(x)
    axc.set_xticklabels(labels, fontsize=6.8)
    axc.set_xlim(-0.72, 2.72)
    axc.set_ylabel("median error (%)")
    axc.set_ylim(0, max(vals) * 1.30)
    axc.set_title("nulls fitted with chemistry held out", fontsize=7.5, pad=3)
    F.panel_label(axc, "c", dx=-0.32)

    F.save(fig, "fig04_blind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
