"""Figure 3 -- does the calibrated prediction land on the measurement?

Every other figure reports a summary statistic. A materials reader does not believe a median; they
believe a curve sitting on the data points they recognise. This shows measured kappa(T) with the
uncorrected calculation above it and the calibrated prediction through it, for nine compounds, each
with its whole chemistry held out of training.

SCOPE. All nine sit inside the paper's declared domain of applicability: 18 valence electrons, and
a cubic C1b structure on record that is not polymorphic. Neither screen is invented here and
neither is post hoc -- make_paper_predictions.py has applied both to every issued prediction from
the start. They had simply never been applied to the validation set, and doing so moves the blind
test from 35.8% / 78.4% to 29.7% / 86.8% across 38 compounds.

Out-of-domain compounds are NOT hidden; they carry the argument for the domain restriction and so
belong beside the family analysis in Figure 4, where a reader meets them while asking "why restrict
it?" rather than while asking "does it work?".

THE PANELS ARE STRATIFIED, NOT SELECTED. Best, middle and hardest thirds, drawn at the tercile
boundaries so the choice is arithmetic. Nine successes would be cherry-picking, a referee would
assume it had happened, and the honest version is more convincing precisely because row 3 is
visibly imperfect. The script asserts that the nine shown have the same median as all eligible
compounds, so the claim in the caption is checked rather than asserted.
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

import matplotlib.ticker as mticker

import figlib as F
from make_paper_predictions import structure_status


def yzfam(f):
    """The (Y,Z) bonding family -- the unit that carries validation in this work."""
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else ""
    except Exception:  # noqa: BLE001
        return ""

MIN_POINTS = 5
MIN_SPAN = 300.0


def ve(e):
    g = Element(str(e)).group
    return g if g <= 12 else g - 10


def vec(f):
    try:
        return int(sum(ve(e) for e in Composition(f).elements))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import extend_blind_test as E
    import matplotlib.pyplot as plt

    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["VEC"] = d.compound.map(vec)
    st = {c_: structure_status(str(c_)) for c_ in d.compound.unique()}
    d["in_domain"] = (d.VEC == 18) & d.compound.map(lambda c_: st[c_][0] and not st[c_][1])

    dom = d[d.in_domain]
    c, p = E.fit_cp(dom)
    # NESTED, not a single fit over all 38. Figure 4 already does this and carries a comment
    # naming the defect: a single fit scores each compound with a calibration that saw it,
    # which holds the method to a weaker standard than the nulls it is compared against. This
    # panel had kept the in-sample version, so its per-compound percentages were computed
    # under a different protocol from Table 1 and from Figure 6.
    from run_loco_chemistry import cluster_of
    dom_n = F.nested_calibrate(dom, E.fit_cp, E.apply_cp, cluster_of)
    out = d[~d.in_domain].copy()
    out["k_cal"] = out.k_pred * E.apply_cp(out["T"].values, c, p)
    d = pd.concat([dom_n, out], ignore_index=True)
    d["ape2"] = (d.k_cal - d.k_ref).abs() / d.k_ref * 100
    err = d.groupby("compound").ape2.median()

    g = d.groupby("compound").agg(n=("T", "size"), lo=("T", "min"), hi=("T", "max"),
                                  ind=("in_domain", "first"))
    g["span"] = g.hi - g.lo
    g["ape"] = err
    ok = g[(g.n >= MIN_POINTS) & (g.span >= MIN_SPAN) & g.ind].sort_values("ape")
    n = len(ok)
    print(f"  in-domain compounds with a usable curve: {n}")
    print(f"  calibration fitted in domain: c={c:.3f}, p={p:.3f}")

    pick = [ok.index[i] for i in (0, 1, 2)] \
        + [ok.index[i] for i in (n // 2 - 1, n // 2, n // 2 + 1)] \
        + [ok.index[i] for i in (n - 3, n - 2, n - 1)]
    shown, allm = ok.loc[pick].ape.median(), ok.ape.median()
    print(f"  nine shown median {shown:.1f}%  vs  all {n} eligible {allm:.1f}%")
    # the caption claims the panels are representative; check it rather than assert it
    assert abs(shown - allm) < 6.0, (
        f"the nine panels are not representative ({shown:.1f}% vs {allm:.1f}%) -- "
        "either widen the selection or drop the claim from the caption")

    rows = ["best third", "middle third", "hardest third"]
    F.use_style()
    fig, axes = plt.subplots(3, 3, figsize=(F.DOUBLE, F.DOUBLE * 0.80), sharey=True)
    for k, cmp_ in enumerate(pick):
        ax = axes[k // 3][k % 3]
        s = d[d.compound == cmp_].sort_values("T")
        ax.plot(s["T"], s.k_pred, **{**F.ROLE["raw"], "label": None})
        ax.plot(s["T"], s.k_cal, **{**F.ROLE["ours"], "label": None})
        ax.plot(s["T"], s.k_ref, **{**F.ROLE["measured"], "label": None})
        ax.set_title(f"{cmp_}   ({yzfam(cmp_)})", fontsize=8, pad=2.5, loc="left")
        ax.text(0.97, 0.93, f"{err[cmp_]:.0f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=F.OURS, fontweight="bold")
        ax.set_yscale("log")
        ax.margins(x=0.06)
        ax.tick_params(labelsize=6.5, pad=1.5)
        ax.tick_params(axis="y", which="minor", length=1.2)
        if k % 3 == 0:
            ax.text(-0.30, 0.5, rows[k // 3], transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=7.5, fontstyle="italic", color="#555555")

    # one shared y range, chosen from the data, with integer labels: a log axis left to its
    # defaults would label only 10^1 across a span running from about 1.5 to 30.
    allk = np.concatenate([d[d.compound.isin(pick)][c_].values
                           for c_ in ("k_ref", "k_pred", "k_cal")])
    ylo, yhi = float(np.nanmin(allk)) * 0.75, float(np.nanmax(allk)) * 1.3
    for ax in axes.ravel():
        ax.set_ylim(ylo, yhi)
        # y only -- the temperature axis is linear and its default locator is already right.
        # An earlier pass set the X locator to these kappa values, which put every x tick off
        # the axis and shipped the grid with no temperature scale.
        keep = [v for v in (1, 2, 3, 5, 10, 20, 30) if ylo <= v <= yhi]
        ax.yaxis.set_major_locator(mticker.FixedLocator(keep))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in keep]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    fig.supxlabel("temperature (K)", fontsize=8)
    fig.supylabel("lattice thermal conductivity (W m$^{-1}$ K$^{-1}$)", fontsize=8)
    handles = [plt.Line2D([], [], **{a: b for a, b in F.ROLE[r].items() if a != "zorder"})
               for r in ("measured", "raw", "ours")]
    # "uncorrected SURROGATE", not "uncorrected DFT". k_pred in target_blind_test.csv is the
    # CatBoost prediction, not the published transport value -- surrogate_vs_published.json
    # records a 15.4% median difference between them, and Section 4.4 scores the two routes
    # separately. Figure 4 plots the published values and may say "calculated"; this one may not.
    _labels = {"measured": F.ROLE["measured"]["label"],
               "raw": "uncorrected surrogate",
               "ours": F.ROLE["ours"]["label"]}
    fig.legend(handles=handles,
               labels=[_labels[r] for r in ("measured", "raw", "ours")],
               loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3,
               frameon=False, fontsize=7.5)
    F.save(fig, "fig03_curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
