"""Figure 1 -- why this paper exists: calculations are plentiful, measurements are not.

The paper's entire premise is one ratio. Two hundred and eighty-nine half Heuslers carry a
published full Boltzmann-transport lattice thermal conductivity; fifty-two have been measured in a
laboratory. Any model trained on the abundant quantity learns to reproduce CALCULATIONS, and
inherits whatever systematic offset the calculations carry. This figure has to make that scarcity
felt in about two seconds, because every later figure depends on the reader accepting it.

Note 289, not the 505 quoted elsewhere for "tier 1": 43% of nominally tier-1 half-Heusler formulae
carry a semi-empirical Slack or Debye-Callaway label rather than a transport calculation, and the
transfer function this paper fits applies to the latter. The looser count would inflate the ratio
the figure is built on.

  (a) The funnel, drawn PROPORTIONALLY. A schematic with equal-sized boxes would show the
      structure and hide the point; the collapse from 289 to 52 is the point.
  (b) How many independent publications stand behind each measured compound. A first draft plotted
      the measured and calculated clouds together and titled it "and sit systematically above
      them" -- but 2,400 measured points buried the calculated ones and the offset was invisible,
      so the panel asserted something the reader could not see. Figure 2 makes that argument
      properly on temperature-matched pairs. This panel instead carries information nothing else
      in the paper does: most of the 52 compounds rest on a SINGLE publication, which is the honest
      limit on the evidence and sets up the tiering used for the worked examples.
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
from pymatgen.core import Composition

import figlib as F

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from run_loco_chemistry import sites

    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    tr = tr[tr.k > 0]
    tr["klass"] = tr.red.map(lambda r: (sites(r) or ["?"])[0])
    half = tr[tr.klass == "half"]
    m = half.method.astype(str).str.lower()
    meas = half[half.tier == 0]
    calc = half[(half.tier == 1) & m.str.contains(BTE, regex=True)
                & ~m.str.contains(NOT_BTE, regex=True)]

    stages = [("Heusler compounds\nwith any $\\kappa$ record", tr.red.nunique()),
              ("half Heuslers", half.red.nunique()),
              ("with a first-principles\ncalculation", calc.red.nunique()),
              ("MEASURED in a\nlaboratory", meas.red.nunique())]
    print("  " + "   ".join(f"{a.splitlines()[0]}={b}" for a, b in stages))
    print(f"  papers behind the measured set: {meas.source_doi.nunique()}")

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, F.DOUBLE * 0.38))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.26)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) proportional funnel
    top = stages[0][1]
    cols = ["#C9CDD2", "#9AA0A6", F.RAW_DFT, F.MEASURED]
    for i, ((lab, n), col) in enumerate(zip(stages, cols)):
        w = n / top
        axa.add_patch(Rectangle(((1 - w) / 2, -i - 0.36), w, 0.72,
                                facecolor=col, edgecolor="none", zorder=3))
        # A proportional funnel makes the last bar 5% of the first, which is the point -- but the
        # count then does not fit inside it. Below a width threshold the number moves outside and
        # takes the bar's colour, so the smallest stage stays the most legible rather than the least.
        if w > 0.22:
            axa.text(0.5, -i, f"{n:,}", ha="center", va="center", fontsize=9,
                     fontweight="bold", color="white" if i >= 2 else "#333333", zorder=4)
        else:
            axa.text((1 - w) / 2 - 0.035, -i, f"{n:,}", ha="right", va="center", fontsize=9,
                     fontweight="bold", color=col, zorder=4)
        axa.text(1.04, -i, lab, ha="left", va="center", fontsize=6.8, color="#333333")
    axa.set_xlim(-0.30, 1.9)
    axa.set_ylim(-3.75, 0.62)
    axa.axis("off")
    # computed, never typed -- an earlier draft of this line read "10 to 1" from memory while the
    # data said 5.6, which is the exact defect that got three headline numbers withdrawn
    ratio = calc.red.nunique() / meas.red.nunique()
    axa.set_title(f"Calculations outnumber measurements {ratio:.0f} to 1",
                  fontsize=8, loc="left", pad=4)
    F.panel_label(axa, "a", dx=-0.09, dy=1.16)

    # (b) how much independent evidence stands behind each measured compound
    src = meas.groupby("red").source_doi.nunique()
    bins = [1, 2, 3, 5, 10, 10_000]
    names = ["1", "2", "3-4", "5-9", "10+"]
    counts = [int(((src >= lo) & (src < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
    # a single source is the weak case and is coloured as such; the rest strengthen left to right
    bcols = [F.VERMILLION] + [F.RAW_DFT] * 2 + [F.OURS] * 2
    x = np.arange(len(names))
    axb.bar(x, counts, color=bcols, width=0.66, zorder=3)
    for i, v in enumerate(counts):
        axb.text(i, v + max(counts) * 0.03, str(v), ha="center", fontsize=7.5,
                 fontweight="bold" if i == 0 else "normal",
                 color=F.VERMILLION if i == 0 else "#555555")
    axb.set_xticks(x)
    axb.set_xticklabels(names)
    axb.set_xlabel("independent publications reporting the compound")
    axb.set_ylabel("number of measured compounds")
    axb.set_ylim(0, max(counts) * 1.22)
    single = counts[0] / sum(counts) * 100
    axb.set_title(f"and {single:.0f}% of measurements rest on a single paper",
                  fontsize=8, loc="left", pad=4)
    print(f"  sources per measured compound: " +
          ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    F.panel_label(axb, "b", dx=-0.20, dy=1.16)

    F.save(fig, "fig01_corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
