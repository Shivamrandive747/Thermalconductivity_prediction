"""Figure 8 -- the method end to end.

WHY THE PAPER NEEDS ONE. Sections 2 and 3 describe a corpus, a tiering rule, a regression model, a
transfer function, a holdout protocol and four screens, and a reader has to assemble them from
prose. Most papers of this kind carry a schematic; ours did not.

WHAT IT IS BUILT TO SAY. One thing above all: measurements and calculations are never pooled. They
run in separate lanes from the corpus to the point where the transfer function maps one onto the
other, and that separation is the reason the paper can claim to predict experiment rather than to
reproduce other people's DFT. Drawing them as two lanes makes the claim structural instead of
something the reader has to take on trust from a sentence in Section 2.

DESIGN RULES, all inherited rather than invented here:
  * colour follows the paper's semantic system -- measurements near-black, calculations grey and
    visibly provisional, the transfer function in the accent blue that means "our contribution"
    and is spent on nothing else, outcomes graded from that blue down to the refusal grey;
  * every box carries its count, so the schematic doubles as the funnel from 289 calculations and
    52 measurements down to five issued predictions;
  * every box that corresponds to an equation cites it, so the figure indexes Section 3 instead of
    paraphrasing it;
  * the dashed enclosure marks exactly what the chemistry holdout removes -- the model AND the
    calibration -- because holding out one but not the other is the single easiest way to get this
    kind of result wrong, and this paper has made that mistake once already.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # release layout: analysis/ is a sibling of paper/
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import figlib as F

CALC = "#8A8F94"      # calculations: provisional
MEAS = F.MEASURED     # measurements: the ground truth, darkest ink
OURS = F.OURS         # the transfer function
INK = "#2B2B2B"


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    F.use_style()
    fig, ax = plt.subplots(figsize=(F.DOUBLE, F.DOUBLE * 0.40))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x0, x1, yc, h, title, body, edge, *, lw=0.9, fc="white", tcol=None, fs=6.9):
        ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.018",
                                    fc=fc, ec=edge, lw=lw, zorder=3))
        ax.text((x0 + x1) / 2, yc + h * 0.22, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tcol or edge, zorder=4)
        ax.text((x0 + x1) / 2, yc - h * 0.20, body, ha="center", va="center",
                fontsize=5.9, color="#4A4A4A", zorder=4, linespacing=1.35)

    def arrow(x0, y0, x1, y1, col="#9A9A9A", rad=0.0, lw=0.8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=lw, color=col, zorder=2))

    YU, YL, YC, H = 0.795, 0.205, 0.50, 0.235

    # ---- the holdout enclosure, drawn first so everything sits on top --------------------
    ax.add_patch(FancyBboxPatch((0.352, 0.045), 0.363, 0.905,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#F7F9FB", ec="#B9C6D0", lw=0.8, ls=(0, (4, 2.5)), zorder=1))
    ax.text(0.534, 0.982, "one chemistry cluster withheld from BOTH",
            ha="center", va="top", fontsize=6.2, color="#5A7F94", fontstyle="italic", zorder=5)

    # ---- upper lane: calculations -------------------------------------------------------
    ax.text(0.005, 0.955, "CALCULATIONS", fontsize=6.6, fontweight="bold", color=CALC,
            ha="left", va="center")
    box(0.012, 0.163, YU, H, "Published $\\kappa_L$", "289 compounds\n32 sources", CALC)
    box(0.196, 0.340, YU, H, "Tier by method", "tier 1 = full BTE\nEq. (3)", CALC)
    box(0.372, 0.516, YU, H, "Training pool", "4765 rows\n543 compounds", CALC)

    # ---- lower lane: measurements -------------------------------------------------------
    ax.text(0.005, 0.045, "MEASUREMENTS", fontsize=6.6, fontweight="bold", color=MEAS,
            ha="left", va="center")
    box(0.012, 0.163, YL, H, "Measured $\\kappa_\\mathrm{tot}$", "52 compounds\n164 papers", MEAS)
    box(0.196, 0.340, YL, H, "Wiedemann\u2013Franz", "$\\kappa_L=\\kappa_\\mathrm{tot}-LT/\\rho$\n"
        "Eq. (1)", MEAS)
    box(0.372, 0.516, YL, H, "Domain screens", "VEC 18, cubic,\nnon-polymorphic\n38 compounds", MEAS)

    for y in (YU, YL):
        arrow(0.163, y, 0.196, y)
        arrow(0.340, y, 0.372, y)

    # ---- the meeting point --------------------------------------------------------------
    box(0.532, 0.700, YC, 0.34, "TRANSFER FUNCTION",
        "$\\kappa^\\mathrm{expt}=\\kappa^\\mathrm{BTE}\\times$\n"
        "$\\min[\\,c\\,(T/300)^{p},\\;1\\,]$\n"
        "$c=0.510$,  $p=0.800$\nEq. (2), (5), (6)",
        OURS, lw=1.5, fc="#F2F8FC")
    arrow(0.516, YU - 0.05, 0.532, YC + 0.12, col=CALC, rad=-0.16, lw=1.0)
    arrow(0.516, YL + 0.05, 0.532, YC - 0.12, col=MEAS, rad=0.16, lw=1.0)

    # ---- screens and outcomes -------------------------------------------------------------
    box(0.742, 0.846, YC, 0.30, "SCREENS",
        "validated family\nstructure on record\nsource agreement", "#8A9BA8")
    arrow(0.700, YC, 0.742, YC, col=OURS, lw=1.0)

    tiers = [("ISSUED", "5", OURS, 0.875), ("FLAGGED", "5", "#8A9BA8", 0.625),
             ("CONDITIONAL", "10", "#5A7F94", 0.375), ("REFUSED", "6", "#A8ACB0", 0.125)]
    for name, n, col, y in tiers:
        ax.add_patch(FancyBboxPatch((0.876, y - 0.082), 0.120, 0.164,
                                    boxstyle="round,pad=0.003,rounding_size=0.014",
                                    fc="white", ec=col, lw=1.4 if name == "ISSUED" else 0.9,
                                    zorder=3))
        ax.text(0.936, y + 0.030, n, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=col, zorder=4)
        ax.text(0.936, y - 0.044, name, ha="center", va="center", fontsize=5.6, color=col,
                zorder=4)
        arrow(0.846, YC, 0.876, y, col="#BFC5CA", rad=0.0 if abs(y - YC) < 0.02 else
              (-0.12 if y > YC else 0.12), lw=0.7)

    F.save(fig, "fig08_method")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
