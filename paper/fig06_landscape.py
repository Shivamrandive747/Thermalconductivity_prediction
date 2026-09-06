"""Figure 6 -- everything the method can say, and what each statement is worth.

One panel, FOUR tiers, ordered by how much evidence stands behind them:

  ISSUED       five compounds the method predicts. Bonding family validated against two or more
               measurements, structure on record, sources in agreement. These are the five the
               abstract commits to.
  FLAGGED      five more that clear the family test but fail one specific check -- a non-cubic
               polymorph on record, or a value outside the range the family has ever been
               measured in. Section 5 is explicit that these are "flagged rather than issued":
               the number is reported with its reason, and is NOT a prediction the paper makes.
  CONDITIONAL  ten compounds that pass every structural and chemical screen but sit in a family
               with no usable measured anchor. Nine are Ni-Bi, which needs TWO sound
               measurements: its only measured member, YNiBi, carries a bipolar contribution its
               own authors report, so the family stands at zero usable anchors rather than one.
The six REFUSED compounds are deliberately NOT drawn. Four contain thorium or uranium and two
have VEC != 18, so for them the calibrated number is meaningless -- plotting it on a kappa axis,
with a conformal interval, would assert that a quantity we decline to give is nonetheless
estimated to lie in a range. They are named in the caption with the screen each fails. Flagged and
refused together remain the "eleven declined" of the abstract; only the five with meaningful
numbers are drawn.

Drawn as one figure because the reader's question is "what do you actually know, and how well?"
-- which is a question about the whole ladder, not about any one rung.
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

import figlib as F

PRED = "data/Target_Materials/PAPER_PREDICTIONS.csv"
COND = "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv"
COND_COL = "#8FB8CE"     # conditional: blue-leaning, predictable once a measurement exists
FLAG_COL = "#8A9BA8"     # flagged: grey-leaning, declined -- must NOT read as issued blue


def short_reason(note: str) -> str:
    """The one word that says why a compound is not in the clean tier."""
    n = str(note)
    if n.startswith("POLYMORPHIC"):
        return "polymorphic"
    if n.startswith("MARGINAL"):
        return "marginal"
    if n.startswith("EXTRAP"):
        return "extrapolation"
    if "VEC" in n:
        return "VEC $\\neq$ 18"
    if "radioactive" in n or "actinide" in n:
        return "radioactive"
    if "disagree" in n or "factor of" in n:
        return "sources disagree"
    return "flagged"


def main() -> int:
    import matplotlib.pyplot as plt

    P = pd.read_csv(PRED)
    C = pd.read_csv(COND)
    iss = P[P.status == "ISSUED"].sort_values("kappa_pred_300")
    flg = P[P.status == "FLAGGED"].sort_values("kappa_pred_300")
    con = C.sort_values(["family", "kappa_pred_300"])
    ref = P[P.status == "REFUSED"].sort_values("kappa_pred_300")
    print(f"  issued {len(iss)}   flagged {len(flg)}   conditional {len(con)}   "
          f"refused {len(ref)}")
    assert len(iss) + len(flg) + len(ref) == len(P), "a status was dropped from the ladder"

    F.use_style()
    n = len(iss) + len(flg) + len(con) + 3
    fig, ax = plt.subplots(figsize=(F.ONEHALF, 0.215 * n + 1.6))

    y, ticks, labs, seps = 0, [], [], []

    def draw(r, col, filled, bold=False, note=""):
        ax.plot([r.lo90, r.hi90], [y, y], lw=1.0, color=col, alpha=.55, zorder=3)
        ax.plot([r.lo50, r.hi50], [y, y], lw=2.9, color=col, alpha=.92, zorder=4)
        ax.plot([r.kappa_pred_300], [y], marker="o", ms=4.4, mew=1.3, ls="none",
                mfc=(col if filled else "white"), mec=col, zorder=5)
        ax.text(r.hi90 * 1.05, y, f"{r.kappa_pred_300:.1f}", va="center", fontsize=6.6,
                color=col, fontweight="bold" if bold else "normal")
        if note:
            ax.text(r.hi90 * 1.05 * 1.42, y, note, va="center", fontsize=5.9,
                    color="#8A8A8A", fontstyle="italic")

    # ---- tier 1: issued outright -------------------------------------------------
    for r in iss.itertuples():
        draw(r, F.OURS, True, bold=True)
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1
    seps.append(y + 0.5); y -= 0.55

    # ---- tier 2: flagged -- declined with a named reason, NOT issued -------------
    for r in flg.itertuples():
        draw(r, FLAG_COL, False, note=short_reason(r.note))
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1
    seps.append(y + 0.5); y -= 0.55

    # ---- tier 3: conditional on a family measurement -----------------------------
    for r in con.itertuples():
        cav = str(getattr(r, "caveat", "") or "")
        note = "Sm valence" if cav.startswith("FLAGGED") else ""
        draw(r, COND_COL, False, note=note)
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1

    # tier 4 -- REFUSED -- is intentionally not drawn. See the module docstring: for a compound
    # refused on physics the calibrated value is not a quantity, and giving it an interval would
    # claim more than we are willing to claim. The caption names them instead.

    for s in seps:
        ax.axhline(s, color="#CFCFCF", lw=.6, ls=":")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=6.6)
    for t, v in zip(ax.get_yticklabels(), ticks):
        if v < seps[1]:
            t.set_color("#5A7F94")           # conditional
        elif v < seps[0]:
            t.set_color("#6E7F8A")           # flagged -- declined, not the issued blue
    ax.set_xscale("log")
    # limits from the data: CrSnPt sits at 0.15 and was being clipped off the left edge
    # limits from what is DRAWN. The refused compounds are no longer plotted, and leaving
    # them in this list stretched the axis to cover a value the figure does not show.
    drawn = pd.concat([iss, flg])
    allv = (list(drawn.lo90.dropna()) + list(drawn.hi90.dropna())
            + list(C.lo90) + list(C.hi90))
    ax.set_xlim(min(allv) * 0.72, max(allv) * 3.0)
    F.log_ticks(ax, [1, 2, 3, 5, 10, 20], both=False)
    ax.set_ylim(y + 0.6, 0.9)
    ax.set_xlabel("predicted $\\kappa_L$ at 300 K  (W m$^{-1}$ K$^{-1}$)")

    def frac(v):
        return (v - (y + 0.6)) / (0.9 - (y + 0.6))

    for anchor, text, col in (
            (0.55, f"ISSUED  ({len(iss)})", F.OURS),
            (seps[0] - 0.45, f"FLAGGED  ({len(flg)})   not issued \u2013 reason given",
             FLAG_COL),
            (seps[1] - 0.45,
             f"CONDITIONAL  ({len(con)})   family has no measured member", "#5A7F94")):
        ax.text(.012, frac(anchor), text, transform=ax.transAxes, fontsize=7,
                fontweight="bold", color=col, va="center")

    hl = [plt.Line2D([], [], color=F.OURS, lw=2.9),
          plt.Line2D([], [], color=F.OURS, lw=1.0, alpha=.55),
          plt.Line2D([], [], color=F.OURS, marker="o", mfc=F.OURS, ls="none", ms=4.4),
          plt.Line2D([], [], color=FLAG_COL, marker="o", mfc="white", ls="none", ms=4.4)]
    # placed above the plot rather than over the refused rows it was covering
    # plain "%" -- this style renders text with mathtext, not LaTeX, so "\%" prints the backslash
    ax.legend(hl, ["50% interval", "90% interval", "issued", "not issued"],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              fontsize=6.2, frameon=False, borderaxespad=.2, handletextpad=.5,
              columnspacing=1.2)
    F.save(fig, "fig06_landscape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
