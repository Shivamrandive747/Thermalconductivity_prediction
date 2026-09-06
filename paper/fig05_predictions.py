"""Figure 5 -- the issued predictions, drawn inside the measurements that license them.

WHAT THIS REPLACES AND WHY. The previous Figure 5 was a horizontal ladder of predictions and
refusals at 300 K. Figure 6 is also a horizontal ladder of predictions and refusals, with a
conditional tier added, so the paper carried the same object twice and spent two of its six
figures making one point. Worse, both showed only 300 K -- and the transfer function carries a
temperature exponent p = 0.80, so the paper's own correction varied with temperature in a way no
figure let a reader see.

WHAT IT SHOWS INSTEAD. For each validated bonding family, every measured member is drawn as a
recessive black curve, and the compounds we predict are drawn over them in the accent blue with
their conformal band. The claim of Section 5 is that an issued prediction is an interpolation
within chemistry that experiment has already visited; that claim is either visibly true here or
visibly false, which is the point of drawing it.

IT ALSO ADMITS SOMETHING THE LADDER HID. Only two of the five issued compounds -- ErSbPt and
HoSbPt -- have a temperature-resolved calculation to calibrate. TmSbPt, GdNiSb and TbNiSb have a
single value at 300 K, and are drawn as single points with their interval rather than as curves.
A ladder at 300 K made all five look alike. They are not alike, and the figure now says so.

Log y, because the correction is multiplicative: on a logarithmic axis a constant factor is a
constant vertical shift, so the reader can see one number doing the work in every panel.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))  # release layout: analysis/ is a sibling of paper/
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pymatgen.core import Composition

import figlib as F

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
PRED = "data/Target_Materials/PAPER_PREDICTIONS.csv"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"

# recessive black: these are real measurements, so they keep the measurement ink, but they are
# CONTEXT for the prediction rather than its subject, so they are thin and partly transparent.
CTX = dict(color=F.MEASURED, lw=0.85, alpha=0.5, zorder=2)

# A white outline behind every label. With fourteen direct labels on two panels some overlap
# is unavoidable; a halo keeps each one legible instead of requiring the layout to be perfect.
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def bin50(s: pd.DataFrame) -> pd.DataFrame:
    """One point per 50 K, median within the bin.

    Sources report on different temperature grids and some contribute hundreds of rows for a
    single compound; drawn raw, one densely sampled source would dominate the visual weight of a
    curve that is meant to represent the compound.
    """
    if not len(s):
        return pd.DataFrame(columns=["T", "k"])
    s = s.copy()
    s["Tb"] = (s["T"] / 50).round() * 50
    out = s.groupby("Tb", as_index=False).k.median()
    return out.rename(columns={"Tb": "T"}).sort_values("T").reset_index(drop=True)


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import extend_blind_test as E
    import matplotlib.pyplot as plt
    from make_paper_predictions import structure_status, vec

    # ---- the calibration, fitted on the declared domain exactly as everywhere else ----------
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    Bh = B[B.klass == "half"].copy()
    Bh["VEC"] = Bh.compound.map(vec)
    _st = {c_: structure_status(str(c_)) for c_ in Bh.compound.unique()}
    Bh["ind"] = (Bh.VEC == 18) & Bh.compound.map(lambda c_: _st[c_][0] and not _st[c_][1])
    c, p = E.fit_cp(Bh[Bh.ind])
    cf = json.load(open("data/exports/kappa_v2/conformal_indomain.json"))["levels"]
    b90 = float(cf["0.90"]["factor"])
    print(f"  calibration c={c:.3f}, p={p:.3f}   90% band x/div {b90}")

    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    tr = tr[(tr.k > 0) & tr["T"].between(150, 1050)]
    tr["fam"] = tr.red.map(yzfam)
    m = tr.method.fillna("").str.lower()
    meas = tr[tr.tier == 0]
    bte = tr[(tr.tier == 1) & m.str.contains(BTE, regex=True) & ~m.str.contains(NOT, regex=True)]

    P = pd.read_csv(PRED)
    iss = P[P.status == "ISSUED"].copy()
    fams = ["Sb-Pt", "Ni-Sb"]
    assert set(iss.family) == set(fams), f"issued families changed: {sorted(set(iss.family))}"

    F.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, F.DOUBLE * 0.42), sharey=True)

    for ax, fam in zip(axes, fams):
        # ---- context: the ENVELOPE this family has been measured in -------------------------
        #
        # Not one curve per member. The claim is "the prediction lies inside the range experiment
        # has visited for this family", so the range is the object to draw. Fourteen labelled
        # curves asked the reader to reconstruct that range themselves, and they could not.
        mem = sorted(set(meas[meas.fam == fam].red.dropna()))
        grid = np.arange(200, 1001, 25.0)
        stack = []
        for cm in mem:
            s = bin50(meas[meas.red == cm])
            if len(s) < 3:
                continue
            # interpolate only inside each member's own measured range -- never extrapolate a
            # measurement to a temperature nobody measured it at.
            v = np.interp(grid, s["T"].values, s.k.values, left=np.nan, right=np.nan)
            v[(grid < s["T"].min()) | (grid > s["T"].max())] = np.nan
            stack.append(v)
        M = np.vstack(stack)
        n_at = np.sum(~np.isnan(M), axis=0)
        ok = n_at >= 2                      # a range needs at least two members to exist
        lo = np.nanmin(M, axis=0)
        hi = np.nanmax(M, axis=0)
        med = np.nanmedian(M, axis=0)
        ax.fill_between(grid[ok], lo[ok], hi[ok], color="#E4E4E4", alpha=1.0, lw=0, zorder=1,
                        label="measured range of the family")
        ax.plot(grid[ok], med[ok], color="#8A8A8A", lw=0.9, ls="-", zorder=6)
        taken: list = []          # (x, log10 y) of every label placed on this panel
        n_env = len(stack)
        print(f"    {fam}: envelope from {n_env} measured members, "
              f"{int(ok.sum())} of {len(grid)} grid points have >=2")

        # ---- the predictions ----------------------------------------------------------------
        nsingle = sum(1 for r in iss[iss.family == fam].itertuples()
                      if len(bin50(bte[bte.red == r.compound])) < 4)
        npoint = 0
        plab: list = []
        for r in iss[iss.family == fam].itertuples():
            s = bin50(bte[bte.red == r.compound])
            if len(s) >= 4:
                kc = s.k.values * np.minimum(c * (s["T"].values / 300.0) ** p, 1.0)
                ax.fill_between(s["T"], kc / b90, kc * b90, color=F.OURS, alpha=0.085,
                                lw=0, zorder=3)
                for edge in (kc / b90, kc * b90):
                    ax.plot(s["T"], edge, color=F.OURS, lw=0.5, ls=(0, (3, 2)),
                            alpha=0.55, zorder=4)
                ax.plot(s["T"], kc, color=F.OURS, lw=1.7, zorder=5)
                plab.append((float(s["T"].iloc[-1]) + 16, float(kc[-1]), r.compound, "left"))
            else:
                k3 = float(r.kappa_pred_300)
                # both single-temperature predictions in a family sit at exactly 300 K and would
                # overprint; the horizontal offset is cosmetic and the caption says so.
                x = 300 + 46 * (npoint - (nsingle - 1) / 2.0)
                ax.errorbar([x], [k3], yerr=[[k3 - k3 / b90], [k3 * b90 - k3]],
                            fmt="o", ms=4.8, mfc="white", mec=F.OURS, mew=1.4,
                            ecolor=F.OURS, elinewidth=1.0, capsize=2.2, zorder=6)
                # beside the marker, alternating sides: the error bar is tall, so there is
                # horizontal room and none vertically.
                side = "right" if npoint % 2 == 0 else "left"
                plab.append((x + (-13 if side == "right" else 13), k3, r.compound, side))
                npoint += 1

        # draw the prediction labels last, nudged apart in log space -- ErSbPt and HoSbPt are
        # predicted 3% apart and would otherwise overprint.
        plab.sort(key=lambda z: z[1])
        prev = -1e9
        for lx, ly, txt, side in plab:
            ly10 = np.log10(ly)
            # clear the other prediction labels ...
            ly10 = max(ly10, prev + 0.052) if ly10 - prev <= 0.052 else ly10
            # ... and any context label close in BOTH temperature and conductivity. Labels far
            # apart in x may share a y, so the test cannot be on y alone.
            for _ in range(24):
                clash = [ty for tx, ty in taken
                         if abs(tx - lx) < 150 and abs(ty - ly10) < 0.050]
                if not clash:
                    break
                ly10 = max(clash) + 0.052
            prev = ly10
            taken.append((lx, ly10))
            ax.text(lx, 10 ** ly10, txt, fontsize=6.6, color=F.OURS, fontweight="bold",
                    va="center", ha=side, path_effects=HALO, zorder=9)

        ax.set_yscale("log")
        ax.set_xlim(130, 1230)
        ax.set_xlabel("temperature (K)")
        ax.set_title(f"{fam}      envelope from {n_env} measured members, "
                     f"{len(iss[iss.family == fam])} predicted")
        n_curve = sum(1 for r in iss[iss.family == fam].itertuples()
                      if len(bte[bte.red == r.compound]) >= 4)
        print(f"  {fam}: {len(mem)} measured members, {len(iss[iss.family == fam])} issued "
              f"({n_curve} as curves, {len(iss[iss.family == fam]) - n_curve} at 300 K only)")

    axes[0].set_ylabel("lattice thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    axes[0].set_ylim(0.62, 44)
    keep = [1, 2, 3, 5, 10, 20, 30, 40]
    for ax in axes:
        ax.yaxis.set_major_locator(mticker.FixedLocator(keep))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(v) for v in keep]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    F.panel_label(axes[0], "a", dx=-0.11)
    F.panel_label(axes[1], "b", dx=-0.05)

    from matplotlib.patches import Patch
    handles = [Patch(fc="#E4E4E4", ec="none"),
               plt.Line2D([], [], color="#8A8A8A", lw=0.9),
               plt.Line2D([], [], color=F.OURS, lw=1.7),
               plt.Line2D([], [], color=F.OURS, marker="o", ls="none", ms=4.6,
                          mfc="white", mew=1.4)]
    fig.legend(handles, ["range measured across the family",
                         "median of measured members",
                         "our prediction, with 90% band",
                         "our prediction, 300 K calculation only"],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
               fontsize=7)
    F.save(fig, "fig05_predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
