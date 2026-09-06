"""Figure 2 -- the DFT-to-experiment offset, before and after the transfer function.

This is the figure the paper's central sentence rests on: published Boltzmann-transport values sit
systematically above laboratory measurements, and a two-parameter transfer removes most of that
offset. Two panels, same axes, same points -- the only difference is the correction, so a reader
can see the cloud move rather than take the median on trust.

Log-log and square, because the quantity being judged is a RATIO. On linear axes a factor-of-two
error at 15 W/m/K would dominate the eye while the same factor at 2 W/m/K vanished, which is the
opposite of how the result is scored.
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
BTE_METHODS = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def pairs() -> pd.DataFrame:
    """Temperature-matched (measured, calculated) pairs for half Heuslers.

    Matched within 50 K rather than pooled per compound: kappa falls steeply with temperature, so
    comparing a 300 K measurement against a 900 K calculation would manufacture an offset that is
    just the temperature dependence.
    """
    from run_loco_chemistry import sites
    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    tr = tr[(tr.k > 0) & tr["T"].between(200, 1300)]
    tr = tr[tr.red.map(lambda r: (sites(r) or ["?"])[0] == "half")]
    m = tr.method.astype(str).str.lower()
    meas = tr[tr.tier == 0]
    dft = tr[(tr.tier == 1) & m.str.contains(BTE_METHODS, regex=True)
             & ~m.str.contains(NOT_BTE, regex=True)]
    rows = []
    for r in sorted(set(meas.red) & set(dft.red)):
        mm, dd = meas[meas.red == r], dft[dft.red == r]
        for T, k in zip(mm["T"].values, mm.k.values):
            j = int(np.abs(dd["T"].values - T).argmin())
            if abs(dd["T"].values[j] - T) <= 50:
                rows.append(dict(compound=r, fam=yzfam(r), T=float(T),
                                 k_exp=float(k), k_dft=float(dd.k.values[j])))
    return pd.DataFrame(rows)


def main() -> int:
    import extend_blind_test as E
    from pymatgen.core import Element
    from make_paper_predictions import structure_status

    def _vec(f):
        try:
            return int(sum((Element(str(e)).group if Element(str(e)).group <= 12
                            else Element(str(e)).group - 10)
                           for e in Composition(f).elements))
        except Exception:  # noqa: BLE001
            return None

    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    Bh = B[B.klass == "half"].copy()
    Bh["VEC"] = Bh.compound.map(_vec)
    _st = {c_: structure_status(str(c_)) for c_ in Bh.compound.unique()}
    Bh["ind"] = (Bh.VEC == 18) & Bh.compound.map(lambda c_: _st[c_][0] and not _st[c_][1])
    c, p = E.fit_cp(Bh[Bh.ind])

    raw = pairs()
    # SAME DOMAIN AS EVERY OTHER FIGURE.
    #
    # The matched-pair set and the in-domain blind set both happen to contain 38 compounds, but
    # they are not the same 38 -- they overlap in 34. Reporting one figure's "38" beside another's
    # in the Results section would read as a single consistent set and would be wrong. Restricting
    # here to the declared domain makes the scope identical throughout the paper.
    keep = {c_ for c_ in raw.compound.unique()
            if _vec(c_) == 18 and structure_status(str(c_))[0]
            and not structure_status(str(c_))[1]}
    dropped = sorted(set(raw.compound.unique()) - keep)
    if dropped:
        print(f"  excluded as out of domain: {dropped}")
    raw = raw[raw.compound.isin(keep)]
    if not len(raw):
        print("no matched pairs found")
        return 1

    # AGGREGATE TO ONE POINT PER COMPOUND PER 100 K BIN.
    #
    # Two reasons, and both would otherwise mislead a reader. First, plotting all 2,426 matched
    # rows implies far more independent evidence than exists -- they come from 38 compounds, and
    # ZrNiSn alone contributes 646 of them, so its curve would dominate the visual impression of
    # every statistic on the panel. Second, the calibration is FITTED per compound, so scoring the
    # figure per row measures something the fit never optimised: pooled by row the corrected median
    # ratio reads 1.38, per compound it is close to 1. Mismatching the fit objective and the
    # reporting unit has already cost this project 9 pp once.
    raw["Tbin"] = (raw["T"] / 100).round() * 100
    d = (raw.groupby(["compound", "fam", "Tbin"], as_index=False)
            .agg(k_exp=("k_exp", "median"), k_dft=("k_dft", "median"), n_rows=("k_exp", "size")))
    d = d.rename(columns={"Tbin": "T"})
    d["k_cal"] = d.k_dft * np.minimum(c * (d["T"] / 300.0) ** p, 1.0)
    print(f"  matched rows {len(raw)} -> {len(d)} compound-temperature points "
          f"across {d.compound.nunique()} compounds")
    print(f"  raw median ratio  {np.median(d.k_dft / d.k_exp):.2f}")
    print(f"  calibrated        {np.median(d.k_cal / d.k_exp):.2f}   (c={c:.3f}, p={p:.3f})")

    # ENCODE TEMPERATURE, NOT FAMILY.
    #
    # An earlier version gave each of eight bonding families its own hue and marker, with a
    # two-column legend in the corner. That spent the whole qualitative palette on a distinction
    # this figure does not make -- the claim here is the SIZE of the offset, and family identity
    # is carried by Figures 4 and 6 where it decides something. Worse, the legend was the most
    # visually dominant object on a panel whose subject is a cloud of points.
    #
    # Temperature is the right covariate to show. The transfer function carries an exponent
    # p = 0.80, so the offset is temperature-dependent by construction, and until now no figure
    # in the paper let a reader see that.
    F.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, F.DOUBLE * 0.52))
    allv = np.concatenate([d.k_exp.values, d.k_dft.values, d.k_cal.values])
    lo, hi = float(np.min(allv)) * 0.7, float(np.max(allv)) * 1.4
    cmap = F.temp_cmap()
    norm = Normalize(vmin=float(d["T"].min()), vmax=float(d["T"].max()))

    for ax, col, ttl in ((axes[0], "k_dft", "Uncorrected calculation"),
                         (axes[1], "k_cal", "After transfer function")):
        F.band_2x(ax, lo, hi, label=None)
        sc = ax.scatter(d.k_exp, d[col], c=d["T"], cmap=cmap, norm=norm, s=17,
                        marker="o", lw=0.35, edgecolors="white", zorder=3)
        F.parity_axes(ax, lo, hi, "measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)",
                      "calculated $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
        F.log_ticks(ax, [v for v in (2, 3, 5, 10, 20, 30) if lo <= v <= hi])
        ratio = np.median(d[col] / d.k_exp)
        within = float(((d[col] / d.k_exp).between(0.5, 2.0)).mean() * 100)
        ax.set_title(ttl)
        ax.text(0.045, 0.955,
                f"median ratio {ratio:.2f}\nwithin 2$\\times$  {within:.0f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.4, pad=2.4, alpha=0.94))

    axes[1].set_ylabel("")
    F.panel_label(axes[0], "a", dx=-0.19)
    F.panel_label(axes[1], "b", dx=-0.10)
    axes[0].text(0.97, 0.035, f"{d.compound.nunique()} compounds, {len(d)} points",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=7, color="#444444")
    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.012, aspect=30, shrink=0.72)
    cb.set_label("temperature (K)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    F.save(fig, "fig02_offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
