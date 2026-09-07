"""Shared helpers for the manuscript figures.

Two jobs. First, keep the six figures dimensionally consistent -- a journal cares about column
width in millimetres and matplotlib thinks in inches, and converting by hand every time is how
figures end up subtly different sizes. Second, always write a PNG next to the PDF, because the PDF
is what the journal gets and the PNG is what gets LOOKED at before it is called finished. A figure
that has never been viewed is not a finished figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
STYLE = str(ROOT / "paperstyle.mplstyle")
OUTDIR = ROOT / "figures"

# Elsevier column widths, millimetres -> inches
MM = 1 / 25.4
SINGLE = 90 * MM
ONEHALF = 140 * MM
DOUBLE = 190 * MM

# Okabe-Ito, named so figures read by intent rather than by hex
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#8C8C8C"
LIGHTGREY = "#D9D9D9"

# ---------------------------------------------------------------------------------------------
# SEMANTIC ROLES -- the single most important consistency decision in the paper.
#
# Colour encodes the ROLE a quantity plays, not merely which series it is, and the same role wears
# the same colour in every figure. A reader learns the scheme once in Figure 2 and can then read
# Figures 3-5 without consulting a legend at all:
#
#     measurement        near-black, filled, solid   -- the ground truth is always the darkest ink
#     uncorrected DFT    grey, open, DASHED          -- always visibly provisional
#     our calibration    accent blue, solid          -- the paper's contribution, one accent only
#     issued prediction  accent blue, OPEN marker    -- same colour, hollow: no measurement exists
#     refused / flagged  desaturated grey            -- present, deliberately recessive
#
# Two rules follow and must not be broken. Nothing else in the paper may use the accent blue, or it
# stops meaning "our result". And every role carries a marker or line style as well as a hue, so
# the encoding survives greyscale printing and colour-vision deficiency.
MEASURED = "#1A1A1A"
RAW_DFT = "#9AA0A6"
OURS = BLUE
PREDICTED = BLUE
REFUSED = "#B8BCC0"
BAND = "#E8E8E8"

ROLE = {
    "measured":  dict(color=MEASURED, marker="o", ls="none", ms=4.0, mfc=MEASURED,
                      mec="white", mew=0.5, zorder=5, label="measured"),
    "raw":       dict(color=RAW_DFT, marker="none", ls="--", lw=1.1, zorder=3,
                      label="uncorrected DFT"),
    "ours":      dict(color=OURS, marker="none", ls="-", lw=1.5, zorder=4,
                      label="calibrated (this work)"),
    "predicted": dict(color=PREDICTED, marker="o", ls="none", ms=4.0, mfc="white",
                      mec=PREDICTED, mew=1.2, zorder=5, label="prediction"),
}


# Continuous ramp for temperature. Truncated `inferno`: warm reads as hot, it is perceptually
# uniform and monotonic in lightness (so it survives greyscale), and it stays clear of the accent
# blue, which means "our result" and must not be spent on a covariate. The ends are trimmed
# because pure black is confusable with the measurement ink and pure near-white disappears.
def temp_cmap(lo: float = 0.12, hi: float = 0.90):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    base = plt.get_cmap("inferno")
    return mcolors.LinearSegmentedColormap.from_list(
        "temp", base(np.linspace(lo, hi, 256)))


def use_style() -> None:
    plt.style.use(STYLE)


def nested_calibrate(d, fit_cp, apply_cp, cluster_of):
    """Calibrate each compound with its OWN chemistry cluster held out of the (c, p) fit.

    Every figure that quotes an accuracy statistic must use this, not a single fit over the whole
    set. A review found the figures scoring an in-sample calibration against leave-one-out nulls
    and printing 30% beside a table that reported 31.0% -- the method held to a weaker standard
    than its own baselines, in the one panel whose title advertises that the nulls are held out.

    Returns the frame with a `k_cal` column and the per-fold constants in `_c`, `_p`.
    """
    import pandas as pd
    out = []
    clus = d.compound.map(lambda c: cluster_of(c) or "?")
    for cl in sorted(set(clus)):
        te, tr = d[clus == cl], d[clus != cl]
        if not len(te) or len(tr) < 4:
            continue
        ci, pi = fit_cp(tr)
        if not np.isfinite(ci):
            continue
        out.append(te.assign(k_cal=te.k_pred.values * apply_cp(te["T"].values, ci, pi),
                             _c=ci, _p=pi))
    return pd.concat(out) if out else None


def figure(width: float = SINGLE, height: float | None = None, **kw):
    """A figure at a journal column width. Height defaults to a 4:3-ish block."""
    use_style()
    return plt.subplots(figsize=(width, height if height else width * 0.72), **kw)


def save(fig, name: str, also_png: bool = True) -> Path:
    """Write the PDF the journal wants, plus a PNG so the figure can actually be inspected."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{name}.pdf"
    # dpi applies ONLY to artists marked rasterized=True; everything else stays vector. 600 is the
    # usual journal floor for a raster panel, and without it a rasterized layer would fall back to
    # the ~100 dpi screen default and look soft in print.
    fig.savefig(pdf, dpi=600)
    if also_png:
        fig.savefig(OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf}  (+ .png for inspection)")
    return pdf


def band_2x(ax, lo: float, hi: float, color: str = "#EDEDED", alpha: float = 1.0,
            label: str | None = "within 2x") -> None:
    """Shade the factor-of-two envelope around the 1:1 line on a parity plot.

    The band is the paper's headline criterion, so it belongs under the data rather than in the
    caption -- a reader should be able to count the points outside it. It therefore needs an EDGE:
    an unbounded grey wedge reads as background texture, while a bounded one reads as a threshold
    that a point is inside or outside of. Fill lightened and edges drawn for that reason.
    """
    x = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.fill_between(x, x / 2, x * 2, color=color, alpha=alpha, lw=0, zorder=0, label=label)
    for f in (0.5, 2.0):
        ax.plot([lo, hi], [lo * f, hi * f], color="#C4C4C4", lw=0.5, ls=(0, (4, 2.5)), zorder=1)
    ax.plot([lo, hi], [lo, hi], color="#6E6E6E", lw=0.8, ls="--", zorder=2)


def log_ticks(ax, values=(2, 3, 5, 10, 20, 30, 50), both: bool = True) -> None:
    """Label a log axis at readable values rather than at decades.

    Over one and a half decades matplotlib's default leaves a single 10^1 on the axis, and the
    reader cannot estimate any quantity from the figure. Plain integers, not powers.
    """
    import matplotlib.ticker as mticker
    for axis in ((ax.xaxis, ax.yaxis) if both else (ax.xaxis,)):
        axis.set_major_locator(mticker.FixedLocator(values))
        axis.set_major_formatter(mticker.FixedFormatter([str(v) for v in values]))
        axis.set_minor_locator(mticker.NullLocator())


def parity_axes(ax, lo: float, hi: float, xlabel: str, ylabel: str) -> None:
    """Square, log-log, equal limits -- the only honest framing for a ratio-valued comparison."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def annotate_n(ax, n: int, where: str = "lower right") -> None:
    pos = {"lower right": (0.97, 0.03, "right", "bottom"),
           "lower left": (0.03, 0.03, "left", "bottom"),
           "upper left": (0.03, 0.97, "left", "top"),
           "upper right": (0.97, 0.97, "right", "top")}[where]
    ax.text(pos[0], pos[1], f"n = {n}", transform=ax.transAxes,
            ha=pos[2], va=pos[3], fontsize=7, color="#444444")


def check_cvd(colors: list) -> None:
    """Crude deuteranopia simulation, so a palette choice is checked rather than assumed.

    Not a substitute for a proper simulator, but it catches the common failure of two series that
    are obviously different to normal vision and identical without the green channel.
    """
    def sim(hexc):
        r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
        # Vienot-Brettel-Mollon deuteranope approximation in linear-ish RGB
        return (0.625 * r + 0.375 * g, 0.7 * r + 0.3 * g, b)
    out = [sim(c) for c in colors]
    print("  deuteranopia check (pairs closer than 0.10 are a problem):")
    worst = 1.0
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            d = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(out[i], out[j]))))
            worst = min(worst, d)
            if d < 0.10:
                print(f"    {colors[i]} vs {colors[j]}   separation {d:.3f}  TOO CLOSE")
    print(f"    minimum separation across all pairs: {worst:.3f}")
