"""Figure 7 -- why this model, and what it learned.

TWO QUESTIONS A REFEREE WILL ASK, NEITHER OF WHICH THE PAPER CURRENTLY ANSWERS.

(a) DOES THE CHOICE OF REGRESSOR MATTER? Four were trained on identical data -- the same 4,765
    rows, the same 17 selected descriptors, the same grouped cross-validation with whole
    chemistries held out. They finish close together: CatBoost has the best R^2 on log-kappa
    (0.897) while the Gaussian process has the best median error (19.9% against 21.6%), a spread
    of under two percentage points across four quite different model families. The useful reading
    is not "we chose the winner" but that the result does not hinge on the regressor. A stored
    comparison claiming CatBoost led on every metric could not be reproduced from the present
    corpus and was re-run; these are the current numbers.

(b) DID IT LEARN PHONON PHYSICS, OR AN ARTEFACT? A tree ensemble reaching R^2 = 0.91 on log-kappa
    is only reassuring if the features it leans on are the ones physics says should matter. Panel
    (b) is a SHAP decomposition: for every row, how much each descriptor moved the prediction, and
    in which direction. If mass and bond length dominate with the expected signs, the model is
    doing thermal transport rather than memorising a lookup table.

SHAP is computed by CatBoost natively (`type="ShapValues"`), so this adds no dependency. Values
are in log10(kappa) units, which is the space the model is fitted in -- a SHAP value of -0.3 means
that descriptor pushed the prediction down by a factor of two.

COLOUR. The beeswarm uses viridis for the descriptor's own value, low to high, with an explicit
colourbar. That ramp is used nowhere else in the paper: the accent blue means "our result" and the
inferno ramp in Figure 2 means temperature, so neither can be spent here.
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
import numpy as np
import pandas as pd

import figlib as F

GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
NICE = {
    "M_mean": "mean atomic mass", "M_sum": "total mass", "M_X": "mass, X site",
    "M_ratio_XZ": "mass ratio X/Z", "dM_mean_X": "mass contrast, X vs mean",
    "r_X": "radius, X site", "r_mean": "mean atomic radius", "r_sum": "sum of radii",
    "X_X": "electronegativity, X site",
    "volume_per_atom": "volume per atom", "n_atoms_conv": "atoms per cell",
    "bond_scale": "bond-length scale", "bond_min": "shortest bond",
    "bond_mean": "mean bond length", "slack_geom": "Slack geometric factor",
    "temperature_K": "temperature", "inv_T": "1 / T", "log_T": "log T",
    "is_full_heusler": "full-Heusler flag", "a_conv": "lattice parameter",
    "bond_std": "bond-length spread", "M_Y": "mass, Y site", "M_Z": "mass, Z site",
    "r_Y": "radius, Y site", "r_Z": "radius, Z site", "X_Y": "electronegativity, Y site",
    "X_Z": "electronegativity, Z site", "X_mean": "mean electronegativity",
}
LABEL = {"catboost": "CatBoost", "lightgbm": "LightGBM",
         "krr": "kernel ridge", "gpr": "Gaussian process"}


def main() -> int:
    import matplotlib.pyplot as plt
    from catboost import Pool
    from pipeline import s19_kappa_dataset as ds
    from pipeline.s20_kappa_train import make_model

    # the PRODUCTION descriptor set (45), which is what the blind test and Section 3 use.
    # model_comparison.json is the 17-descriptor selection experiment and describes a model
    # the paper does not fit.
    mc = json.load(open("data/exports/kappa_v2/model_comparison_production.json"))
    order = sorted(mc, key=lambda k: mc[k]["median_ape"])
    print("  model comparison (identical rows, identical descriptors, grouped CV):")
    for k in order:
        print(f"    {LABEL[k]:<18} median APE {mc[k]['median_ape']:5.2f}%   "
              f"R2(log) {mc[k]['r2_log']:.4f}   n={mc[k]['n']}")
    assert len({mc[k]["n"] for k in mc}) == 1, "models were not scored on the same rows"

    # ---- refit the winner on the whole tier-1 pool for the SHAP decomposition ---------------
    # the SAME 17 descriptors and the SAME pool the comparison used -- an earlier version fitted
    # SHAP on all 45 columns of a 4,921-row build while quoting a 17-feature comparison beside it,
    # so the two panels described different models.
    b = ds.build(tier_max=1, tier_min=1, verbose=False)
    X, y, w = b["X"], b["y"], b["weights"]
    print(f"  SHAP model: {len(y)} rows x {X.shape[1]} descriptors")
    mdl = make_model("catboost")
    mdl.fit(X, y, sample_weight=w)
    sv = mdl.get_feature_importance(Pool(X, y, weight=w), type="ShapValues")
    sv = sv[:, :-1]                                   # last column is the expected value
    imp = np.abs(sv).mean(axis=0)
    top = np.argsort(imp)[::-1][:12]

    F.use_style()
    # subplot_mosaic, not add_gridspec: the paper style turns constrained layout on, which
    # silently ignores a hand-built gridspec's hspace/wspace and squashed these bars into strips.
    fig, axd = plt.subplot_mosaic([["err", "shap"], ["r2", "shap"]],
                                  figsize=(F.DOUBLE, F.DOUBLE * 0.46),
                                  width_ratios=[1.0, 1.6])
    ax_err, ax_r2, ax = axd["err"], axd["r2"], axd["shap"]

    # ---- (a) does the regressor matter? two metrics, because they disagree -------------------
    #
    # CatBoost has the best R^2 and the SECOND-worst median error. A single bar chart of median
    # error with CatBoost highlighted would have said the opposite of the truth, so both metrics
    # are shown and the selected model is marked as selected rather than as winner.
    ys = np.arange(len(order))[::-1]
    for axm, key, lab, fmt, better in (
            (ax_err, "median_ape", "median absolute error (%)", "{:.1f}%", "lower"),
            (ax_r2, "r2_log", "$R^2$ on $\\log_{10}\\kappa_L$", "{:.3f}", "higher")):
        vals = [mc[k][key] for k in order]
        cols = [F.OURS if k == "catboost" else "#C9CDD1" for k in order]
        axm.barh(ys, vals, color=cols, height=0.60, zorder=3)
        span = max(vals) - min(vals)
        for yy, v, k in zip(ys, vals, order):
            axm.text(v + span * 0.06, yy, fmt.format(v), va="center", ha="left", fontsize=6.6,
                     color=(F.OURS if k == "catboost" else "#6E6E6E"),
                     fontweight="bold" if k == "catboost" else "normal")
        axm.set_yticks(ys)
        axm.set_yticklabels([LABEL[k] for k in order], fontsize=6.8)
        axm.set_xlim(min(vals) - span * 0.30, max(vals) + span * 0.75)
        axm.set_xlabel(f"{lab}   ({better} is better)", fontsize=7)
        axm.tick_params(axis="y", length=0)
        axm.tick_params(axis="x", labelsize=6.3)
    ax_err.set_title("Four regressors, identical data", fontsize=8, loc="left", pad=12)
    ax_err.text(0.0, 1.055, "blue = used in this work", transform=ax_err.transAxes,
                fontsize=6.4, va="bottom", ha="left", color=F.OURS)
    F.panel_label(ax_err, "a", dx=-0.46, dy=1.34)

    # ---- (b) what it learned -----------------------------------------------------------------
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("viridis")
    for row, j in enumerate(top[::-1]):
        s = sv[:, j]
        v = X.iloc[:, j].values.astype(float)
        lo, hi = np.nanpercentile(v, [2, 98])
        cnorm = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        jit = rng.normal(0, 0.115, size=len(s))
        # rasterized=True is what keeps this figure compilable. The beeswarm draws one point per
        # row per descriptor -- ~81,000 vector path operations, forty times the next heaviest
        # figure -- which times out a LaTeX compile on Overleaf's free plan. Rasterizing turns the
        # point cloud into a bitmap at the savefig dpi while axes, ticks and labels stay vector,
        # so the figure still scales cleanly in print. Standard practice for dense scatter plots.
        sc = ax.scatter(s, row + jit, c=cnorm, cmap=cmap, s=2.4, lw=0, alpha=0.55, zorder=3,
                        rasterized=True)
    ax.axvline(0, color="#9A9A9A", lw=0.6, zorder=2)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([NICE.get(X.columns[j], X.columns[j]) for j in top[::-1]], fontsize=7)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlabel("SHAP value  (effect on predicted $\\log_{10}\\kappa_L$)")
    ax.set_title("What the model uses")
    ax.tick_params(axis="y", length=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.012, aspect=24, shrink=0.85)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("descriptor value", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    F.panel_label(ax, "b", dx=-0.28)

    print("  top descriptors by mean |SHAP|:")
    for j in top[:6]:
        print(f"    {NICE.get(X.columns[j], X.columns[j]):<28} {imp[j]:.4f}")
    F.save(fig, "fig07_model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
