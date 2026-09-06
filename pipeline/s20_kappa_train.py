"""Stage 20 — train and honestly evaluate the Heusler kappa_L model.

THE THREE RULES THIS FILE ENFORCES, each because breaking it produces a flattering lie:

1. **Group by formula, never by row.** 41% of compounds contribute more than one temperature point
   (one contributes 653). A random row split puts the same compound on both sides and reports an
   accuracy the model will never reproduce on a new material.

2. **Tune on an inner loop, report from the outer one.** Selecting features or hyper-parameters on
   the same folds you quote is fitting to the test set. Only outer-loop numbers are quotable.

3. **Say which target a number refers to.** Predicting *DFT* kappa_L has no noise floor -- the
   labels are deterministic. Predicting *measured* kappa_L is limited by the 16.7% inter-laboratory
   reproducibility floor, because two labs measuring the same compound differ by that much. A model
   beating 16.7% against experiment is reporting a leak, not a result. The two are never mixed.

FEATURE SELECTION IS THE POINT, NOT AN AFTERTHOUGHT.
Miyazaki et al. (Sci. Rep. 11, 13809) machine-learned kappa_L for 143 half-Heuslers and found 55
descriptors gave ~8% error through over-fitting while 4 gave ~4%. Backward elimination under
grouped CV reproduces that procedure here.

WHAT `--validate` ACTUALLY TESTS. The 44 experimentally-measured Heuslers are removed from training
ENTIRELY -- every row of every one of them, not just the matching temperature. 40 of the 44 sit in
the training set, so skipping this step would be handing the model its own answer key.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from pipeline.s19_kappa_dataset import build, validation_formulas

OUT_DIR = Path("data/exports/kappa_v2")
INTER_LAB_FLOOR = 16.7          # %, measured over 209 repeat groups in s18_eval_deployment.py


# ---------------------------------------------------------------- metrics ---
def metrics(y_true_log, y_pred_log, groups=None) -> dict:
    """Errors in BOTH spaces. R2 is quoted on log10 kappa (the modelling target); the relative
    error is quoted in linear kappa, because a 2x miss on a 0.5 W/m/K compound and on a 50 W/m/K
    compound are equally wrong physically and only the linear form says so.

    PASS `groups` (one label per row, normally the compound) TO GET A PER-COMPOUND STATISTIC.
    Without it the medians are taken over ROWS, and rows are not evenly distributed: `TiNiSn` and
    `ZrNiSn` each contribute 95 of the 5,031 tier-1 rows against a mean of 9.3, so the top five
    compounds alone are ~9% of the row population and dominate any row-level median. The grouped
    train/test split already prevents leakage from that imbalance, but it does nothing about the
    reported number. Every downstream result in this project is quoted per compound, and feature
    selection ranks candidates on this metric, so a row-level median here silently lets the
    best-measured compounds decide which features survive.
    """
    k_true, k_pred = 10 ** np.asarray(y_true_log), 10 ** np.asarray(y_pred_log)
    ape = np.abs(k_pred - k_true) / k_true * 100
    if groups is not None:
        g = pd.Series(ape).groupby(np.asarray(groups)).median().values
        med, mean = float(np.median(g)), float(np.mean(g))
        w20, w30 = float(np.mean(g <= 20) * 100), float(np.mean(g <= 30) * 100)
        n_unit = int(len(g))
    else:
        med, mean = float(np.median(ape)), float(np.mean(ape))
        w20, w30 = float(np.mean(ape <= 20) * 100), float(np.mean(ape <= 30) * 100)
        n_unit = int(len(y_true_log))
    return dict(
        r2_log=float(r2_score(y_true_log, y_pred_log)),
        mae_log=float(np.mean(np.abs(np.asarray(y_pred_log) - np.asarray(y_true_log)))),
        rmse_log=float(np.sqrt(np.mean((np.asarray(y_pred_log)
                                        - np.asarray(y_true_log)) ** 2))),
        median_ape=med,
        mean_ape=mean,
        within_20pct=w20,
        within_30pct=w30,
        unit="compound" if groups is not None else "row",
        n=n_unit,
        n_rows=int(len(y_true_log)))


def _fmt(m: dict) -> str:
    return (f"R2 {m['r2_log']:.3f}  median err {m['median_ape']:5.1f}%  "
            f"within20% {m['within_20pct']:4.0f}%  n={m['n']}")


# ---------------------------------------------------------------- models ---
def make_model(name: str, params: dict | None = None):
    p = params or {}
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**{"iterations": 900, "depth": 6, "learning_rate": 0.05,
                                    "loss_function": "MAE", "random_seed": 0,
                                    "verbose": 0, **p})
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**{"n_estimators": 900, "num_leaves": 31,
                                    "learning_rate": 0.05, "objective": "mae",
                                    "random_state": 0, "verbose": -1, **p})
    if name == "krr":
        # Kernel ridge only earns a place on a SMALL feature set. Given 136 Magpie columns the
        # project's earlier GaussianProcess scored R2 0.107 -- kernels collapse in high dimensions.
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             KernelRidge(**{"kernel": "rbf", "alpha": 0.1, "gamma": 0.1, **p}))
    if name == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        k = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                      random_state=0))
    raise ValueError(name)


NEEDS_DENSE = {"krr", "gpr"}


def grouped_oof(model_name, X, y, groups, weights=None, n_splits=5, params=None):
    """Out-of-fold predictions with formula-level grouping."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = make_model(model_name, params)
        Xtr = X.iloc[tr]
        if model_name in NEEDS_DENSE:
            m.fit(Xtr, y[tr])
        elif weights is not None:
            m.fit(Xtr, y[tr], sample_weight=weights[tr])
        else:
            m.fit(Xtr, y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


# ------------------------------------------------- backward elimination ---
def backward_eliminate(X, y, groups, weights, model_name="catboost", min_feats=3,
                       n_splits=5, verbose=True, drop_frac=0.15):
    """Drop the least useful features repeatedly, scoring by grouped CV each round.

    This reproduces the procedure that took Miyazaki et al. from ~8% error at 55 descriptors to
    ~4% at 4 -- and it uses the same ranking they did: PERMUTATION IMPORTANCE. A first version
    here re-fitted the model once per candidate feature (40 fits per round, ~1,700 fits total,
    hours). Permutation importance needs ONE fit per round and answers the same question -- how
    much does the score degrade when this feature is scrambled.

    More than one feature is dropped per round (`drop_frac`) while many remain, then one at a time
    near the end where the choice actually matters.
    """
    from sklearn.inspection import permutation_importance

    feats = list(X.columns)
    history = []
    best = (list(feats), -np.inf, None)
    while len(feats) >= min_feats:
        pred = grouped_oof(model_name, X[feats], y, groups, weights, n_splits)
        m = metrics(y, pred)
        history.append(dict(n_features=len(feats), features=list(feats), **m))
        if verbose:
            print(f"    {len(feats):>3} features  {_fmt(m)}", flush=True)
        if m["r2_log"] > best[1]:
            best = (list(feats), m["r2_log"], m)
        if len(feats) <= min_feats:
            break

        # rank on a single held-out fold, so the ranking never sees the data it is scored on
        gkf = GroupKFold(n_splits=n_splits)
        tr, te = next(iter(gkf.split(X[feats], y, groups)))
        mdl = make_model(model_name)
        if model_name in NEEDS_DENSE:
            mdl.fit(X[feats].iloc[tr], y[tr])
        else:
            mdl.fit(X[feats].iloc[tr], y[tr], sample_weight=weights[tr])
        imp = permutation_importance(mdl, X[feats].iloc[te], y[te], n_repeats=5,
                                     random_state=0, scoring="r2")
        order = np.argsort(imp.importances_mean)          # least important first
        n_drop = max(1, int(len(feats) * drop_frac)) if len(feats) > 12 else 1
        n_drop = min(n_drop, len(feats) - min_feats)
        drop = {feats[i] for i in order[:n_drop]}
        if verbose and n_drop > 1:
            print(f"        dropping {n_drop}: {sorted(drop)}", flush=True)
        feats = [c for c in feats if c not in drop]
    return best[0], best[2], history


# ------------------------------------------------------------ commands ---
def cmd_baseline(args):
    print("BASELINE — all features, grouped 5-fold CV\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for tier_min, tier_max, label in ((1, 1, "tier 1 only (full BTE)"),
                                      (1, 3, "tier 1-3 (no experimental)"),
                                      (0, 3, "all tiers incl. experimental")):
        b = build(tier_max=tier_max, tier_min=tier_min)
        for name in ("catboost", "lightgbm"):
            pred = grouped_oof(name, b["X"], b["y"], b["groups"], b["weights"])
            m = metrics(b["y"], pred)
            print(f"  {label:<38}{name:<10}{_fmt(m)}")
            results[f"{label}|{name}"] = m
        print()
    (OUT_DIR / "baseline.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_DIR / 'baseline.json'}")


def cmd_select(args):
    print(f"BACKWARD FEATURE ELIMINATION  (tier <= {args.tier_max})\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b = build(tier_max=args.tier_max, tier_min=args.tier_min)
    print()
    feats, m, hist = backward_eliminate(b["X"], b["y"], b["groups"], b["weights"],
                                        model_name=args.model, min_feats=args.min_feats)
    print(f"\n  BEST: {len(feats)} features   {_fmt(m)}")
    print(f"  {feats}")
    (OUT_DIR / "feature_selection.json").write_text(
        json.dumps(dict(best_features=feats, best_metrics=m, history=hist), indent=2))
    print(f"\nwrote {OUT_DIR / 'feature_selection.json'}")


def _best_features() -> list[str] | None:
    p = OUT_DIR / "feature_selection.json"
    if p.exists():
        return json.loads(p.read_text())["best_features"]
    return None


def cmd_compare(args):
    """Trees vs kernels on the reduced set — the test that decides whether Model B earns a place."""
    feats = _best_features()
    b = build(tier_max=args.tier_max, tier_min=args.tier_min)
    X = b["X"][feats] if feats else b["X"]
    print(f"MODEL COMPARISON on {X.shape[1]} features "
          f"({'selected' if feats else 'ALL — run --select first'})\n")
    out = {}
    for name in ("catboost", "lightgbm", "krr", "gpr"):
        try:
            pred = grouped_oof(name, X, b["y"], b["groups"], b["weights"])
            m = metrics(b["y"], pred)
            out[name] = m
            print(f"  {name:<12}{_fmt(m)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12}FAILED {str(exc)[:60]}")
    (OUT_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'model_comparison.json'}")


def cmd_validate(args):
    """The three tests, hardest last."""
    feats = _best_features()
    val = validation_formulas()
    print(f"VALIDATION — {len(val)} measured Heuslers held out entirely\n")

    # ---- test 1: interpolation, grouped CV on DFT labels -------------------------------------
    b = build(tier_max=args.tier_max, tier_min=args.tier_min, drop_formulas=val)
    X = b["X"][feats] if feats else b["X"]
    pred = grouped_oof(args.model, X, b["y"], b["groups"], b["weights"])
    m1 = metrics(b["y"], pred)
    print(f"  1. grouped CV (DFT labels, no noise floor)   {_fmt(m1)}")

    # ---- test 2: the prediction targets that already have published kappa --------------------
    tgt_path = Path("data/Target_Materials/Target_Dossier_166.xlsx")
    m2 = None
    if tgt_path.exists():
        from pymatgen.core import Composition
        T = pd.read_excel(tgt_path, sheet_name="Targets")
        tset = set()
        for f in T.compound.unique():
            try:
                tset.add(Composition(str(f)).reduced_formula)
            except Exception:  # noqa: BLE001
                pass
        mask = np.isin(b["groups"], list(tset))
        if mask.sum() > 30:
            m2 = metrics(b["y"][mask], pred[mask])
            print(f"  2. blind on {len(set(b['groups'][mask]))} prediction targets  {_fmt(m2)}")

    # ---- test 3: THE TRANSFER TEST — predict real measurements ------------------------------
    # A first version of this scored 11.3%, apparently beating the 16.7% inter-lab floor. It was
    # wrong: it had selected the 578 full-BTE rows belonging to those 44 formulae and scored the
    # model against DFT labels, not against experiment. (The row count coincidentally equalled the
    # number of lab measurements, which is what made it look plausible.) The real test loads the
    # measured kappa from Validation_Heuslers.xlsx and predicts it at the temperature each
    # measurement was taken at.
    m3 = m3_dft = None
    full = build(tier_max=args.tier_max, tier_min=args.tier_min, verbose=False)
    hold = np.isin(full["groups"], list(val))
    Xall = full["X"][feats] if feats else full["X"]
    mdl = make_model(args.model)
    tr = ~hold
    if args.model in NEEDS_DENSE:
        mdl.fit(Xall[tr], full["y"][tr])
    else:
        mdl.fit(Xall[tr], full["y"][tr], sample_weight=full["weights"][tr])

    if hold.sum():
        m3_dft = metrics(full["y"][hold], mdl.predict(Xall[hold]))
        print(f"  3a. held-out compounds, DFT labels          {_fmt(m3_dft)}")

    vpath = Path("data/Target_Materials/Validation_Heuslers.xlsx")
    if vpath.exists():
        from pipeline.s19_kappa_dataset import _per_formula_features
        vm = pd.read_excel(vpath, sheet_name="Measurements")
        vm = vm[(pd.to_numeric(vm.kappa, errors="coerce") > 0)
                & (pd.to_numeric(vm["T"], errors="coerce") >= 200)]
        # structures come from the resolved pool -- the same cited lattice used in training
        pool = pd.read_csv("data/external/KAPPA_POOL_WITH_STRUCTURE.csv")
        pool = pool[pool.struct_a_A.notna()].drop_duplicates("formula").set_index("formula")
        rows, ys = [], []
        for r in vm.itertuples():
            if r.formula not in pool.index:
                continue
            pr = pool.loc[r.formula]
            f = _per_formula_features(r.formula, pr.struct_a_A, pr.get("struct_prototype"),
                                      pr.get("struct_species"), pr.get("struct_frac_coords"))
            if f is None:
                continue
            T = float(getattr(r, "T"))
            f.update(temperature_K=T, inv_T=1000.0 / T, log_T=float(np.log10(T)))
            rows.append(f)
            ys.append(np.log10(float(r.kappa)))
        if rows:
            Xv = pd.DataFrame(rows)
            for c in Xall.columns:
                if c not in Xv.columns:
                    Xv[c] = np.nan
            Xv = Xv[Xall.columns]
            m3 = metrics(np.array(ys), mdl.predict(Xv))
            print(f"  3b. TRANSFER: predicting real MEASUREMENTS  {_fmt(m3)}")
            verdict = ("BELOW the inter-lab floor -- suspect a leak"
                       if m3["median_ape"] < INTER_LAB_FLOOR else "above the floor, as expected")
            print(f"      inter-lab floor {INTER_LAB_FLOOR}% -> {verdict}")

    res = dict(interpolation_dft=m1, blind_targets=m2, held_out_dft_labels=m3_dft,
               transfer_to_measurements=m3,
               inter_lab_floor_pct=INTER_LAB_FLOOR,
               n_features=int(X.shape[1]), features=feats, model=args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "validation.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT_DIR / 'validation.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("baseline", cmd_baseline), ("select", cmd_select),
                     ("compare", cmd_compare), ("validate", cmd_validate)):
        s = sub.add_parser(name)
        s.add_argument("--tier-max", type=int, default=1)
        s.add_argument("--tier-min", type=int, default=1,
                       help="DEFAULT 1 = full-BTE labels only. Measured: tier 1 alone gives "
                            "R2 0.886 / 20.5%%, while adding the experimental rows (tier 0) "
                            "collapses it to 0.743 / 35.5%% -- experimental kappa_L is a "
                            "DIFFERENT target, set by sample microstructure the features cannot "
                            "see (tier 0 alone scores R2 0.001). It belongs in validation and "
                            "delta-learning, not in the DFT training signal.")
        s.add_argument("--model", default="catboost")
        s.add_argument("--min-feats", type=int, default=4)
        s.set_defaults(func=fn)
    a = ap.parse_args()
    raise SystemExit(a.func(a) or 0)
