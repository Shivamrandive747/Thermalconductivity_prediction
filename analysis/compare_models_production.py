"""Compare the four regressors on the descriptor set the paper actually uses.

WHY THIS EXISTS. `s20_kappa_train compare` restricts to the 17 descriptors chosen by the feature
selection run, but the production model -- the one that generates every k_pred in the blind test,
and the one Section 3 describes -- is fitted on all 45. So the stored comparison described a model
the paper does not use, and quoting it beside a 45-feature SHAP decomposition would have put two
different models in one figure.

It is worth recording why the 17-descriptor selection was never adopted, because the selection
history looks like it should have been (R2 0.910 / 17.7% at 17 features against 0.886 / 20.5% at
40). The curve is not monotonic and not smooth: it reads 17.4% at 8 features, 22.2% at 7, 17.7% at
11 and 24.0% at 5. Differences of that size appearing and disappearing as one descriptor is added
or removed are selection noise, not signal, and picking the argmin of such a curve is a way of
overfitting the selection itself. Keeping all 45 is the conservative choice.

Writes model_comparison_production.json alongside the existing file rather than overwriting it,
so both experiments remain on record.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from pipeline.s19_kappa_dataset import build
from pipeline.s20_kappa_train import grouped_oof, metrics

OUT = Path("data/exports/kappa_v2/model_comparison_production.json")


def main() -> int:
    b = build(tier_max=1, tier_min=1, verbose=False)
    X, y, g, w = b["X"], b["y"], b["groups"], b["weights"]
    print(f"MODEL COMPARISON on the production descriptor set: {X.shape[1]} descriptors, "
          f"{len(y)} rows, {len(set(g))} groups\n")
    out = {}
    for name in ("catboost", "lightgbm", "krr", "gpr"):
        try:
            m = metrics(y, grouped_oof(name, X, y, g, w))
            out[name] = m
            print(f"  {name:<12}R2 {m['r2_log']:.4f}   median err {m['median_ape']:5.2f}%   "
                  f"within20% {m['within_20pct']:5.1f}%   n={m['n']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12}FAILED {str(exc)[:70]}")
    assert len({v["n"] for v in out.values()}) == 1, "models scored on different rows"
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    best_r2 = max(out, key=lambda k: out[k]["r2_log"])
    best_ape = min(out, key=lambda k: out[k]["median_ape"])
    print(f"\n  best R2        : {best_r2}")
    print(f"  best median err: {best_ape}")
    if best_r2 != best_ape:
        print("  -> the two metrics disagree; the figure must show both rather than declare "
              "a winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
