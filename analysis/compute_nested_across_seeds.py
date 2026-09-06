"""Run the nested protocol across MODEL SEEDS, not just across chemistry splits.

THE LIMITATION THIS CLOSES. `nested_validation.py` resamples the chemistry split 200 times, but it
reads ONE fixed set of predictions -- `target_blind_test.csv`, produced with `random_seed=0`. So its
5th-95th range describes "which chemistries landed in the test half" and says nothing about model
seed. That matters here: across five seeds the per-compound median moves 38.5% to 42.0%, and seed 0
sits at the favourable end. Quoting a seed-0 nested figure as the headline repeats, in miniature,
the error that manufactured a 13.6 pp hurdle "gain" earlier in this project -- a single draw
reported as if it were the effect.

WHAT THIS DOES. For each of the five seed prediction files it runs the identical nested protocol
(200 cluster-level 50/50 splits, form/objective/threshold chosen on the design half and applied
frozen to the test half), then reports the distribution ACROSS seeds of the honest test-half median.
The headline becomes the median of those, with the seed range beside it.

Both thresholds are run because they answer different questions:
    threshold free  -- the published protocol, which selects a support filter on the design half
    threshold 0     -- unconditional, every half Heusler scored, no compound dropped
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, "c:/Users/ojasw/Documents/Data_collection_setup")
warnings.filterwarnings("ignore")
import io
import json
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import nested_validation as NV

SP = "paper/evidence"
SEEDS = (0, 1, 2, 3, 4)
REAL_THRESHOLDS = NV.THRESHOLDS


def run(blind_csv, thresholds, tag):
    NV.THRESHOLDS = thresholds
    buf = io.StringIO()
    with redirect_stdout(buf):
        NV.main(repeats=200, seed=0, klass="half", blind=blind_csv, tag=tag)
    out = f"data/exports/kappa_v2/nested_validation{tag}.json"
    return json.load(open(out))


def main() -> int:
    rows = []
    for mode, th in (("threshold free (published protocol)", REAL_THRESHOLDS),
                     ("threshold 0 (unconditional)", (0,))):
        print(f"\n=== {mode} ===")
        print(f"{'seed':>5}{'median':>9}{'within2x':>10}{'uncorrected':>13}"
              f"{'curse (pp)':>12}")
        per = []
        for s in SEEDS:
            f = f"{SP}/blind_d2_s{s}.csv"
            j = run(f, th, f"_seedscan_s{s}")
            h = j["honest"]
            per.append((h["median_ape"], 100 * h["within_2x"],
                        j["uncorrected_null"]["median_ape"], j["winners_curse_pp"]))
            print(f"{s:>5}{per[-1][0]:>8.1f}%{per[-1][1]:>9.1f}%{per[-1][2]:>12.1f}%"
                  f"{per[-1][3]:>12.1f}")
        a = np.array(per)
        print(f"{'MED':>5}{np.median(a[:,0]):>8.1f}%{np.median(a[:,1]):>9.1f}%"
              f"{np.median(a[:,2]):>12.1f}%{np.median(a[:,3]):>12.1f}")
        print(f"  seed range on the median: {a[:,0].min():.1f}-{a[:,0].max():.1f}% "
              f"({a[:,0].max()-a[:,0].min():.1f} pp)")
        rows.append(dict(mode=mode, median=float(np.median(a[:, 0])),
                         within_2x=float(np.median(a[:, 1])),
                         uncorrected=float(np.median(a[:, 2])),
                         curse_pp=float(np.median(a[:, 3])),
                         seed_lo=float(a[:, 0].min()), seed_hi=float(a[:, 0].max()),
                         n_seeds=len(SEEDS)))
    json.dump(rows, open(f"{SP}/nested_across_seeds.json", "w"), indent=2)
    print(f"\nwrote {SP}/nested_across_seeds.json")
    print("\nThe headline to quote is the MED row of the unconditional block, with the seed range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
