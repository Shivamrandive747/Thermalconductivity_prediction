# What was tried and did not work

Five modifications to the method were implemented in full and rejected on held-out evidence. They
are recorded here because the next person to work on this problem will think of them too, and
because a positive result is easier to judge beside the alternatives that were tried.

Section 3.4 of the paper states these with their statistics. The code is in `analysis/`.

| intervention | result | script |
|---|---|---|
| **Hurdle model** — treat low- and high-conductivity compounds with separate models | Halves the low-κ error on *calculated* labels (p = 0.0005). On the measured compounds: **−1.3 pp, p = 0.84.** The gain does not survive contact with experiment. | — |
| **Magnitude term** — add a κ^β factor to the transfer function | Selected on the design half in **47 of 60** splits, then lost on the held-out half. A textbook winner's-curse result. | `analysis/fit_magnitude_calibration.py` |
| **Add experimental values to training** | +5.5 pp at p = 0.10, and *worse* on the transition-metal subset. | `analysis/run_tier0_experiment.py` |
| **Add doped-composition data** | No gain at all (p ≥ 0.10). | `analysis/run_doped_experiment.py` |
| **Per-family calibration constants** | The most interesting failure. Families genuinely do differ in their raw calculation-to-measurement ratio — but the difference does not transfer. Leave-one-chemistry-out gives **31.1 % against 31.0 %** for one global constant (p = 0.22, better for only 17 of 38 compounds). With two to four measured members per family, the apparent differences are noise. | `analysis/compute_results_indomain.py` |

## Two methodological traps this project fell into

Both cost real time, and both are easy to repeat.

**Fitting one objective and reporting another.** The calibration is fitted to minimise the median
of *per-compound* errors, which is the quantity reported. Fitting per *row* instead lets a single
heavily sampled compound dominate — `TiNiSn` and `ZrNiSn` alone contribute 753 and 644 of the 2,426
matched rows. Getting this wrong once cost a spurious 9 percentage points.

**Letting a compound serve as its own evidence.** An early version of the family-anchor rule
counted the compound being scored among its own family's supporting measurements. That selects for
the conclusion it is testing. Corrected to a genuine leave-one-out count, the evidence ran the
other way and the threshold stayed at two anchors rather than being relaxed to one.

## One result that is easy to misread as a failure

A **family power law** — fit a curve to the measured κ(T) of the held-out compound's own bonding
family and use it — achieves 28.2 % median error against our 31.0 % at the same seed. On absolute
accuracy it beats the method, and the paper says so in bold.

It is still not a substitute, for two reasons. It is undefined for 7 of the 38 in-domain compounds,
whose families hold too few measured members. And it returns one curve per family, so it cannot
order compounds *within* a family — the within-family rank correlation is −0.90 for the family
baseline against +0.50 for the calibrated calculation. Choosing between three members of one
substitution series is exactly the task a screening user faces, and the family baseline is
uninformative for it by construction.
