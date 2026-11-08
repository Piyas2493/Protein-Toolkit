# Pose-label validation against crystal poses

30 complexes selected by rule from RCSB, 243 docked poses; 13% of poses are within 2 A of the crystal pose; Vina's pose 1 is right for 50% of complexes; 21 of 30 complexes have at least one correct pose.

| Scorer | AUC over all poses | AUC on each complex's pose 1 |
|---|---|---|
| vina score | 0.71 (0.62-0.84) | 0.80 |
| label (legacy) | 0.70 (0.62-0.78) | 0.73 |
| label (fixed) | 0.63 (0.55-0.73) | 0.72 |
| fitted model (leave-one-complex-out) | 0.82 (0.73-0.90) | 0.78 |
| fitted model, legacy nonpolar count | 0.82 | n/a |

AUC 0.5 = no information, 1.0 = perfect separation of correct from incorrect poses.

| Label | Version | Poses | Share correct |
|---|---|---|---|
| STRONG | legacy | 56 | 27% |
| MODERATE | legacy | 99 | 12% |
| WEAK | legacy | 88 | 5% |
| STRONG | fixed | 85 | 19% |
| MODERATE | fixed | 83 | 13% |
| WEAK | fixed | 75 | 5% |

Fitted-model probability bands (leave-one-complex-out):

| Band | Poses | Share correct |
|---|---|---|
| (-0.001, 0.25] | 206 | 7% |
| (0.25, 0.5] | 22 | 32% |
| (0.5, 0.75] | 11 | 55% |
| (0.75, 1.0] | 4 | 100% |

Calibration on the pose people actually see (each complex's pose 1, leave-one-complex-out):

| Fitted band | Complexes | Share correct |
|---|---|---|
| (-0.001, 0.25] | 14 | 21% |
| (0.25, 0.5] | 7 | 57% |
| (0.5, 0.75] | 5 | 80% |
| (0.75, 1.0] | 4 | 100% |

Decision rule (fixed in advance): adopt the fitted model only if it beats the rule-based label by more than 0.05 AUC. Difference: +0.185 (bootstrap 95% interval +0.09 to +0.28) -> adopt the fitted model.
